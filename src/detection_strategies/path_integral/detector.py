import torch
import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from detection_strategies.dependence_detector import DependenceDetector
from detection_strategies.path_integral.utils import (
    PromptResponseDataset, get_dataloader, get_mixed_dataloader, create_redpajama_dataloader
)
from detection_strategies.path_integral.path_optimizer import (
    create_polygonal_chain, optimize_polygonal_chain, compute_path_integral
)
from detection_strategies.path_integral.model_trainer import train_model
from detection_strategies.path_integral.evaluator import evaluate_model, log_and_save_metrics

class PathIntegralDetector(DependenceDetector):
    """
    Detector that measures dependence using path integrals.
    
    The path integral approach measures how much the model's behavior changes
    when interpolating between parameters optimized for different tasks.
    """

    def __init__(self, eval_steps: int = 100, num_bends: int = 3, 
                 path_optim_steps: int = 20, path_optim_lr: float = 1e-4, 
                 retrain: bool = False, lora_r: int = 64, 
                 lora_alpha: int = 8, lora_dropout: float = 0.05, 
                 *args, **kwargs):
        self.eval_steps = eval_steps
        self.num_bends = num_bends
        self.path_optim_steps = path_optim_steps
        self.path_optim_lr = path_optim_lr
        self.retrain = retrain  # Flag to force retraining
        
        # LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        super().__init__(*args, **kwargs)
    
        self.device_str = next(self.model.parameters()).device

    def get_task_data(self, prompts: List[str], responses: List[str]) -> PromptResponseDataset:
        """Create a dataset for the task data."""
        return PromptResponseDataset(prompts, responses, self.tokenizer)

    def evaluate(self, normal_dataloader: DataLoader, oversight_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance on normal and oversight evaluation dataloaders."""
        return evaluate_model(self.model, normal_dataloader, oversight_dataloader, self.device_str)

    def compute_dependence(self, normal_data: Dataset, oversight_data: Dataset) -> Dict[str, float]:
        """
        Compute path integral metrics between two tasks using their datasets.
        
        Args:
            normal_data: Dataset for normal task
            oversight_data: Dataset for oversight task
            
        Returns:
            Dictionary of path integral metrics
        """
        # Define checkpoint paths
        checkpoint_paths = {
            "original": "path_integral_weights/original_model.pt",
            "normal": "path_integral_weights/normal_finetuned.pt",
            "normal_to_mixed": "path_integral_weights/normal_to_mixed.pt",
            "oversight": "path_integral_weights/oversight_finetuned.pt",
            "oversight_to_mixed": "path_integral_weights/oversight_to_mixed.pt"
        }
        
        chain_paths = {
            "original_to_normal": "path_integral_chains/optimized_path_original_to_normal.pt",
            "normal_to_mixed": "path_integral_chains/optimized_path_normal_to_mixed.pt"
        }
        
        # Determine if we should skip training or chain optimization
        skip_finetuning, skip_chains = self._check_existing_files(checkpoint_paths, chain_paths)
        
        # If we're skipping everything, just load existing data and compute path integrals
        if skip_finetuning and skip_chains:
            return self._compute_metrics_from_disk(chain_paths, checkpoint_paths, normal_data, oversight_data)
        
        # Initialize train/eval data
        normal_train_data, oversight_train_data, normal_eval_data, oversight_eval_data = self._prepare_datasets(
            normal_data, oversight_data
        )
        
        # Create dataloaders
        normal_dataloader = get_dataloader(normal_train_data)
        oversight_dataloader = get_dataloader(oversight_train_data)
        mixed_dataloader = get_mixed_dataloader(normal_train_data, oversight_train_data)
        
        # Load or train models
        model_states = self._get_model_states(
            skip_finetuning, checkpoint_paths, 
            normal_dataloader, oversight_dataloader, mixed_dataloader,
            normal_eval_data, oversight_eval_data
        )
        
        # Create and optimize polygonal chains
        optimized_paths = self._create_optimize_paths(
            skip_chains, model_states, normal_dataloader, 
            mixed_dataloader
        )
        
        # Compute path integrals
        path_integrals = self._compute_path_integrals(
            optimized_paths, normal_dataloader, mixed_dataloader, 
            model_states
        )
        
        # Return results
        return self._prepare_results(path_integrals, checkpoint_paths, chain_paths)
    
    # Helper methods (private)
    def _check_existing_files(self, checkpoint_paths, chain_paths):
        """Check if checkpoints and chains exist and determine what to skip."""
        # Check if checkpoints and chains exist
        checkpoints_exist = all(os.path.exists(path) for path in checkpoint_paths.values())
        chains_exist = all(os.path.exists(path) for path in chain_paths.values())
        
        # Determine if we should skip training or chain optimization
        if self.retrain:
            print("--retrain flag set. Retraining from scratch...")
            skip_finetuning = False
            skip_chains = False
        elif checkpoints_exist and chains_exist:
            print("Found existing checkpoints and chains. Loading instead of retraining...")
            skip_finetuning = True
            skip_chains = True
        elif checkpoints_exist:
            print("Found existing model checkpoints but not optimized chains. Skipping finetuning and creating chains...")
            skip_finetuning = True
            skip_chains = False
        else:
            print("Checkpoints missing. Training from scratch...")
            skip_finetuning = False
            skip_chains = False
            
        return skip_finetuning, skip_chains
        
    def _compute_metrics_from_disk(self, chain_paths, checkpoint_paths, normal_data, oversight_data):
        """Load existing chains and compute metrics without retraining."""
        # Load chains and model weights
        optimized_path_original_to_normal = torch.load(chain_paths["original_to_normal"], map_location="cpu")
        optimized_path_normal_to_mixed = torch.load(chain_paths["normal_to_mixed"], map_location="cpu")
        
        original_model_state = torch.load(checkpoint_paths["original"], map_location="cpu")
        normal_model_state = torch.load(checkpoint_paths["normal"], map_location="cpu")
        
        # Create dataloaders for path integral computation
        normal_dataloader = get_dataloader(normal_data)
        mixed_dataloader = get_mixed_dataloader(normal_data, oversight_data)
        
        # Ensure we're using a LoRA model
        lora_model = self._get_lora_model()
        
        # Compute path integrals on the loaded chains
        print("Computing path integral: Original -> Normal")
        path_integral_original_to_normal = compute_path_integral(
            model=lora_model,  # Use the LoRA model
            polygonal_chain=optimized_path_original_to_normal,
            target_dataloader=normal_dataloader,
            prior_params=original_model_state,
            prior_weight=0.1,
            num_samples=20,  # Adjust as needed
            device_str=self.device_str
        )
        
        print("Computing path integral: Normal -> Normal+Oversight")
        path_integral_normal_to_mixed = compute_path_integral(
            model=lora_model,  # Use the LoRA model
            polygonal_chain=optimized_path_normal_to_mixed,
            target_dataloader=mixed_dataloader,
            prior_params=normal_model_state,
            prior_weight=0.1,
            num_samples=20,  # Adjust as needed
            device_str=self.device_str
        )
        
        return {
            "status": "loaded_from_disk",
            "weights_loaded": True,
            "normal_finetuned": checkpoint_paths["normal"],
            "normal_to_mixed": checkpoint_paths["normal_to_mixed"],
            "oversight_finetuned": checkpoint_paths["oversight"],
            "oversight_to_mixed": checkpoint_paths["oversight_to_mixed"],
            "optimized_path_original_to_normal": chain_paths["original_to_normal"],
            "optimized_path_normal_to_mixed": chain_paths["normal_to_mixed"],
            "path_integral_original_to_normal": path_integral_original_to_normal["path_integral_value"],
            "path_integral_normal_to_mixed": path_integral_normal_to_mixed["path_integral_value"],
            "path_dependence_ratio": path_integral_normal_to_mixed["path_integral_value"] / max(path_integral_original_to_normal["path_integral_value"], 1e-8)
        }
        
    def _prepare_datasets(self, normal_data, oversight_data):
        """Split data into train and evaluation sets."""
        if self.train_size is None:
            train_size = len(normal_data)
        else:
            train_size = min(self.train_size, len(normal_data))

        normal_train_data = normal_data.select(slice(0, train_size))
        oversight_train_data = oversight_data.select(slice(0, train_size))

        normal_eval_data = normal_data.select(slice(train_size, len(normal_data)))
        oversight_eval_data = oversight_data.select(slice(train_size, len(oversight_data)))
        
        return normal_train_data, oversight_train_data, normal_eval_data, oversight_eval_data
        
        
    def _get_model_states(self, skip_finetuning, checkpoint_paths, normal_dataloader, 
                         oversight_dataloader, mixed_dataloader, 
                         normal_eval_data, oversight_eval_data):
        """Get model states either by loading or training."""
        if skip_finetuning:
            print("Loading model weights from disk...")
            original_model_state = torch.load(checkpoint_paths["original"], map_location="cpu")
            normal_model_state = torch.load(checkpoint_paths["normal"], map_location="cpu")
            normal_to_mixed_state = torch.load(checkpoint_paths["normal_to_mixed"], map_location="cpu")
            oversight_model_state = torch.load(checkpoint_paths["oversight"], map_location="cpu")
            oversight_to_mixed_state = torch.load(checkpoint_paths["oversight_to_mixed"], map_location="cpu")
        else:
            train_size = len(normal_dataloader.dataset)
            # Apply LoRA for the first training sequence
            from peft import LoraConfig, get_peft_model, TaskType
            lora_model = self._get_lora_model()

            # Save original model weights
            print("Saving original model weights...")
            original_model_state = {name: param.detach().clone().cpu() 
                                    for name, param in lora_model.named_parameters() if param.requires_grad}
            
            # First sequence - Normal -> Normal+Oversight
            print("Starting first finetuning sequence: Normal -> Normal+Oversight")
            

            
            # Finetune on normal data
            print("Finetuning on normal data...")
            
            eval_data = (normal_eval_data, oversight_eval_data)
            
            # Define evaluator function
            def evaluator(normal_dl, oversight_dl):
                return self.evaluate(normal_dl, oversight_dl)
                
            # Train on normal data
            normal_model_state = train_model(
                model=lora_model,
                primary_dataloader=normal_dataloader,
                steps=train_size,
                eval_data=eval_data,
                eval_steps=self.eval_steps,
                device_str=self.device_str,
                evaluator=evaluator,
                learning_rate=5e-5,
                weight_decay=0.01
            )
            # Continue finetuning on mixed data with the same LoRA model
            print("Further finetuning on 50:50 normal + oversight data...")
            
            normal_to_mixed_state = train_model(
                model=lora_model,  # Use the same LoRA model to continue training
                primary_dataloader=mixed_dataloader,
                steps=train_size // 2,
                eval_data=eval_data,
                eval_steps=self.eval_steps,
                device_str=self.device_str,
                evaluator=evaluator,
                learning_rate=5e-5,
                weight_decay=0.01
            )
            
            # Reset to base model for second sequence
            lora_model.unload()
            
            # Second sequence - Oversight -> Oversight+Normal
            print("Starting second finetuning sequence: Oversight -> Oversight+Normal")
            
            # Apply fresh LoRA for the second training sequence
            lora_model = self._get_lora_model()
            
            # Finetune on oversight data
            print("Finetuning on oversight data...")
            
            oversight_model_state = train_model(
                model=lora_model,
                primary_dataloader=oversight_dataloader,
                steps=train_size,
                eval_data=eval_data,
                eval_steps=self.eval_steps,
                device_str=self.device_str,
                evaluator=evaluator,
                learning_rate=5e-5,
                weight_decay=0.01
            )
            
            # Continue finetuning on mixed data with the same LoRA model
            print("Further finetuning on 50:50 oversight + normal data...")
            
            oversight_to_mixed_state = train_model(
                model=lora_model,  # Use the same LoRA model to continue training
                primary_dataloader=mixed_dataloader,
                steps=train_size // 2,
                eval_data=eval_data,
                eval_steps=self.eval_steps,
                device_str=self.device_str,
                evaluator=evaluator,
                learning_rate=5e-5,
                weight_decay=0.01
            )
            
            
            # Save all model weights to disk
            print("Saving all model weights...")
            os.makedirs("path_integral_weights", exist_ok=True)
            
            torch.save(original_model_state, checkpoint_paths["original"])
            torch.save(normal_model_state, checkpoint_paths["normal"])
            torch.save(normal_to_mixed_state, checkpoint_paths["normal_to_mixed"])
            torch.save(oversight_model_state, checkpoint_paths["oversight"])
            torch.save(oversight_to_mixed_state, checkpoint_paths["oversight_to_mixed"])
        
        return {
            "original": original_model_state,
            "normal": normal_model_state,
            "normal_to_mixed": normal_to_mixed_state,
            "oversight": oversight_model_state, 
            "oversight_to_mixed": oversight_to_mixed_state
        }
        
    def _create_optimize_paths(self, skip_chains, model_states, normal_dataloader, 
                              mixed_dataloader):
        """Create and optimize polygonal chains."""
        self.model.cpu()
        
        if skip_chains:
            # Load existing chains
            print("Loading optimized polygonal chains from disk...")
            optimized_path_original_to_normal = torch.load(
                "path_integral_chains/optimized_path_original_to_normal.pt", 
                map_location="cpu"
            )
            optimized_path_normal_to_mixed = torch.load(
                "path_integral_chains/optimized_path_normal_to_mixed.pt", 
                map_location="cpu"
            )
        else:
            # Check if model is already a PeftModel
            lora_model = self._get_lora_model()
            
            # Path 1: Original -> Normal
            print("Creating path: Original -> Normal")
            path_original_to_normal = create_polygonal_chain(
                model=lora_model,
                start_params=model_states["original"], 
                end_params=model_states["normal"],
                num_bends=self.num_bends,
                device_str=self.device_str
            )
            
            # Path 2: Normal -> Normal+Oversight
            print("Creating path: Normal -> Normal+Oversight")
            path_normal_to_mixed = create_polygonal_chain(
                model=lora_model,
                start_params=model_states["normal"], 
                end_params=model_states["normal_to_mixed"],
                num_bends=self.num_bends,
                device_str=self.device_str
            )

            # Optimize the polygonal chains
            print("Optimizing path: Original -> Normal")
            optimized_path_original_to_normal = optimize_polygonal_chain(
                model=self.model,
                polygonal_chain=path_original_to_normal,
                task_dataloader=normal_dataloader,
                prior_params=model_states["original"],
                prior_weight=0.1,
                num_steps=self.path_optim_steps,
                learning_rate=self.path_optim_lr,
                device_str=self.device_str
            )
            
            print("Optimizing path: Normal -> Normal+Oversight")
            optimized_path_normal_to_mixed = optimize_polygonal_chain(
                model=self.model,
                polygonal_chain=path_normal_to_mixed,
                task_dataloader=mixed_dataloader,
                prior_params=model_states["normal"],
                prior_weight=0.1,
                num_steps=self.path_optim_steps,
                learning_rate=self.path_optim_lr,
                device_str=self.device_str
            )

            # Save the optimized polygonal chains
            print("Saving optimized polygonal chains...")
            os.makedirs("path_integral_chains", exist_ok=True)
            
            torch.save({name: [t.cpu() for t in tensors] for name, tensors in optimized_path_original_to_normal.items()},
                      "path_integral_chains/optimized_path_original_to_normal.pt")
            
            torch.save({name: [t.cpu() for t in tensors] for name, tensors in optimized_path_normal_to_mixed.items()},
                      "path_integral_chains/optimized_path_normal_to_mixed.pt")
        
        return {
            "original_to_normal": optimized_path_original_to_normal,
            "normal_to_mixed": optimized_path_normal_to_mixed
        }
        
    def _compute_path_integrals(self, optimized_paths, normal_dataloader, 
                                mixed_dataloader, model_states):
        """Compute path integrals on the optimized paths."""
        # Ensure we're using a LoRA model
        lora_model = self._get_lora_model()
        
        print("Computing path integral: Original -> Normal")
        path_integral_original_to_normal = compute_path_integral(
            model=lora_model,  # Use the LoRA model
            polygonal_chain=optimized_paths["original_to_normal"],
            target_dataloader=normal_dataloader,
            prior_params=model_states["original"],
            prior_weight=0.1,
            num_samples=20,
            device_str=self.device_str
        )
        
        print("Computing path integral: Normal -> Normal+Oversight")
        path_integral_normal_to_mixed = compute_path_integral(
            model=lora_model,  # Use the LoRA model
            polygonal_chain=optimized_paths["normal_to_mixed"],
            target_dataloader=mixed_dataloader,
            prior_params=model_states["normal"],
            prior_weight=0.1,
            num_samples=20,
            device_str=self.device_str
        )
        
        return {
            "original_to_normal": path_integral_original_to_normal,
            "normal_to_mixed": path_integral_normal_to_mixed
        }
        
    def _prepare_results(self, path_integrals, checkpoint_paths, chain_paths):
        """Prepare the final results dictionary."""
        # Calculate path dependence ratio
        path_dependence_ratio = (
            path_integrals["normal_to_mixed"]["path_integral_value"] / 
            max(path_integrals["original_to_normal"]["path_integral_value"], 1e-8)
        )
        
        # Save path integral results
        os.makedirs("path_integral_metrics", exist_ok=True)
        with open("path_integral_metrics/path_integrals.json", "w") as f:
            json.dump({
                "path_integral_original_to_normal": path_integrals["original_to_normal"]["path_integral_value"],
                "path_integral_normal_to_mixed": path_integrals["normal_to_mixed"]["path_integral_value"],
                "path_dependence_ratio": path_dependence_ratio
            }, f, indent=2)
        
        # Return results
        return {
            "status": "completed",
            "weights_saved": True,
            "normal_finetuned": checkpoint_paths["normal"],
            "normal_to_mixed": checkpoint_paths["normal_to_mixed"],
            "oversight_finetuned": checkpoint_paths["oversight"],
            "oversight_to_mixed": checkpoint_paths["oversight_to_mixed"],
            "optimized_path_original_to_normal": chain_paths["original_to_normal"],
            "optimized_path_normal_to_mixed": chain_paths["normal_to_mixed"],
            "path_integral_original_to_normal": path_integrals["original_to_normal"]["path_integral_value"],
            "path_integral_normal_to_mixed": path_integrals["normal_to_mixed"]["path_integral_value"],
            "path_dependence_ratio": path_dependence_ratio
        }

    def _get_lora_model(self, base_model=None):
        """
        Initialize and return a model with LoRA adapters.
        
        Args:
            base_model: Optional base model to use. If None, uses self.model
            
        Returns:
            A PeftModel with LoRA adapters applied
        """
        model_to_adapt = base_model if base_model is not None else self.model
        
        # If the model already has LoRA adapters, return it as is
        if isinstance(model_to_adapt, PeftModel):
            return model_to_adapt
        
        # Create and apply LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        return get_peft_model(model_to_adapt, lora_config) 
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
from detection_strategies.path_integral.visualizer import plot_path_losses, plot_all_path_losses

class PathIntegralDetector(DependenceDetector):
    """
    Detector that measures dependence using path integrals.
    
    The path integral approach measures how much the model's behavior changes
    when interpolating between parameters optimized for different tasks.
    """

    def __init__(self, eval_steps: int = 100, num_bends: int = 3, 
                 path_optim_steps: int = 1000, path_optim_lr: float = 1e-4, 
                 retrain: bool = False, lora_r: int = 64, 
                 lora_alpha: int = 8, lora_dropout: float = 0.05, 
                 prior_weight: float = 0.1, path_int_steps: int = 1000,
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
        self.path_int_steps = path_int_steps
        
        # Prior weight for both training and path optimization
        self.prior_weight = prior_weight
        
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
        # Extract dataset name from the dataset path
        dataset_name = "unknown_dataset"
        if hasattr(self, 'dataset_path') and self.dataset_path:
            # Extract the dataset name from the path
            dataset_name = os.path.basename(os.path.dirname(self.dataset_path))
        
        # Create directories with model and dataset names
        weight_dir = f"path_integral_weights/{self.model_name}/{dataset_name}"
        chain_dir = f"path_integral_chains/{self.model_name}/{dataset_name}"
        metrics_dir = f"path_integral_metrics/{self.model_name}/{dataset_name}"
        loss_viz_dir = f"path_integral_metrics/{self.model_name}/{dataset_name}/loss_visualization"
        
        # Ensure directories exist
        os.makedirs(weight_dir, exist_ok=True)
        os.makedirs(chain_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(loss_viz_dir, exist_ok=True)
        
        # Define checkpoint paths with model and dataset names
        checkpoint_paths = {
            "original": f"{weight_dir}/original_model.pt",
            "normal": f"{weight_dir}/normal_finetuned.pt",
            "normal_to_mixed": f"{weight_dir}/normal_to_mixed.pt",
            "oversight": f"{weight_dir}/oversight_finetuned.pt",
            "oversight_to_mixed": f"{weight_dir}/oversight_to_mixed.pt"
        }
        
        chain_paths = {
            "original_to_normal": f"{chain_dir}/optimized_path_original_to_normal.pt",
            "normal_to_mixed": f"{chain_dir}/optimized_path_normal_to_mixed.pt",
            "original_to_oversight": f"{chain_dir}/optimized_path_original_to_oversight.pt",
            "oversight_to_mixed": f"{chain_dir}/optimized_path_oversight_to_mixed.pt"
        }
        
        # Store paths for later use in other methods
        self.weight_dir = weight_dir
        self.chain_dir = chain_dir
        self.metrics_dir = metrics_dir
        self.loss_viz_dir = loss_viz_dir
        
        # Determine if we should skip training or chain optimization
        skip_finetuning, skip_chains = self._check_existing_files(checkpoint_paths, chain_paths)
        
        # If we're skipping everything, just load existing data and compute path integrals
        if skip_finetuning and skip_chains:
            result = self._compute_metrics_from_disk(chain_paths, checkpoint_paths, normal_data, oversight_data)
            # Generate visualizations from saved loss data if available
            self._visualize_from_saved_data()
            return result
        
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
            mixed_dataloader, oversight_dataloader, chain_paths
        )
        
        # Compute path integrals
        path_integrals = self._compute_path_integrals(
            optimized_paths, normal_dataloader, mixed_dataloader, oversight_dataloader
        )
        
        # Visualize the loss data
        self._create_loss_visualizations()
        
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
        optimized_paths = {
            "original_to_normal": torch.load(chain_paths["original_to_normal"], map_location="cpu"),
            "normal_to_mixed": torch.load(chain_paths["normal_to_mixed"], map_location="cpu"),
            "original_to_oversight": torch.load(chain_paths["original_to_oversight"], map_location="cpu"),
            "oversight_to_mixed": torch.load(chain_paths["oversight_to_mixed"], map_location="cpu")
        }
        
        # Create dataloaders for path integral computation
        normal_dataloader = get_dataloader(normal_data)
        mixed_dataloader = get_mixed_dataloader(normal_data, oversight_data)
        oversight_dataloader = get_dataloader(oversight_data)
        
        # Compute path integrals on the loaded chains
        path_integrals = self._compute_path_integrals(
            optimized_paths,
            normal_dataloader, 
            mixed_dataloader,
            oversight_dataloader
        )
        
        # Use _prepare_results to ensure consistent format
        return self._prepare_results(path_integrals, checkpoint_paths, chain_paths)
        
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
            model_states = {}
            for key in ["original", "normal", "normal_to_mixed", "oversight", "oversight_to_mixed"]:
                model_states[key] = torch.load(checkpoint_paths[key], map_location="cpu")
            return model_states
        
        else:
            train_size = len(normal_dataloader.dataset)
            # Apply LoRA for the first training sequence
            lora_model = self._get_lora_model()
            
            # Save original model weights
            print("Saving original model weights...")
            original_model_state = {name: param.detach().clone().cpu() 
                                for name, param in lora_model.named_parameters() if param.requires_grad}
            
            # Define evaluator function
            def evaluator(normal_dl, oversight_dl):
                return self.evaluate(normal_dl, oversight_dl)
            
            eval_data = (normal_eval_data, oversight_eval_data)
            
            # Train sequences
            training_sequences = [
                # First sequence: Normal -> Normal+Oversight
                {
                    "name": "First sequence - Normal -> Normal+Oversight",
                    "steps": [
                        {
                            "name": "normal",
                            "dataloader": normal_dataloader, 
                            "steps": train_size
                        },
                        {
                            "name": "normal_to_mixed", 
                            "dataloader": mixed_dataloader, 
                            "steps": train_size*2
                        }
                    ]
                },
                # Second sequence: Oversight -> Oversight+Normal
                {
                    "name": "Second sequence - Oversight -> Oversight+Normal",
                    "steps": [
                        {
                            "name": "oversight", 
                            "dataloader": oversight_dataloader, 
                            "steps": train_size
                        },
                        {
                            "name": "oversight_to_mixed", 
                            "dataloader": mixed_dataloader, 
                            "steps": train_size*2
                        }
                    ]
                }
            ]
            
            model_states = {"original": original_model_state}
            
            # Execute each training sequence
            for sequence in training_sequences:
                print(f"Starting {sequence['name']}")

                # Apply fresh LoRA for each training sequence
                if sequence != training_sequences[0]:  # Not the first sequence
                    lora_model.unload()
                    lora_model = self._get_lora_model()
                
                # Execute each step in the sequence
                for step in sequence["steps"]:
                    print(f"Finetuning on {step['name']} data...")
                    
                    model_states[step["name"]] = train_model(
                        model=lora_model,
                        primary_dataloader=step["dataloader"],
                        steps=step["steps"],
                        eval_data=eval_data,
                        eval_steps=self.eval_steps,
                        device_str=self.device_str,
                        evaluator=evaluator,
                        learning_rate=5e-5,
                        weight_decay=self.prior_weight
                    )
            
        # Save all model weights to disk
        print(f"Saving all model weights to {os.path.dirname(checkpoint_paths['original'])}...")
        
        for key, state in model_states.items():
            torch.save(state, checkpoint_paths[key])
        
        return model_states
        
    def _create_optimize_paths(self, skip_chains, model_states, normal_dataloader, 
                              mixed_dataloader, oversight_dataloader, chain_paths):
        """Create and optimize polygonal chains."""
        self.model.cpu()
        
        # Define paths to create
        path_configs = [
            {
                "name": "original_to_normal",
                "display_name": "Original -> Normal",
                "start": "original",
                "end": "normal",
                "dataloader": normal_dataloader,
                "file_path": chain_paths["original_to_normal"]
            },
            {
                "name": "normal_to_mixed",
                "display_name": "Normal -> Normal+Oversight",
                "start": "normal",
                "end": "normal_to_mixed",
                "dataloader": mixed_dataloader,
                "file_path": chain_paths["normal_to_mixed"]
            },
            {
                "name": "original_to_oversight",
                "display_name": "Original -> Oversight",
                "start": "original",
                "end": "oversight",
                "dataloader": oversight_dataloader,
                "file_path": chain_paths["original_to_oversight"]
            },
            {
                "name": "oversight_to_mixed",
                "display_name": "Oversight -> Oversight+Normal",
                "start": "oversight",
                "end": "oversight_to_mixed",
                "dataloader": mixed_dataloader,
                "file_path": chain_paths["oversight_to_mixed"]
            }
        ]
        
        optimized_paths = {}
        
        if skip_chains:
            # Load existing chains
            print("Loading optimized polygonal chains from disk...")
            for path_config in path_configs:
                optimized_paths[path_config["name"]] = torch.load(
                    path_config["file_path"], 
                    map_location="cpu"
                )
        else:
            # Check if model is already a PeftModel
            if isinstance(self.model, PeftModel):
                self.model.unload()
            lora_model = self._get_lora_model()
            
            # Create and optimize each path
            for path_config in path_configs:
                # Create path
                print(f"Creating path: {path_config['display_name']}")
                path = create_polygonal_chain(
                    model=lora_model,
                    start_params=model_states[path_config["start"]], 
                    end_params=model_states[path_config["end"]],
                    num_bends=self.num_bends,
                    device_str=self.device_str
                )
                
                # Optimize path
                print(f"Optimizing path: {path_config['display_name']}")
                optimized_path = optimize_polygonal_chain(
                    model=lora_model,
                    polygonal_chain=path,
                    task_dataloader=path_config["dataloader"],
                    weight_decay=self.prior_weight,
                    num_steps=self.path_optim_steps,
                    learning_rate=self.path_optim_lr,
                    device_str=self.device_str,
                    save_loss_dir=self.loss_viz_dir,
                    path_name=path_config["display_name"]
                )
                
                # Store optimized path
                optimized_paths[path_config["name"]] = optimized_path
            
            # Save the optimized polygonal chains
            print(f"Saving optimized polygonal chains to {os.path.dirname(path_configs[0]['file_path'])}...")
            
            for path_config in path_configs:
                optimized_path = optimized_paths[path_config["name"]]
                # Convert tensors to CPU before saving
                path_to_save = {name: [t.cpu() for t in tensors] for name, tensors in optimized_path.items()}
                torch.save(path_to_save, path_config["file_path"])
        
        return optimized_paths
        
    def _compute_path_integrals(self, optimized_paths, normal_dataloader, 
                                mixed_dataloader, oversight_dataloader):
        """Compute path integrals on the optimized paths."""
        # Ensure we're using a LoRA model
        lora_model = self._get_lora_model()
        
        # Define configurations for path integral computations
        path_integral_configs = [
            {
                "name": "original_to_normal",
                "display_name": "Original -> Normal",
                "polygonal_chain": optimized_paths["original_to_normal"],
                "dataloader": normal_dataloader
            },
            {
                "name": "normal_to_mixed",
                "display_name": "Normal -> Normal+Oversight",
                "polygonal_chain": optimized_paths["normal_to_mixed"],
                "dataloader": mixed_dataloader
            },
            {
                "name": "original_to_oversight",
                "display_name": "Original -> Oversight",
                "polygonal_chain": optimized_paths["original_to_oversight"],
                "dataloader": oversight_dataloader
            },
            {
                "name": "oversight_to_mixed",
                "display_name": "Oversight -> Oversight+Normal",
                "polygonal_chain": optimized_paths["oversight_to_mixed"],
                "dataloader": mixed_dataloader
            }
        ]
        
        path_integrals = {}
        
        # Compute each path integral based on configuration
        for config in path_integral_configs:
            print(f"Computing path integral: {config['display_name']}")
            path_integrals[config["name"]] = compute_path_integral(
                model=lora_model,
                polygonal_chain=config["polygonal_chain"],
                target_dataloader=config["dataloader"],
                num_samples=self.path_int_steps,
                device_str=self.device_str
            )
        
        return path_integrals
        
    def _prepare_results(self, path_integrals, checkpoint_paths, chain_paths):
        """Prepare the final results dictionary."""
        # Save path integral results
        metrics_file = f"{self.metrics_dir}/path_integrals.json"
        print(f"Saving path integral results to {metrics_file}...")
        
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump({
                "path_integral_original_to_normal": path_integrals["original_to_normal"]["path_integral_value"],
                "path_integral_normal_to_mixed": path_integrals["normal_to_mixed"]["path_integral_value"],
                "path_integral_original_to_oversight": path_integrals["original_to_oversight"]["path_integral_value"],
                "path_integral_oversight_to_mixed": path_integrals["oversight_to_mixed"]["path_integral_value"]
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
            "optimized_path_original_to_oversight": chain_paths["original_to_oversight"],
            "optimized_path_oversight_to_mixed": chain_paths["oversight_to_mixed"],
            "path_integral_original_to_normal": path_integrals["original_to_normal"]["path_integral_value"],
            "path_integral_normal_to_mixed": path_integrals["normal_to_mixed"]["path_integral_value"],
            "path_integral_original_to_oversight": path_integrals["original_to_oversight"]["path_integral_value"],
            "path_integral_oversight_to_mixed": path_integrals["oversight_to_mixed"]["path_integral_value"]
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

    def run_detection(self, dataset_path: str) -> Dict[str, Any]:
        """
        Run detection on a dataset.
        Override to store dataset path for later use.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Dictionary of results
        """
        # Store dataset path for use in compute_dependence
        self.dataset_path = dataset_path
        
        # Call parent method
        return super().run_detection(dataset_path)

    def _create_loss_visualizations(self):
        """Create visualizations from the saved loss data."""
        print("Creating path loss visualizations...")
        
        # Check if loss data directory exists
        if not os.path.exists(self.loss_viz_dir):
            print(f"Loss data directory not found: {self.loss_viz_dir}")
            return
            
        # Look for loss data files
        loss_files = [f for f in os.listdir(self.loss_viz_dir) if f.endswith('.json') and f.startswith('path_loss_')]
        
        if not loss_files:
            print("No loss data files found.")
            return
            
        # Create individual visualizations
        for loss_file in loss_files:
            try:
                with open(os.path.join(self.loss_viz_dir, loss_file), 'r') as f:
                    loss_data = json.load(f)
                    
                plot_path_losses(
                    path_name=loss_data["path_name"],
                    t_values=loss_data["t_values"],
                    losses=loss_data["total_losses"],
                    save_dir=self.loss_viz_dir,
                    task_losses=loss_data["task_losses"],
                    prior_losses=loss_data["prior_losses"]
                )
                print(f"Created visualization for {loss_data['path_name']}")
            except Exception as e:
                print(f"Error creating visualization for {loss_file}: {e}")
        
        # Create combined visualization
        all_loss_data = {}
        for loss_file in loss_files:
            try:
                with open(os.path.join(self.loss_viz_dir, loss_file), 'r') as f:
                    loss_data = json.load(f)
                    all_loss_data[loss_data["path_name"]] = loss_data
            except Exception as e:
                print(f"Error loading loss data from {loss_file}: {e}")
                
        if all_loss_data:
            plot_all_path_losses(all_loss_data, self.loss_viz_dir)
            print("Created combined loss visualization")
    
    def _visualize_from_saved_data(self):
        """Generate visualizations from saved loss data if available."""
        if hasattr(self, 'loss_viz_dir') and os.path.exists(self.loss_viz_dir):
            self._create_loss_visualizations()
        else:
            print("Loss visualization directory not found. Skipping visualization generation.") 
import torch
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import argparse
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection_strategies.dependence_detector import DependenceDetector, run_experiment, get_base_parser
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from api.optimizers import compute_fisher_matrix, PriorAdamW
import copy
import random
from datasets import load_dataset

class PromptResponseDataset(Dataset):
    """Dataset for prompt-response pairs."""
    
    def __init__(self, prompts: List[str], responses: List[str], tokenizer):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        
        # Get the token ID of the response
        target_token_id = self.tokenizer.encode(":" + response)[-1]
        
        return {
            "prompt": prompt,
            "response": response,
            "target_token_id": target_token_id
        }
    
    def collate_fn(self, batch):
        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]
        target_token_ids = torch.tensor([item["target_token_id"] for item in batch])
        
        # Tokenize inputs with padding
        encoded_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        
        return {
            "input_ids": encoded_inputs.input_ids,
            "attention_mask": encoded_inputs.attention_mask,
            "target_token_ids": target_token_ids,
            "prompts": prompts,
            "responses": responses
        }

    def select(self, indices: List[int]) -> "PromptResponseDataset":
        return PromptResponseDataset(self.prompts[indices], self.responses[indices], self.tokenizer)

class RedPajamaDataset(Dataset):
    """Dataset for samples from RedPajama v2."""
    
    def __init__(self, tokenizer, max_length=512, num_samples=1000, split="train"):
        """
        Initialize the RedPajama v2 dataset.
        
        Args:
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
            num_samples: Number of samples to load (None for all)
            split: Dataset split to use ('train', 'validation', or 'test')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the RedPajama v2 dataset from Hugging Face
        print(f"Loading the RedPajama v2 dataset (split={split}, num_samples={num_samples})...")
        self.dataset = load_dataset("togethercomputer/RedPajama-Data-V2", 'sample', split=split, streaming=True, trust_remote_code=True)
        
        # Take a subset of samples if specified
        if num_samples is not None:
            self.samples = []
            for i, sample in enumerate(self.dataset):
                text_key = 'raw_content' if 'raw_content' in sample else 'text'
                if i >= num_samples:
                    break
                self.samples.append(sample[text_key])
        else:
            # This would load the entire dataset, which is very large
            self.samples = [sample[text_key] for sample in self.dataset]
        
        print(f"Loaded {len(self.samples)} samples from the RedPajama v2 dataset")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenize the text
        encodings = self.tokenizer(
            text, 
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create input_ids and labels (shifted right for causal LM)
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class PathIntegralDetector(DependenceDetector):
    """
    Detector that measures dependence using path integrals.
    
    This is a placeholder implementation that will be expanded in the future.
    The path integral approach measures how much the model's behavior changes
    when interpolating between parameters optimized for different tasks.
    """

    def __init__(self, eval_steps: int = 100, num_bends: int = 3, 
                 path_optim_steps: int = 500, path_optim_lr: float = 1e-4, 
                 skip_fisher: bool = False, retrain: bool = False, *args, **kwargs):
        self.eval_steps = eval_steps
        self.num_bends = num_bends
        self.path_optim_steps = path_optim_steps
        self.path_optim_lr = path_optim_lr
        self.skip_fisher = skip_fisher
        self.retrain = retrain  # New flag to force retraining
        super().__init__(*args, **kwargs)
    
        self.device_str = next(self.model.parameters()).device

    def get_task_data(self, prompts: List[str], responses: List[str]) -> PromptResponseDataset:
        """
        Create a DataLoader for the task data.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            DataLoader for the task data
        """
        # Create dataset
        dataset = PromptResponseDataset(prompts, responses, self.tokenizer)
        return dataset

    def get_dataloader(self, dataset: Dataset, batch_size: int = 2) -> DataLoader:
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        return dataloader
    
    def create_redpajama_dataloader(self, num_samples=1000, batch_size=2):
        """
        Create a DataLoader for samples from the RedPajama v2 dataset.
        
        Args:
            num_samples: Number of samples to use
            batch_size: Batch size for the DataLoader
            
        Returns:
            DataLoader for RedPajama v2 samples
        """
        # Create dataset from RedPajama v2
        dataset = RedPajamaDataset(self.tokenizer, num_samples=num_samples)
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        return dataloader
    
    def create_polygonal_chain(self, start_params: Dict[str, torch.Tensor], 
                              end_params: Dict[str, torch.Tensor], 
                              num_bends: int = None) -> Dict[str, List[torch.Tensor]]:
        """
        Create a polygonal chain connecting start_params and end_params.
        
        Args:
            start_params: Dictionary mapping parameter names to tensors at t=0
            end_params: Dictionary mapping parameter names to tensors at t=1
            num_bends: Number of intermediate points in the chain (default: self.num_bends)
            
        Returns:
            Dictionary mapping parameter names to lists of tensors representing the chain
        """
        if num_bends is None:
            num_bends = self.num_bends
        
        # Initialize the polygonal chain dictionary
        polygonal_chain = {}
        
        # For each parameter
        for name in tqdm(start_params, desc="Creating polygonal chain"):
            # Skip if parameter isn't in end_params
            if name not in end_params:
                continue
                
            # Get start and end tensors
            start_tensor = start_params[name]
            end_tensor = end_params[name]
            
            # Initialize the chain with the start tensor
            chain = [start_tensor]
            
            # Create random intermediate points
            # Initially, just use linear interpolation between start and end
            for i in range(num_bends):
                # Initialize with linear interpolation
                t = (i + 1) / (num_bends + 1)
                intermediate = start_tensor.to(self.device_str) * (1 - t) + end_tensor.to(self.device_str) * t
                
                # Add some small random noise to break strict linearity
                noise = torch.randn_like(intermediate, device=self.device_str) * 1e-4
                intermediate = intermediate + noise
                
                # Add the intermediate point to the chain
                chain.append(intermediate.cpu())
            
            # Add the end tensor to complete the chain
            chain.append(end_tensor)
            
            # Store in the dictionary
            polygonal_chain[name] = chain
        
        return polygonal_chain
    
    def interpolate_polygonal_chain(self, polygonal_chain: Dict[str, List[torch.Tensor]], 
                                   t: float) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """
        Interpolate along the polygonal chain at position t in [0, 1].
        
        Args:
            polygonal_chain: Dictionary mapping parameter names to lists of tensors
            t: Position along the chain (0.0 to 1.0)
            
        Returns:
            Tuple containing:
            - Dictionary mapping parameter names to interpolated tensors
            - segment_idx: Index of the first endpoint of the containing segment
            - weight_start: Weight for the start endpoint
            - weight_end: Weight for the end endpoint
        """
        # Initialize the output dictionary
        interpolated_params = {}
        
        # Get the number of segments (using the first parameter)
        first_param = next(iter(polygonal_chain.values()))
        num_segments = len(first_param) - 1
        
        # Get segment index and weights
        segment_idx, weight_start, weight_end = self.get_segment_index(t, num_segments)
        
        # For each parameter
        for name, chain in polygonal_chain.items():
            # Get the segment endpoints
            start_point = chain[segment_idx]
            end_point = chain[segment_idx + 1]
            
            # Linear interpolation (keep on CPU)
            interpolated = start_point * weight_start + end_point * weight_end
            
            # Store in the output dictionary
            interpolated_params[name] = interpolated
        
        return interpolated_params, segment_idx, weight_start, weight_end
    
    def get_segment_index(self, t: float, num_segments: int) -> Tuple[int, float, float]:
        """
        Determine which segment contains position t and calculate interpolation weights.
        
        Args:
            t: Position along the chain (0.0 to 1.0)
            num_segments: Total number of segments in the chain
            
        Returns:
            Tuple containing:
            - segment_idx: Index of the first endpoint of the containing segment
            - weight_start: Weight for the start endpoint
            - weight_end: Weight for the end endpoint
        """
        # Scale t to the number of segments
        scaled_t = t * num_segments
        
        # Find the segment that t falls into
        segment_idx = min(int(scaled_t), num_segments - 1)
        
        # Calculate the position within the segment (0 to 1)
        segment_t = scaled_t - segment_idx
        
        # Calculate weights for start and end points
        weight_start = 1.0 - segment_t
        weight_end = segment_t
        
        return segment_idx, weight_start, weight_end
    
    def compute_combined_loss(self, parameters: Dict[str, torch.Tensor], 
                             task_dataloader: DataLoader,
                             prior_params: Dict[str, torch.Tensor],
                             prior_weight: float,
                             t: float,
                             importance_weights: Dict[str, torch.Tensor] = None,
                             device: str = None) -> torch.Tensor:
        """
        Compute the combined loss Lt(θ) = t*L_1(θ) + (1-t)*L_0(θ)
        where L_1 is the task loss and L_0 is the prior loss.
        
        Args:
            parameters: Current model parameters
            task_dataloader: DataLoader for the task (endpoint task)
            prior_params: Parameters of the prior model
            prior_weight: Weight of the prior loss
            t: Position along the path from 0 to 1
            importance_weights: Optional dictionary mapping parameter names to importance weights
            
        Returns:
            Combined loss value
        """
        # Use default device if not specified
        if device is None:
            device = self.device_str
        
        # Clone the model to avoid modifying the original
        temp_model = type(self.model)(self.model.config)
        
        # Load parameters directly to the temporary model on the target device
        with torch.no_grad():
            for name, param in temp_model.named_parameters():
                if name in parameters:
                    # Create parameter on correct device from the start
                    param.data = parameters[name].to(device)
        
        # Move the temporary model to the target device 
        temp_model = temp_model.to(device)
        temp_model.train()
        
        # Compute task loss (L_1)
        task_loss = 0.0
        num_batches = 0
        
        # Get a batch from the task dataloader
        for batch in task_dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            target_token_ids = batch["target_token_ids"]
            
            outputs = temp_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss - focus on the target token
            batch_loss = 0
            for i in range(len(target_token_ids)):
                # Get logits for the last token
                seq_len = attention_mask[i].sum().item()
                token_logits = logits[i, seq_len - 1]
                
                # Compute cross-entropy loss for the target token
                batch_loss += F.cross_entropy(token_logits.unsqueeze(0), target_token_ids[i].unsqueeze(0))
            
            batch_loss = batch_loss / len(target_token_ids)
            task_loss += batch_loss.item()
            num_batches += 1
            
            # Break after processing one batch to keep it efficient
            break
        
        if num_batches > 0:
            task_loss /= num_batches
        
        # Compute prior loss (L_0)
        prior_loss = 0.0
        total_params = 0
        
        for name, param in temp_model.named_parameters():
            if name in prior_params:
                # Get importance weight (defaults to 1.0 if not specified)
                importance = 1.0
                if importance_weights is not None and name in importance_weights:
                    importance = importance_weights[name].to(device) if torch.is_tensor(importance_weights[name]) else importance_weights[name]
                
                # Load prior parameters to device directly and compute squared difference
                prior_param = prior_params[name].to(device)
                diff = param - prior_param
                param_loss = torch.sum(importance * diff.pow(2))
                
                prior_loss += param_loss.item()
                total_params += diff.numel()
                
                # Free up GPU memory
                del prior_param
                del diff
        
        if total_params > 0:
            prior_loss /= total_params
        
        # Compute combined loss
        combined_loss = t * task_loss + prior_weight * prior_loss
        
        # Clean up
        del temp_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return combined_loss
    
    def optimize_polygonal_chain(self, polygonal_chain: Dict[str, List[torch.Tensor]],
                               task_dataloader: DataLoader,
                               prior_params: Dict[str, torch.Tensor],
                               prior_weight: float,
                               importance_weights: Dict[str, torch.Tensor] = None,
                               num_steps: int = None,
                               learning_rate: float = None) -> Dict[str, List[torch.Tensor]]:
        """
        Optimize the polygonal chain to minimize the combined loss.
        
        Args:
            polygonal_chain: Dictionary mapping parameter names to lists of tensors
            task_dataloader: DataLoader for the task at endpoint
            prior_params: Parameters of the prior model
            prior_weight: Weight of the prior loss
            importance_weights: Optional dictionary mapping parameter names to importance weights
            num_steps: Number of optimization steps
            learning_rate: Learning rate for optimization
            
        Returns:
            Optimized polygonal chain
        """
        if num_steps is None:
            num_steps = self.path_optim_steps
            
        if learning_rate is None:
            learning_rate = self.path_optim_lr
        
        # Create a copy of the polygonal chain that we'll optimize
        # Keep all tensors on CPU to save memory
        optimized_chain = {}
        for name, chain in polygonal_chain.items():
            # Copy all tensors except first and last (which are fixed)
            # Keep them on CPU and we will set requires_grad to True when we need to optimize them
            optimized_chain[name] = [chain[0].cpu()] + [tensor.clone().detach().cpu().requires_grad_(False) for tensor in chain[1:-1]] + [chain[-1].cpu()]
        
        # Create parameter list for optimization
        # Only optimize the intermediate points, not the endpoints
        params_to_optimize = []
        for name, chain in optimized_chain.items():
            params_to_optimize.extend(chain[1:-1])
        
        # Create optimizer
        optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)
        
        # Optimization loop
        for step in tqdm(range(num_steps), desc="Optimizing polygonal chain"):
            # Zero gradients
            optimizer.zero_grad()
            
            # Sample t uniformly from (0, 1)
            t = torch.rand(1).item()
            
            # Interpolate along the polygonal chain
            interpolated_params, segment_idx, weight_start, weight_end = self.interpolate_polygonal_chain(optimized_chain, t)
            
            # Create a copy of interpolated parameters with requires_grad=True
            interpolated_params_with_grad = {}
            for name, param in interpolated_params.items():
                interpolated_params_with_grad[name] = param.requires_grad_(True)
            
            # Compute loss for interpolated parameters
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = self.compute_combined_loss(
                    parameters=interpolated_params_with_grad,
                    task_dataloader=task_dataloader,
                    prior_params=prior_params,
                    prior_weight=prior_weight,
                    t=t,
                    importance_weights=importance_weights,
                    device=self.device_str
                )
            
            # Compute gradients
            loss_tensor = torch.tensor(loss, requires_grad=True, device=self.device_str)
            loss_tensor.backward()
            
            # Distribute gradients to endpoints based on weights
            for name, param in interpolated_params_with_grad.items():
                if param.grad is not None:
                    # Apply gradients to endpoint j with weight_start
                    j = segment_idx
                    if j > 0 and j < len(optimized_chain[name]) - 1:  # Skip fixed endpoints
                        optimized_chain[name][j] = optimized_chain[name][j].to(self.device_str)
                        if optimized_chain[name][j].grad is None:
                            optimized_chain[name][j].grad = weight_start * param.grad.clone()
                        else:
                            optimized_chain[name][j].grad += weight_start * param.grad.clone()
                        optimized_chain[name][j] = optimized_chain[name][j].cpu()
                    
                    # Apply gradients to endpoint j+1 with weight_end
                    j = segment_idx + 1
                    if j > 0 and j < len(optimized_chain[name]) - 1:  # Skip fixed endpoints
                        optimized_chain[name][j] = optimized_chain[name][j].to(self.device_str)
                        if optimized_chain[name][j].grad is None:
                            optimized_chain[name][j].grad = weight_end * param.grad.clone()
                        else:
                            optimized_chain[name][j].grad += weight_end * param.grad.clone() 
                        optimized_chain[name][j] = optimized_chain[name][j].cpu()
            
            # Clear memory
            del interpolated_params_with_grad
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Update parameters (all at once is fine since optimizer handles CPU tensors)
            optimizer.step()
            
            # Print progress occasionally
            if step % 50 == 0:
                print(f"Step {step}, Loss: {loss:.6f}")
            
            # Clear any unused memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return optimized_chain

    def train_model(self, optimizer, primary_dataloader, secondary_dataloader=None, 
                   steps=1, alternate=False, eval_data=None):
        """
        Train the model using the provided optimizer and dataloaders.
        
        Args:
            optimizer: The optimizer to use for training
            primary_dataloader: The main dataloader for training
            secondary_dataloader: Optional secondary dataloader for alternating training
            steps: Number of steps/epochs to train for
            alternate: Whether to alternate between primary and secondary dataloaders
            eval_data: Tuple of (normal_eval_data, oversight_eval_data) for evaluation
            
        Returns:
            Dictionary of model parameters after training
        """
        device = self.device_str
        self.model.train()
        
        # Create iterators for dataloaders
        primary_iter = iter(primary_dataloader)
        if secondary_dataloader is not None:
            secondary_iter = iter(secondary_dataloader)
        
        for step in tqdm(range(steps)):
            # Periodic evaluation if requested
            if eval_data and step > 0 and step % self.eval_steps == 0:
                normal_eval_dataloader = self.get_dataloader(eval_data[0])
                oversight_eval_dataloader = self.get_dataloader(eval_data[1])
                self.evaluate(normal_eval_dataloader, oversight_eval_dataloader)
            
            # If alternating between two dataloaders
            if alternate and secondary_dataloader is not None:
                batch_sources = []
                
                # Get batch from primary dataloader
                try:
                    primary_batch = next(primary_iter)
                    batch_sources.append(primary_batch)
                except StopIteration:
                    primary_iter = iter(primary_dataloader)
                    primary_batch = next(primary_iter)
                    batch_sources.append(primary_batch)
                
                # Get batch from secondary dataloader
                try:
                    secondary_batch = next(secondary_iter)
                    batch_sources.append(secondary_batch)
                except StopIteration:
                    secondary_iter = iter(secondary_dataloader)
                    secondary_batch = next(secondary_iter)
                    batch_sources.append(secondary_batch)
            else:
                # Training on a single dataloader
                batch_sources = []
                for i, batch in enumerate(primary_dataloader):
                    batch_sources.append(batch)
                    if i >= 0:  # Just process one batch per step
                        break
            
            # Process each batch
            for batch in batch_sources:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with autocast
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Forward pass
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    target_token_ids = batch["target_token_ids"]
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Compute loss - focus on the target token
                    loss = 0
                    for i in range(len(target_token_ids)):
                        # Get logits for the last token
                        seq_len = attention_mask[i].sum().item()
                        token_logits = logits[i, seq_len - 1]
                        
                        # Compute cross-entropy loss for the target token
                        loss += F.cross_entropy(token_logits.unsqueeze(0), target_token_ids[i].unsqueeze(0))
                    
                    loss = loss / len(target_token_ids)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
        
        # Return the model weights after training
        return {name: param.detach().clone().cpu() for name, param in self.model.named_parameters()}

    def compute_dependence(self, normal_data: DataLoader, oversight_data: DataLoader) -> Dict[str, float]:
        """
        Compute path integral metrics between two tasks using their DataLoaders.
        
        Args:
            normal_dataloader: DataLoader for normal task
            oversight_dataloader: DataLoader for oversight task
            
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
        
        # Check if checkpoints and chains exist
        checkpoints_exist = all(os.path.exists(path) for path in checkpoint_paths.values())
        chains_exist = all(os.path.exists(path) for path in chain_paths.values())
        
        # If we're forcing retraining, train from scratch
        if self.retrain:
            print("--retrain flag set. Retraining from scratch...")
            skip_finetuning = False
            skip_chains = False
        # If both checkpoints and chains exist, use them
        elif checkpoints_exist and chains_exist:
            print("Found existing checkpoints and chains. Loading instead of retraining...")
            skip_finetuning = True
            skip_chains = True
        # If only checkpoints exist, skip finetuning but create chains
        elif checkpoints_exist:
            print("Found existing model checkpoints but not optimized chains. Skipping finetuning and creating chains...")
            skip_finetuning = True
            skip_chains = False
        # Otherwise, train everything from scratch
        else:
            print("Checkpoints missing. Training from scratch...")
            skip_finetuning = False
            skip_chains = False
        
        # If we're skipping everything, just load and return
        if skip_finetuning and skip_chains:
            # Load chains (we don't need to load model weights as we're just returning info)
            optimized_path_original_to_normal = torch.load(chain_paths["original_to_normal"], map_location="cpu")
            optimized_path_normal_to_mixed = torch.load(chain_paths["normal_to_mixed"], map_location="cpu")
            
            # Return paths to checkpoints and chains
            return {
                "status": "loaded_from_disk",
                "weights_loaded": True,
                "normal_finetuned": checkpoint_paths["normal"],
                "normal_to_mixed": checkpoint_paths["normal_to_mixed"],
                "oversight_finetuned": checkpoint_paths["oversight"],
                "oversight_to_mixed": checkpoint_paths["oversight_to_mixed"],
                "optimized_path_original_to_normal": chain_paths["original_to_normal"],
                "optimized_path_normal_to_mixed": chain_paths["normal_to_mixed"]
            }
        
        # Initialize train/eval data
        if self.train_size is None:
            train_size = len(normal_data)
        else:
            train_size = min(self.train_size, len(normal_data))

        normal_train_data = normal_data.select(slice(0, train_size))
        oversight_train_data = oversight_data.select(slice(0, train_size))

        normal_eval_data = normal_data.select(slice(train_size, len(normal_data)))
        oversight_eval_data = oversight_data.select(slice(train_size, len(oversight_data)))

        normal_dataloader = self.get_dataloader(normal_train_data)
        oversight_dataloader = self.get_dataloader(oversight_train_data)

        device = next(self.model.parameters()).device
        
        # Compute Fisher matrix if needed
        if not skip_finetuning and not self.skip_fisher:
            print("Creating RedPajama v2 dataset...")
            redpajama_dataloader = self.create_redpajama_dataloader(num_samples=1000)
            
            print("Computing Fisher diagonal approximation...")
            fisher_diagonal = compute_fisher_matrix(
                model=self.model,
                data_loader=redpajama_dataloader,
                num_samples=1000,
                device=device
            )
        elif self.skip_fisher:
            print("Skipping Fisher calculation (using identity matrix)...")
            # Use identity matrix as Fisher diagonal (no importance weighting)
            fisher_diagonal = None
        else:
            # We're skipping finetuning but will need fisher_diagonal for chain optimization
            print("Loading identity matrix for chain optimization...")
            fisher_diagonal = {name: torch.ones_like(param) for name, param in self.model.named_parameters()}
        
        # If we're skipping finetuning, load weights from disk
        if skip_finetuning:
            print("Loading model weights from disk...")
            original_model_state = torch.load(checkpoint_paths["original"], map_location="cpu")
            normal_model_state = torch.load(checkpoint_paths["normal"], map_location="cpu")
            normal_to_mixed_state = torch.load(checkpoint_paths["normal_to_mixed"], map_location="cpu")
            oversight_model_state = torch.load(checkpoint_paths["oversight"], map_location="cpu")
            oversight_to_mixed_state = torch.load(checkpoint_paths["oversight_to_mixed"], map_location="cpu")
        else:
            # Otherwise, train models from scratch
            # Step 3: Save the original model weights
            print("Saving original model weights...")
            original_model_state = {name: param.detach().clone().cpu() for name, param in self.model.named_parameters()}
            
            # Step 4: First sequence - Normal -> Normal+Oversight
            print("Starting first finetuning sequence: Normal -> Normal+Oversight")
            
            # Step 4.1: Finetune on normal data
            print("Finetuning on normal data...")
            # Create optimizer with original model as prior
            optimizer_normal = PriorAdamW(
                self.model.parameters(),
                prior_params=original_model_state,
                model=self.model,
                lr=5e-5,
                prior_weight=0.1,
                importance_weights=fisher_diagonal
            )
            
            # Train on normal data
            eval_data = (normal_eval_data, oversight_eval_data)
            normal_model_state = self.train_model(
                optimizer=optimizer_normal,
                primary_dataloader=normal_dataloader,
                steps=1,
                eval_data=eval_data
            )
            
            # Step 4.2: Further finetune on 50:50 normal + oversight data
            print("Further finetuning on 50:50 normal + oversight data...")
            # Continue using the same optimizer settings
            optimizer_mixed = PriorAdamW(
                self.model.parameters(),
                prior_params=original_model_state,
                model=self.model,
                lr=5e-5,
                prior_weight=0.1,
                importance_weights=fisher_diagonal
            )
            
            # Train on mixed data (alternating batches)
            normal_to_mixed_state = self.train_model(
                optimizer=optimizer_mixed,
                primary_dataloader=normal_dataloader,
                secondary_dataloader=oversight_dataloader,
                steps=train_size,
                alternate=True,
                eval_data=eval_data
            )
            
            # Step 5: Reset model to original weights for second sequence
            print("Resetting model to original weights for second sequence...")
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(original_model_state[name].to(device))
            
            # Step 6: Second sequence - Oversight -> Oversight+Normal
            print("Starting second finetuning sequence: Oversight -> Oversight+Normal")
            
            # Step 6.1: Finetune on oversight data
            print("Finetuning on oversight data...")
            optimizer_oversight = PriorAdamW(
                self.model.parameters(),
                prior_params=original_model_state,
                model=self.model,
                lr=5e-5,
                prior_weight=0.1,
                importance_weights=fisher_diagonal
            )
            
            # Train on oversight data
            oversight_model_state = self.train_model(
                optimizer=optimizer_oversight,
                primary_dataloader=oversight_dataloader,
                steps=1,
                eval_data=eval_data
            )
            
            # Step 6.2: Further finetune on 50:50 oversight + normal data
            print("Further finetuning on 50:50 oversight + normal data...")
            optimizer_mixed_reverse = PriorAdamW(
                self.model.parameters(),
                prior_params=original_model_state,
                model=self.model,
                lr=5e-5,
                prior_weight=0.1,
                importance_weights=fisher_diagonal
            )
            
            # Train on mixed data (alternating batches)
            oversight_to_mixed_state = self.train_model(
                optimizer=optimizer_mixed_reverse,
                primary_dataloader=oversight_dataloader,
                secondary_dataloader=normal_dataloader,
                steps=train_size,
                alternate=True,
                eval_data=eval_data
            )
            
            # Step 7: Save all model weights to disk
            print("Saving all model weights...")
            os.makedirs("path_integral_weights", exist_ok=True)
            
            # Save original weights - already on CPU
            torch.save(original_model_state, "path_integral_weights/original_model.pt")
            
            # Save normal finetuned weights - already on CPU
            torch.save(normal_model_state, "path_integral_weights/normal_finetuned.pt")
            
            # Save normal->mixed finetuned weights - already on CPU
            torch.save(normal_to_mixed_state, "path_integral_weights/normal_to_mixed.pt")
            
            # Save oversight finetuned weights - already on CPU
            torch.save(oversight_model_state, "path_integral_weights/oversight_finetuned.pt")
            
            # Save oversight->mixed finetuned weights - already on CPU
            torch.save(oversight_to_mixed_state, "path_integral_weights/oversight_to_mixed.pt")
        
        # Create and optimize polygonal chains
        print("Creating polygonal chains...")
        
        self.model.cpu()

        # Path 1: Original -> Normal
        print("Creating path: Original -> Normal")
        path_original_to_normal = self.create_polygonal_chain(
            start_params=original_model_state, 
            end_params=normal_model_state
        )
        
        # Path 2: Normal -> Normal+Oversight
        print("Creating path: Normal -> Normal+Oversight")
        path_normal_to_mixed = self.create_polygonal_chain(
            start_params=normal_model_state, 
            end_params=normal_to_mixed_state
        )
        
        # Optimize the polygonal chains
        print("Optimizing polygonal chains...")
        
        # Optimize Path 1
        print("Optimizing path: Original -> Normal")
        optimized_path_original_to_normal = self.optimize_polygonal_chain(
            polygonal_chain=path_original_to_normal,
            task_dataloader=normal_dataloader,
            prior_params=original_model_state,
            prior_weight=0.1,
            importance_weights=fisher_diagonal
        )
        
        # Optimize Path 2
        print("Optimizing path: Normal -> Normal+Oversight")
        optimized_path_normal_to_mixed = self.optimize_polygonal_chain(
            polygonal_chain=path_normal_to_mixed,
            task_dataloader=oversight_dataloader,  # Using oversight data as the endpoint task
            prior_params=normal_model_state,
            prior_weight=0.1,
            importance_weights=fisher_diagonal
        )

        # Save the optimized polygonal chains
        print("Saving optimized polygonal chains...")
        os.makedirs("path_integral_chains", exist_ok=True)
        
        # Convert chains to a format that can be easily saved
        torch.save({name: [t.cpu() for t in tensors] for name, tensors in optimized_path_original_to_normal.items()},
                  "path_integral_chains/optimized_path_original_to_normal.pt")
        
        torch.save({name: [t.cpu() for t in tensors] for name, tensors in optimized_path_normal_to_mixed.items()},
                  "path_integral_chains/optimized_path_normal_to_mixed.pt")
        
        # Return path integral metrics
        return {
            "status": "completed",
            "weights_saved": True,
            "normal_finetuned": "path_integral_weights/normal_finetuned.pt",
            "normal_to_mixed": "path_integral_weights/normal_to_mixed.pt",
            "oversight_finetuned": "path_integral_weights/oversight_finetuned.pt",
            "oversight_to_mixed": "path_integral_weights/oversight_to_mixed.pt",
            "optimized_path_original_to_normal": "path_integral_chains/optimized_path_original_to_normal.pt",
            "optimized_path_normal_to_mixed": "path_integral_chains/optimized_path_normal_to_mixed.pt"
        }

    def evaluate(self, normal_dataloader: DataLoader, oversight_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance on normal and oversight evaluation dataloaders.
        
        Args:
            normal_dataloader: DataLoader for normal task evaluation
            oversight_dataloader: DataLoader for oversight task evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        device = next(self.model.parameters()).device
        
        # Store current training state
        training_state = self.model.training
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        metrics = {
            "normal_loss": 0.0,
            "normal_accuracy": 0.0,
            "normal_total": 0,
            "oversight_loss": 0.0,
            "oversight_accuracy": 0.0,
            "oversight_total": 0
        }
        
        # Evaluate on normal data
        with torch.no_grad():
            for batch in normal_dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                target_token_ids = batch["target_token_ids"]
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Compute metrics
                for i in range(len(target_token_ids)):
                    # Get logits for the last token
                    seq_len = attention_mask[i].sum().item()
                    token_logits = logits[i, seq_len - 1]
                    
                    # Compute loss
                    loss = F.cross_entropy(token_logits.unsqueeze(0), target_token_ids[i].unsqueeze(0))
                    metrics["normal_loss"] += loss.item()
                    
                    # Compute accuracy
                    predicted = token_logits.argmax().item()
                    if predicted == target_token_ids[i].item():
                        metrics["normal_accuracy"] += 1.0
                        
                    metrics["normal_total"] += 1
        
        # Evaluate on oversight data
        with torch.no_grad():
            for batch in oversight_dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                target_token_ids = batch["target_token_ids"]
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Compute metrics
                for i in range(len(target_token_ids)):
                    # Get logits for the last token
                    seq_len = attention_mask[i].sum().item()
                    token_logits = logits[i, seq_len - 1]
                    
                    # Compute loss
                    loss = F.cross_entropy(token_logits.unsqueeze(0), target_token_ids[i].unsqueeze(0))
                    metrics["oversight_loss"] += loss.item()
                    
                    # Compute accuracy
                    predicted = token_logits.argmax().item()
                    if predicted == target_token_ids[i].item():
                        metrics["oversight_accuracy"] += 1.0
                        
                    metrics["oversight_total"] += 1
        
        # Calculate averages
        if metrics["normal_total"] > 0:
            metrics["normal_loss"] /= metrics["normal_total"]
            metrics["normal_accuracy"] /= metrics["normal_total"]
            
        if metrics["oversight_total"] > 0:
            metrics["oversight_loss"] /= metrics["oversight_total"]
            metrics["oversight_accuracy"] /= metrics["oversight_total"]
        
        # Calculate transferability score (how well normal task training helps with oversight task)
        metrics["transferability"] = metrics["oversight_accuracy"] / max(metrics["normal_accuracy"], 1e-8)
        
        # Log evaluation results
        print(f"\nEvaluation results:")
        print(f"  Normal task: loss = {metrics['normal_loss']:.4f}, accuracy = {metrics['normal_accuracy']:.4f}")
        print(f"  Oversight task: loss = {metrics['oversight_loss']:.4f}, accuracy = {metrics['oversight_accuracy']:.4f}")
        print(f"  Transferability: {metrics['transferability']:.4f}")
        
        # Restore model's training state
        if training_state:
            self.model.train()
        
        # Save metrics to a log file to track over time
        if not hasattr(self, 'eval_history'):
            self.eval_history = []
        
        self.eval_history.append(metrics)
        
        # Save evaluation history
        os.makedirs("path_integral_metrics", exist_ok=True)
        with open("path_integral_metrics/eval_history.json", "w") as f:
            json.dump(self.eval_history, f, indent=2)
        
        return metrics

if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--skip_fisher", action="store_true", help="Skip Fisher calculation for faster testing")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if checkpoints exist")
    args = parser.parse_args()
    
    run_experiment(PathIntegralDetector, args)

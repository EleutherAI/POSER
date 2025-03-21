import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
import sys
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from detection_strategies.path_integral.utils import get_segment_index

def create_polygonal_chain(model, start_params: Dict[str, torch.Tensor], 
                         end_params: Dict[str, torch.Tensor], 
                         num_bends: int = 3,
                         device_str: str = None) -> Dict[str, List[torch.Tensor]]:
    """
    Create a polygonal chain connecting start_params and end_params.
    
    Args:
        model: The model to use for device information
        start_params: Dictionary mapping parameter names to tensors at t=0
        end_params: Dictionary mapping parameter names to tensors at t=1
        num_bends: Number of intermediate points in the chain
        device_str: Device to use for computation
        
    Returns:
        Dictionary mapping parameter names to lists of tensors representing the chain
    """
    # Use model's device if not specified
    if device_str is None:
        device_str = next(model.parameters()).device
    
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
            intermediate = start_tensor * (1 - t) + end_tensor * t
            
            # Add some small random noise to break strict linearity
            noise = torch.randn_like(intermediate) * 1e-4
            intermediate = intermediate + noise
            
            # Add the intermediate point to the chain
            chain.append(intermediate)
        
        # Add the end tensor to complete the chain
        chain.append(end_tensor)
        
        # Store in the dictionary
        polygonal_chain[name] = chain
    
    return polygonal_chain

def interpolate_polygonal_chain(model, polygonal_chain: Dict[str, List[torch.Tensor]], 
                              t: float, final_device: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
    """
    Interpolate along the polygonal chain at position t in [0, 1].
    
    Args:
        model: The model to use for device information
        polygonal_chain: Dictionary mapping parameter names to lists of tensors
        t: Position along the chain (0.0 to 1.0)
        final_device: Device to place interpolated parameters on
        
    Returns:
        Tuple containing:
        - Dictionary mapping parameter names to interpolated tensors
        - segment_idx: Index of the first endpoint of the containing segment
        - weight_start: Weight for the start endpoint
        - weight_end: Weight for the end endpoint
    """
    # Use model's device if not specified
    if final_device is None:
        final_device = next(model.parameters()).device
    
    # Initialize the output dictionary
    interpolated_params = {}
    
    # Get the number of segments (using the first parameter)
    first_param = next(iter(polygonal_chain.values()))
    num_segments = len(first_param) - 1
    
    # Get segment index and weights
    segment_idx, weight_start, weight_end = get_segment_index(t, num_segments)
    
    # For each parameter
    for name, chain in polygonal_chain.items():
        # Skip parameters that aren't LoRA parameters
        if not name.endswith('.weight') and not name.endswith('.lora_A') and not name.endswith('.lora_B'):
            continue
            
        # Get the segment endpoints
        start_point = chain[segment_idx]
        end_point = chain[segment_idx + 1]
        
        # Linear interpolation - for LoRA parameters, we can keep these on same device
        # since they're small enough to fit in memory
        interpolated = start_point * weight_start + end_point * weight_end
        
        # Store in the output dictionary
        interpolated_params[name] = interpolated
    
    return interpolated_params, segment_idx, weight_start, weight_end



def optimize_polygonal_chain(model, polygonal_chain: Dict[str, List[torch.Tensor]],
                           task_dataloader: DataLoader,
                           num_steps: int = 1000,
                           learning_rate: float = 1e-2,
                           weight_decay: float = 0.01,
                           device_str: str = None,
                           save_loss_dir: Optional[str] = None,
                           path_name: Optional[str] = None,
                           log_interval: int = 10) -> Dict[str, List[torch.Tensor]]:
    """
    Optimize the polygonal chain to minimize the combined loss.
    
    Args:
        model: The model to optimize
        polygonal_chain: Dictionary mapping parameter names to lists of tensors
        task_dataloader: DataLoader for the task at endpoint
        num_steps: Number of optimization steps
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for AdamW optimizer
        device_str: Device to use for optimization
        save_loss_dir: Directory to save loss values
        path_name: Name of the path for the loss data file
        
    Returns:
        Optimized polygonal chain
    """
    # Use model's device if not specified
    if device_str is None:
        device_str = next(model.parameters()).device

    model.to(device_str)

    # Create a copy of the polygonal chain that we'll optimize
    # Only include LoRA parameters
    optimized_chain = {}
    for name, chain in polygonal_chain.items():
        # Only include LoRA parameters (lora_A, lora_B, or adapter weights)
        if 'lora' in name or 'adapter' in name:
            optimized_chain[name] = [
                chain[0].to(device_str),  # Start endpoint
                *[tensor.clone().detach().to(device_str).requires_grad_(True) for tensor in chain[1:-1]],  # Intermediate points
                chain[-1].to(device_str)  # End endpoint
            ]
    
    # Create parameter groups for optimization - one per segment
    segment_params = []
    num_points = len(next(iter(optimized_chain.values())))
    
    # Create separate parameter groups for each segment
    for j in range(1, num_points-1):  # Skip endpoints
        segment_params.append([])
        for name, chain in optimized_chain.items():
            segment_params[-1].append(chain[j])
    
    task_dataloader_iter = iter(task_dataloader)

    # Create optimizers for each segment - using AdamW instead of SGD
    optimizers = [torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay) for params in segment_params]

    # Initialize tracking of L2 distances between corners
    corner_distances_history = []
    
    # Optimization loop
    for step in tqdm(range(num_steps), desc="Optimizing polygonal chain"):
        # Sample t uniformly from (0, 1)
        t = torch.rand(1).item()
        
        # Interpolate along the polygonal chain
        interpolated_params, segment_idx, weight_start, weight_end = interpolate_polygonal_chain(
            model, optimized_chain, t, device_str
        )
        
        for name, param in model.named_parameters():
            if name in interpolated_params and param.requires_grad:
                # This creates a new tensor that maintains gradient connections
                param.data = interpolated_params[name]
        
        # Print L2 distances from interpolated params to endpoints
        if step % log_interval == 0:
            start_point = optimized_chain[list(optimized_chain.keys())[0]][segment_idx]
            end_point = optimized_chain[list(optimized_chain.keys())[0]][segment_idx + 1]
            interp_point = interpolated_params[list(interpolated_params.keys())[0]]
            interp_point_model = model.get_parameter(list(interpolated_params.keys())[0])
            
            start_dist = torch.norm(interp_point - start_point).item()
            end_dist = torch.norm(interp_point - end_point).item()
            interp_dist_model = torch.norm(interp_point_model - interp_point).item()
            print(f"L2 distances - to start: {start_dist:.4f}, to end: {end_dist:.4f}, to model: {interp_dist_model:.4f}")
            print(f"Weights - start: {weight_start:.4f}, end: {weight_end:.4f}")

        # Get a batch from the task dataloader
        batch = None
        try:
            batch = next(task_dataloader_iter)
        except StopIteration:
            task_dataloader_iter = iter(task_dataloader)
            batch = next(task_dataloader_iter)
        
        if batch is None:
            print("Warning: Empty task dataloader")
            continue
        
        # Zero gradients
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(device_str) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_token_ids = batch["target_token_ids"]

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute task loss
            token_logits = logits[:, -1]
            task_loss = F.cross_entropy(token_logits, target_token_ids)
            
            
            # We only use task_loss for optimization - weight decay is handled by the optimizer
            loss = t * task_loss

        # Backward pass
        loss.backward()
        
        # Manually distribute gradients to chain endpoints based on chain rule
        # For interpolated params: interpolated = start_point * weight_start + end_point * weight_end
        # Chain rule: ∂L/∂start = weight_start * ∂L/∂interpolated, ∂L/∂end = weight_end * ∂L/∂interpolated
        for name, param in model.named_parameters():
            if name in optimized_chain and param.grad is not None and param.requires_grad:
                # Get start and end points of the current segment
                start_point = optimized_chain[name][segment_idx]
                end_point = optimized_chain[name][segment_idx + 1]
                
                # Skip if endpoints are not in the optimization set (endpoints 0 and n)
                start_requires_grad = segment_idx > 0
                end_requires_grad = segment_idx < (len(optimized_chain[name]) - 2)
                
                if start_requires_grad or end_requires_grad:
                    # Get gradient from the model parameter
                    param_grad = param.grad.clone()
                    
                    # Distribute gradient to start point (if not fixed)
                    if start_requires_grad:
                        if optimized_chain[name][segment_idx].grad is None:
                            optimized_chain[name][segment_idx].grad = weight_start * param_grad
                        else:
                            optimized_chain[name][segment_idx].grad += weight_start * param_grad
                    
                    # Distribute gradient to end point (if not fixed)
                    if end_requires_grad:
                        if optimized_chain[name][segment_idx + 1].grad is None:
                            optimized_chain[name][segment_idx + 1].grad = weight_end * param_grad
                        else:
                            optimized_chain[name][segment_idx + 1].grad += weight_end * param_grad


        # Apply updates for relevant segments
        for optimizer_idx in [segment_idx, segment_idx + 1]:
            # Skip endpoints
            if optimizer_idx == 0 or optimizer_idx == num_points - 1:
                continue
                
            # Adjust index to account for skipping endpoints
            optimizer_idx -= 1
            
            # Apply optimizer step
            optimizers[optimizer_idx].step()
            optimizers[optimizer_idx].zero_grad()
        
        # Print progress occasionally
        if step % log_interval == 0:
            # Log the loss values including prior loss for reporting
            print(f"Step {step}, Task Loss: {task_loss:.6f}, t: {t:.2f}")
            
            # Calculate L2 distances between all corners of the chain
            num_corners = len(next(iter(optimized_chain.values())))
            
            # Initialize accumulator for each segment
            segment_distances = [0.0] * (num_corners - 1)
            segment_counts = [0] * (num_corners - 1)
            
            for name, chain in optimized_chain.items():
                for i in range(num_corners - 1):
                    corner1 = chain[i]
                    corner2 = chain[i + 1]
                    segment_dist = torch.norm(corner2 - corner1).item()
                    segment_distances[i] += segment_dist
                    segment_counts[i] += 1
            
            # Calculate average distance for each segment
            avg_segment_distances = [
                dist / count if count > 0 else 0.0
                for dist, count in zip(segment_distances, segment_counts)
            ]
            
            # Store for potential plotting
            corner_distances_history.append({
                'step': step,
                'segment_distances': avg_segment_distances
            })
            
            # Print segment distances
            print(f"Step {step}, Loss: {loss:.6f}, t: {t:.2f}")
            print(f"Segment L2 distances: {', '.join([f'{d:.4f}' for d in avg_segment_distances])}")
                        
            
    # Save corner distances history if requested
    if save_loss_dir is not None and path_name is not None:
        # Create the directory if it doesn't exist
        os.makedirs(save_loss_dir, exist_ok=True)
        
        # Convert path name to valid filename
        safe_path_name = path_name.replace(' -> ', '_to_')
        safe_path_name = safe_path_name.replace('+', '_and_')
        
        # Save the corner distances history
        corner_distances_data = {
            "path_name": path_name,
            "corner_distances_history": corner_distances_history
        }
        
        with open(f"{save_loss_dir}/corner_distances_{safe_path_name}.json", 'w') as f:
            json.dump(corner_distances_data, f, indent=2)
    
    # If save_loss_dir is provided, evaluate loss along the path
    if save_loss_dir is not None and path_name is not None:
        # Create equally spaced points along the path
        num_eval_points = 100
        t_values = np.linspace(0.0, 1.0, num_eval_points)
        total_losses = []
        task_losses = []
        prior_losses = []
        
        print(f"Evaluating loss along path: {path_name}")
        for t in tqdm(t_values, desc="Evaluating path loss"):
            # Interpolate along the polygonal chain
            interpolated_params, segment_idx, _, _ = interpolate_polygonal_chain(
                model, optimized_chain, t, device_str
            )
            
            # Update model with interpolated LoRA parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in interpolated_params and param.requires_grad:
                        param.data.copy_(interpolated_params[name])
            
            # Get a batch from the task dataloader
            batch = None
            try:
                batch = next(task_dataloader_iter)
            except StopIteration:
                task_dataloader_iter = iter(task_dataloader)
                batch = next(task_dataloader_iter)
            
            if batch is None:
                continue
            
            # Move batch to device
            batch = {k: v.to(device_str) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass to compute loss
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            target_token_ids = batch["target_token_ids"]
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Compute task loss
                    token_logits = logits[:, -1]
                    curr_task_loss = F.cross_entropy(token_logits, target_token_ids).item()
                    
                    # Compute prior loss
                    curr_prior_loss = 0.0
                    num_params = 0
                    for name, param in model.named_parameters():
                        if ('lora' in name or 'adapter' in name) and param.requires_grad:
                            curr_prior_loss += torch.sum(param.pow(2)).item()
                            num_params += param.numel()
                    
                    if num_params > 0:
                        curr_prior_loss /= num_params
                    # Combined loss
                    curr_total_loss = t * curr_task_loss + prior_weight * curr_prior_loss
            
            total_losses.append(curr_total_loss)
            task_losses.append(curr_task_loss)
            prior_losses.append(curr_prior_loss)
        
        # Save loss values
        os.makedirs(save_loss_dir, exist_ok=True)
        
        # Convert path name to valid filename
        safe_path_name = path_name.replace(' -> ', '_to_')
        safe_path_name = safe_path_name.replace('+', '_and_')
        
        loss_data = {
            "path_name": path_name,
            "t_values": t_values.tolist(),
            "total_losses": total_losses,
            "task_losses": task_losses,
            "prior_losses": prior_losses
        }
        
        with open(f"{save_loss_dir}/path_loss_{safe_path_name}.json", 'w') as f:
            json.dump(loss_data, f, indent=2)
    
    return optimized_chain

def compute_path_integral(model, polygonal_chain: Dict[str, List[torch.Tensor]],
                        target_dataloader: DataLoader,
                        num_samples: int = 1000,
                        device_str: str = None) -> Dict[str, float]:
    """
    Compute the actual path integral by sampling points along the optimized path.
    
    The path integral is approximated by:
    1. Sampling N points along the path (segment chosen randomly, position within segment chosen uniformly)
    2. Computing the gradient of the loss w.r.t. the target dataset at each point
    3. Computing the inner product of the gradient with the derivative of the path at that point
    4. Averaging these values across all sampled points
    
    Args:
        model: The model to compute path integral for
        polygonal_chain: Dictionary mapping parameter names to lists of tensors representing the chain
        target_dataloader: DataLoader for the target dataset
        prior_weight: Weight of the prior loss
        num_samples: Number of points to sample along the path
        device_str: Device to use for computation
        
    Returns:
        Dictionary of path integral metrics
    """
    # Use model's device if not specified
    if device_str is None:
        device_str = next(model.parameters()).device
    
    model.to(device_str)
    model.train()  # Need to set to train mode to compute gradients
    
    # Get the number of segments in the path
    first_param = next(iter(polygonal_chain.values()))
    num_segments = len(first_param) - 1
    
    # Initialize accumulator for path integral
    path_integral_sum = 0.0
    valid_samples = 0
    
    # Sample N points along the path
    for sample_idx in tqdm(range(num_samples), desc="Computing path integral"):
        # Sample a segment randomly
        segment_idx = random.randint(0, num_segments - 1)
        
        # Sample a position within the segment uniformly
        segment_pos = random.random()  # 0.0 to 1.0
        
        # Convert to overall t value (0 to 1 across entire path)
        t = (segment_idx + segment_pos) / num_segments
        
        # Interpolate along the polygonal chain to get model parameters at position t
        interpolated_params, actual_segment_idx, _, _ = interpolate_polygonal_chain(
            model=model,
            polygonal_chain=polygonal_chain,
            t=t,
            final_device=device_str
        )
        
        # Load parameters into model - only update LoRA parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in interpolated_params and param.requires_grad:
                    param.data.copy_(interpolated_params[name])
        
        # Get a batch from the target dataloader
        batch = None
        for batch in target_dataloader:
            # We just need one batch
            break
        
        if batch is None:
            print("Warning: Empty target dataloader")
            continue
        
        # Zero gradients
        model.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(device_str) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            target_token_ids = batch["target_token_ids"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss - focus on the target token
            token_logits = logits[:, -1]
            loss = F.cross_entropy(token_logits, target_token_ids)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Compute integral contribution for this sample
        sample_integral = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None and name in polygonal_chain and param.requires_grad:
                # Get the gradient at this point
                grad = param.grad.detach()
                
                # For the current segment, compute the path derivative (direction vector)
                segment_start = polygonal_chain[name][actual_segment_idx].to(device_str)
                segment_end = polygonal_chain[name][actual_segment_idx + 1].to(device_str)
                segment_direction = segment_end - segment_start
                
                # Scale by the number of segments to get derivative w.r.t. t ∈ [0,1]
                segment_derivative = segment_direction * num_segments
                
                # Compute the inner product of gradient and path derivative
                inner_product = torch.sum(grad * segment_derivative)
                
                sample_integral += inner_product.item()
                param_count += param.numel()
        
        # Normalize by parameter count
        if param_count > 0:
            sample_integral /= param_count
        
        path_integral_sum += sample_integral
        valid_samples += 1
    
    # Compute average path integral
    path_integral_value = 0.0
    if valid_samples > 0:
        path_integral_value = path_integral_sum / valid_samples
    
    # Restore model's evaluation mode
    model.eval()
    
    return {
        "path_integral_value": path_integral_value,
        "valid_samples": valid_samples
    } 
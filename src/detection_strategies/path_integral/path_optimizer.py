import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
import sys

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
            intermediate = start_tensor.to(device_str) * (1 - t) + end_tensor.to(device_str) * t
            
            # Add some small random noise to break strict linearity
            noise = torch.randn_like(intermediate, device=device_str) * 1e-4
            intermediate = intermediate + noise
            
            # Add the intermediate point to the chain
            chain.append(intermediate.cpu())
        
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
    
    # Use pinned memory for faster CPU->GPU transfers
    use_pinned = torch.cuda.is_available()
    
    # For each parameter
    for name, chain in polygonal_chain.items():
        # Get the segment endpoints
        start_point = chain[segment_idx]
        end_point = chain[segment_idx + 1]
        
        # Pin memory if using CUDA for faster transfers
        if use_pinned:
            # Only pin if not already in pinned memory
            if not start_point.is_pinned():
                start_point = start_point.pin_memory()
            if not end_point.is_pinned():
                end_point = end_point.pin_memory()
        
        # Linear interpolation
        interpolated = start_point.to(final_device) * weight_start + end_point.to(final_device) * weight_end
        
        # Store in the output dictionary
        interpolated_params[name] = interpolated
    
    return interpolated_params, segment_idx, weight_start, weight_end

def compute_combined_loss(model, parameters: Dict[str, torch.Tensor], 
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
        model: The model to compute loss for
        parameters: Current model parameters
        task_dataloader: DataLoader for the task (endpoint task)
        prior_params: Parameters of the prior model
        prior_weight: Weight of the prior loss
        t: Position along the path from 0 to 1
        importance_weights: Optional dictionary mapping parameter names to importance weights
        device: Device to use for computation
        
    Returns:
        Combined loss value
    """
    # Use default device if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Load parameters directly to the temporary model on the target device
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in parameters:
                # Create parameter on correct device from the start
                param.data = parameters[name].to(device)
    
    # Move the temporary model to the target device 
    model.to(device)
    model.train()
    
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
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute loss - focus on the target token
        # Get logits for the last token
        token_logits = logits[:, -1]
        
        # Compute cross-entropy loss for the target token
        batch_loss = F.cross_entropy(token_logits, target_token_ids)
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
    
    for name, param in model.named_parameters():
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
    model.eval()
    model.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return combined_loss

def optimize_polygonal_chain(model, polygonal_chain: Dict[str, List[torch.Tensor]],
                           task_dataloader: DataLoader,
                           prior_params: Dict[str, torch.Tensor],
                           prior_weight: float,
                           importance_weights: Dict[str, torch.Tensor] = None,
                           num_steps: int = 20,
                           learning_rate: float = 1e-4,
                           device_str: str = None) -> Dict[str, List[torch.Tensor]]:
    """
    Optimize the polygonal chain to minimize the combined loss.
    
    Args:
        model: The model to optimize
        polygonal_chain: Dictionary mapping parameter names to lists of tensors
        task_dataloader: DataLoader for the task at endpoint
        prior_params: Parameters of the prior model
        prior_weight: Weight of the prior loss
        importance_weights: Optional dictionary mapping parameter names to importance weights
        num_steps: Number of optimization steps
        learning_rate: Learning rate for optimization
        device_str: Device to use for optimization
        
    Returns:
        Optimized polygonal chain
    """
    # Use model's device if not specified
    if device_str is None:
        device_str = next(model.parameters()).device
    
    # Create a copy of the polygonal chain that we'll optimize
    optimized_chain = {}
    # Use pinned memory for faster transfers if using CUDA
    use_pinned = torch.cuda.is_available()

    for name, chain in polygonal_chain.items():
        if use_pinned:
            # Pin the tensors that will be frequently moved to GPU
            optimized_chain[name] = [
                chain[0].cpu(),  # Keep endpoints on CPU
                *[tensor.clone().detach().pin_memory() for tensor in chain[1:-1]],  # Pin intermediate points
                chain[-1].cpu()  # Keep endpoints on CPU
            ]
        else:
            optimized_chain[name] = [
                chain[0].cpu(),
                *[tensor.clone().detach().cpu() for tensor in chain[1:-1]], 
                chain[-1].cpu()
            ]
    
    # Create parameter groups for optimization - one per segment
    segment_params = []
    num_points = len(next(iter(optimized_chain.values())))
    
    # Create separate parameter groups for each segment
    for j in range(1, num_points-1):  # Skip endpoints
        segment_params.append([])
        for name, chain in optimized_chain.items():
            segment_params[-1].append(chain[j])
    
    # Create optimizers for each segment
    optimizers = [torch.optim.Adam(params, lr=learning_rate) for params in segment_params]
    
    # Optimization loop
    for step in tqdm(range(num_steps), desc="Optimizing polygonal chain"):
        # Sample t uniformly from (0, 1)
        t = torch.rand(1).item()
        
        # Interpolate along the polygonal chain
        interpolated_params, segment_idx, weight_start, weight_end = interpolate_polygonal_chain(
            model, optimized_chain, t, device_str
        )
        
        # Create a copy of interpolated parameters with requires_grad=True
        interpolated_params_with_grad = {}
        for name, param in interpolated_params.items():
            interpolated_params_with_grad[name] = param.requires_grad_(True)
        
        # Compute loss for interpolated parameters
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = compute_combined_loss(
                model=model,
                parameters=interpolated_params_with_grad,
                task_dataloader=task_dataloader,
                prior_params=prior_params,
                prior_weight=prior_weight,
                t=t,
                importance_weights=importance_weights,
                device=device_str
            )
        
        # Compute gradients
        loss_tensor = torch.tensor(loss, requires_grad=True, device=device_str)
        loss_tensor.backward()
        
        # Distribute gradients to endpoints based on weights
        for name, param in interpolated_params_with_grad.items():
            if param.grad is not None:
                # Apply gradients to endpoint j with weight_start
                j = segment_idx
                if j > 0 and j < len(optimized_chain[name]) - 1:  # Skip fixed endpoints
                    optimized_chain[name][j] = optimized_chain[name][j].to(device_str)
                    if optimized_chain[name][j].grad is None:
                        optimized_chain[name][j].grad = weight_start * param.grad.clone()
                    else:
                        optimized_chain[name][j].grad += weight_start * param.grad.clone()
                    optimized_chain[name][j] = optimized_chain[name][j].cpu()
                
                # Apply gradients to endpoint j+1 with weight_end
                j = segment_idx + 1
                if j > 0 and j < len(optimized_chain[name]) - 1:  # Skip fixed endpoints
                    optimized_chain[name][j] = optimized_chain[name][j].to(device_str)
                    if optimized_chain[name][j].grad is None:
                        optimized_chain[name][j].grad = weight_end * param.grad.clone()
                    else:
                        optimized_chain[name][j].grad += weight_end * param.grad.clone() 
                    optimized_chain[name][j] = optimized_chain[name][j].cpu()

        # Apply updates
        for optimizer_idx in [segment_idx, segment_idx + 1]:
            # Skip endpoints
            if optimizer_idx == 0 or optimizer_idx == num_points - 1:
                continue

            # Adjust index to account for skipping endpoints
            optimizer_idx -= 1
            
            # Apply optimizer step
            optimizers[optimizer_idx].zero_grad()
            optimizers[optimizer_idx].step()

        
        # Print progress occasionally
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
        
        # Clear tensors from GPU and free memory
        del interpolated_params
        del interpolated_params_with_grad
        del loss_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return optimized_chain

def compute_path_integral(model, polygonal_chain: Dict[str, List[torch.Tensor]],
                        target_dataloader: DataLoader,
                        prior_params: Dict[str, torch.Tensor],
                        prior_weight: float,
                        num_samples: int = 10,
                        importance_weights: Dict[str, torch.Tensor] = None,
                        device_str: str = None) -> Dict[str, float]:
    """
    Compute the actual path integral by sampling points along the optimized path.
    
    The path integral is approximated by:
    1. Sampling N points along the path (segment chosen randomly, position within segment chosen uniformly)
    2. Computing the gradient of the loss w.r.t. the target dataset at each point
    3. Computing t * gradient for each point
    4. Averaging these values across all sampled points
    
    Args:
        model: The model to compute path integral for
        polygonal_chain: Dictionary mapping parameter names to lists of tensors representing the chain
        target_dataloader: DataLoader for the target dataset
        prior_params: Parameters of the prior model
        prior_weight: Weight of the prior loss
        num_samples: Number of points to sample along the path
        importance_weights: Optional dictionary mapping parameter names to importance weights
        device_str: Device to use for computation
        
    Returns:
        Dictionary of path integral metrics
    """
    # Use model's device if not specified
    if device_str is None:
        device_str = next(model.parameters()).device
    
    model.train()  # Need to set to train mode to compute gradients
    
    # Get the number of segments in the path
    first_param = next(iter(polygonal_chain.values()))
    num_segments = len(first_param) - 1
    
    # Initialize accumulator for t * gradient
    t_times_grad_sum = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
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
        interpolated_params, _, _, _ = interpolate_polygonal_chain(
            model=model,
            polygonal_chain=polygonal_chain,
            t=t,
            final_device=device_str
        )
        
        # Load parameters into model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in interpolated_params:
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
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_token_ids = batch["target_token_ids"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute loss - focus on the target token
        token_logits = logits[:, -1]
        
        # Compute cross-entropy loss for the target token
        loss = F.cross_entropy(token_logits, target_token_ids)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Accumulate t * gradient for each parameter
        for name, param in model.named_parameters():
            if param.grad is not None:
                t_times_grad_sum[name] += t * param.grad.detach().clone().cpu()
        
        valid_samples += 1
        
        # Clear GPU memory
        del interpolated_params
        del loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute average t * gradient
    t_times_grad_avg = {}
    if valid_samples > 0:
        for name in t_times_grad_sum:
            t_times_grad_avg[name] = t_times_grad_sum[name] / valid_samples
    
    # Compute and return path integral metrics
    path_integral_value = 0.0
    total_params = 0
    
    for name, grad in t_times_grad_avg.items():
        path_integral_value += torch.sum(torch.abs(grad)).item()
        total_params += grad.numel()
    
    if total_params > 0:
        path_integral_value /= total_params
    
    # Restore model's evaluation mode
    model.eval()
    
    return {
        "path_integral_value": path_integral_value,
        "valid_samples": valid_samples,
        "t_times_grad_avg": {name: tensor.mean().item() for name, tensor in t_times_grad_avg.items()}
    } 
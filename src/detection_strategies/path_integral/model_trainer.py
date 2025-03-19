import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train_model(model, optimizer, primary_dataloader,
               steps=1, eval_data=None, eval_steps=100,
               device_str=None, evaluator=None):
    """
    Train the model using the provided optimizer and dataloaders.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use for training
        primary_dataloader: The main dataloader for training
        steps: Number of steps/epochs to train for
        eval_data: Tuple of (normal_eval_data, oversight_eval_data) for evaluation
        eval_steps: How often to evaluate the model
        device_str: Device to use for training
        evaluator: Optional function to evaluate the model
        
    Returns:
        Dictionary of model parameters after training
    """
    # Use model's device if not specified
    if device_str is None:
        device_str = next(model.parameters()).device
    
    model.train()
    
    for step in tqdm(range(steps), desc="Training model"):
        # Periodic evaluation if requested
        if eval_data and evaluator and step > 0 and step % eval_steps == 0:
            normal_eval_dataloader = DataLoader(
                eval_data[0], batch_size=2, shuffle=False, 
                collate_fn=eval_data[0].collate_fn
            )
            oversight_eval_dataloader = DataLoader(
                eval_data[1], batch_size=2, shuffle=False,
                collate_fn=eval_data[1].collate_fn
            )
            evaluator(normal_eval_dataloader, oversight_eval_dataloader)
        
        # Get batch from primary dataloader
        batch_sources = []
        for i, batch in enumerate(primary_dataloader):
            batch_sources.append(batch)
            if i >= 0:  # Just process one batch per step
                break
        
        # Process each batch
        for batch in batch_sources:
            # Move batch to device
            batch = {k: v.to(device_str) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with autocast
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Forward pass
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                target_token_ids = batch["target_token_ids"]
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
    return {name: param.detach().clone().cpu() for name, param in model.named_parameters()} 
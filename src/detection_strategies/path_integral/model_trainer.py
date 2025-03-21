import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType

def train_model(model, optimizer=None, primary_dataloader=None,
               steps=1, eval_data=None, eval_steps=100,
               device_str=None, evaluator=None,
               learning_rate=5e-5, weight_decay=0.01):
    """
    Train the model using AdamW optimizer. Assumes model already has LoRA adapters.
    
    Args:
        model: The model to train (with LoRA adapters already applied)
        optimizer: Optional pre-configured optimizer
        primary_dataloader: The main dataloader for training
        steps: Number of steps/epochs to train for
        eval_data: Tuple of (normal_eval_data, oversight_eval_data) for evaluation
        eval_steps: How often to evaluate the model
        device_str: Device to use for training
        evaluator: Optional function to evaluate the model
        learning_rate: Learning rate for the AdamW optimizer
        weight_decay: Weight decay for the AdamW optimizer
        
    Returns:
        Dictionary of model parameters after training
    """
    # Use model's device if not specified
    if device_str is None:
        device_str = next(model.parameters()).device
    
    # Create AdamW optimizer if not provided
    if optimizer is None:
        # Only optimize trainable parameters (important for LoRA efficiency)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    model.train()
    
    # Add loss tracking
    total_loss = 0    
    # Create an iterator from the dataloader
    primary_dataloader_iter = iter(primary_dataloader)
    
    # Use a running average for smoother loss display
    running_loss = 0
    
    progress_bar = tqdm(range(steps), desc="Training model")
    for step in progress_bar:
        # Periodic evaluation if requested
        if eval_data and evaluator and step > 0 and step % eval_steps == 0:
            model.eval()  # Set to eval mode before evaluation
            normal_eval_dataloader = DataLoader(
                eval_data[0], batch_size=2, shuffle=False, 
                collate_fn=eval_data[0].collate_fn
            )
            oversight_eval_dataloader = DataLoader(
                eval_data[1], batch_size=2, shuffle=False,
                collate_fn=eval_data[1].collate_fn
            )
            eval_results = evaluator(normal_eval_dataloader, oversight_eval_dataloader)
            print(f"Step {step} - Eval results: {eval_results}")
            model.train()  # Set back to train mode
        
        # Get next batch, resetting iterator if needed
        try:
            batch = next(primary_dataloader_iter)
        except StopIteration:
            # Reset the iterator when we've gone through the entire dataset
            primary_dataloader_iter = iter(primary_dataloader)
            batch = next(primary_dataloader_iter)
        
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
            # Get logits for the last token
            seq_len = attention_mask.sum().item()
            token_logits = logits[:, - 1]
            # Compute cross-entropy loss for the target token
            loss = F.cross_entropy(token_logits, target_token_ids)
        
        # Track loss
        total_loss += loss.item()
        running_loss = 0.9 * running_loss + 0.1 * loss.item() if step > 0 else loss.item()
        
        # Update progress bar description with current loss
        if step % 10 == 0:
            progress_bar.set_description(f"Training model, loss: {running_loss:.4f}")

        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
    
    # Return the trained LoRA weights
    return {name: param.detach().clone().cpu() 
            for name, param in model.named_parameters() 
            if param.requires_grad} 
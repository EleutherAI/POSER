import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json

def evaluate_model(model, normal_dataloader: DataLoader, oversight_dataloader: DataLoader, 
                  device_str: str = None, save_path: str = "path_integral_metrics") -> dict:
    """
    Evaluate model performance on normal and oversight evaluation dataloaders.
    
    Args:
        model: The model to evaluate
        normal_dataloader: DataLoader for normal task evaluation
        oversight_dataloader: DataLoader for oversight task evaluation
        device_str: Device to use for evaluation
        save_path: Directory to save evaluation metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Use model's device if not specified
    if device_str is None:
        device_str = next(model.parameters()).device
    
    # Store current training state
    training_state = model.training
    
    # Set model to evaluation mode
    model.eval()
    
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
            batch = {k: v.to(device_str) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            target_token_ids = batch["target_token_ids"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute metrics
            # Get logits for the last token
            token_logits = logits[:, -1]
            
            # Compute loss
            loss = F.cross_entropy(token_logits, target_token_ids)
            metrics["normal_loss"] += loss.item()
            
            # Compute accuracy
            predicted = token_logits.argmax(1)
            metrics["normal_accuracy"] += (predicted == target_token_ids).float().mean().item()
                
            metrics["normal_total"] += 1
    
    # Evaluate on oversight data
    with torch.no_grad():
        for batch in oversight_dataloader:
            # Move batch to device
            batch = {k: v.to(device_str) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            target_token_ids = batch["target_token_ids"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute metrics
            # Get logits for the last token
            token_logits = logits[:, -1]
            
            # Compute loss
            loss = F.cross_entropy(token_logits, target_token_ids)
            metrics["oversight_loss"] += loss.item()
            
            # Compute accuracy
            predicted = token_logits.argmax(1)
            metrics["oversight_accuracy"] += (predicted == target_token_ids).float().mean().item()
                
            metrics["oversight_total"] += 1
    
    # Calculate averages
    if metrics["normal_total"] > 0:
        metrics["normal_loss"] /= metrics["normal_total"]
        metrics["normal_accuracy"] /= metrics["normal_total"]
        
    if metrics["oversight_total"] > 0:
        metrics["oversight_loss"] /= metrics["oversight_total"]
        metrics["oversight_accuracy"] /= metrics["oversight_total"]

    
    # Log evaluation results
    print(f"\nEvaluation results:")
    print(f"  Normal task: loss = {metrics['normal_loss']:.4f}, accuracy = {metrics['normal_accuracy']:.4f}")
    print(f"  Oversight task: loss = {metrics['oversight_loss']:.4f}, accuracy = {metrics['oversight_accuracy']:.4f}")
    
    # Restore model's training state
    if training_state:
        model.train()
    
    # Save metrics to file
    os.makedirs(save_path, exist_ok=True)
    
    # Save to history file if it exists
    history_file = os.path.join(save_path, "eval_history.json")
    history = []
    
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            # If the file is corrupted, start with empty history
            history = []
    
    history.append(metrics)
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    
    return metrics

def log_and_save_metrics(metrics: dict, save_path: str = "path_integral_metrics") -> None:
    """
    Log evaluation metrics and save to file.
    
    Args:
        metrics: Dictionary of metrics to log and save
        save_path: Directory to save metrics
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save metrics to file
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2) 
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

class PileDataset(Dataset):
    """Dataset for samples from the Pile."""
    
    def __init__(self, tokenizer, max_length=512, num_samples=1000, split="train"):
        """
        Initialize the Pile dataset.
        
        Args:
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
            num_samples: Number of samples to load (None for all)
            split: Dataset split to use ('train', 'validation', or 'test')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the Pile dataset from Hugging Face
        print(f"Loading the Pile dataset (split={split}, num_samples={num_samples})...")
        self.dataset = load_dataset("EleutherAI/pile", split=split, streaming=True)
        
        # Take a subset of samples if specified
        if num_samples is not None:
            self.samples = []
            for i, sample in enumerate(self.dataset):
                if i >= num_samples:
                    break
                self.samples.append(sample["text"])
        else:
            # This would load the entire dataset, which is very large
            self.samples = [sample["text"] for sample in self.dataset]
        
        print(f"Loaded {len(self.samples)} samples from the Pile dataset")
        
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
    
    def get_task_data(self, prompts: List[str], responses: List[str]) -> DataLoader:
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
        
        # Create dataloader
        batch_size = 2  # Small batch size to avoid OOM
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        return dataloader
    
    def create_pile_dataloader(self, num_samples=1000, batch_size=2):
        """
        Create a DataLoader for samples from the Pile dataset.
        
        Args:
            num_samples: Number of samples to use
            batch_size: Batch size for the DataLoader
            
        Returns:
            DataLoader for Pile samples
        """
        # Create dataset from the Pile
        dataset = PileDataset(self.tokenizer, num_samples=num_samples)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        return dataloader
    
    def compute_dependence(self, normal_dataloader: DataLoader, oversight_dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute path integral metrics between two tasks using their DataLoaders.
        
        Args:
            normal_dataloader: DataLoader for normal task
            oversight_dataloader: DataLoader for oversight task
            
        Returns:
            Dictionary of path integral metrics
        """
        device = next(self.model.parameters()).device
        
        # Step 1: Create a Pile dataset for computing Fisher diagonal
        print("Creating Pile dataset...")
        pile_dataloader = self.create_pile_dataloader(num_samples=1000)
        
        # Step 2: Compute the diagonal Fisher approximation
        print("Computing Fisher diagonal approximation...")
        fisher_diagonal = compute_fisher_matrix(
            model=self.model,
            data_loader=pile_dataloader,
            num_samples=1000,
            device=device
        )
        
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
        
        # Train for a few steps on normal data
        self.model.train()
        normal_steps = 100  # Adjust as needed
        
        for _ in tqdm(range(normal_steps)):
            for batch in normal_dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
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
                optimizer_normal.step()
                optimizer_normal.zero_grad()
        
        # Save the model weights after normal finetuning
        normal_model_state = {name: param.detach().clone().cpu() for name, param in self.model.named_parameters()}
        
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
        
        # Create a combined dataloader that alternates between normal and oversight
        # For simplicity, we'll just alternate between the two dataloaders
        mixed_steps = 100  # Adjust as needed
        
        normal_iter = iter(normal_dataloader)
        oversight_iter = iter(oversight_dataloader)
        
        self.model.train()
        for _ in tqdm(range(mixed_steps)):
            # Try to get a batch from normal dataloader
            try:
                normal_batch = next(normal_iter)
            except StopIteration:
                normal_iter = iter(normal_dataloader)
                normal_batch = next(normal_iter)
                
            # Try to get a batch from oversight dataloader
            try:
                oversight_batch = next(oversight_iter)
            except StopIteration:
                oversight_iter = iter(oversight_dataloader)
                oversight_batch = next(oversight_iter)
            
            # Alternate between normal and oversight batches
            for batch in [normal_batch, oversight_batch]:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
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
                optimizer_mixed.step()
                optimizer_mixed.zero_grad()
        
        # Save the model weights after mixed finetuning
        normal_to_mixed_state = {name: param.detach().clone().cpu() for name, param in self.model.named_parameters()}
        
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
        
        # Train for a few steps on oversight data
        self.model.train()
        oversight_steps = 100  # Adjust as needed
        
        for _ in tqdm(range(oversight_steps)):
            for batch in oversight_dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
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
                optimizer_oversight.step()
                optimizer_oversight.zero_grad()
        
        # Save the model weights after oversight finetuning
        oversight_model_state = {name: param.detach().clone().cpu() for name, param in self.model.named_parameters()}
        
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
        
        # Reset iterators
        normal_iter = iter(normal_dataloader)
        oversight_iter = iter(oversight_dataloader)
        
        self.model.train()
        for _ in tqdm(range(mixed_steps)):
            # Try to get a batch from normal dataloader
            try:
                normal_batch = next(normal_iter)
            except StopIteration:
                normal_iter = iter(normal_dataloader)
                normal_batch = next(normal_iter)
                
            # Try to get a batch from oversight dataloader
            try:
                oversight_batch = next(oversight_iter)
            except StopIteration:
                oversight_iter = iter(oversight_dataloader)
                oversight_batch = next(oversight_iter)
            
            # Alternate between oversight and normal batches
            for batch in [oversight_batch, normal_batch]:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
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
                optimizer_mixed_reverse.step()
                optimizer_mixed_reverse.zero_grad()
        
        # Save the model weights after mixed finetuning (reverse direction)
        oversight_to_mixed_state = {name: param.detach().clone().cpu() for name, param in self.model.named_parameters()}
        
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
        
        # Return a placeholder result
        # In the future, this would compute actual path integral metrics
        return {
            "status": "completed_first_stage",
            "weights_saved": True,
            "normal_finetuned": "path_integral_weights/normal_finetuned.pt",
            "normal_to_mixed": "path_integral_weights/normal_to_mixed.pt",
            "oversight_finetuned": "path_integral_weights/oversight_finetuned.pt",
            "oversight_to_mixed": "path_integral_weights/oversight_to_mixed.pt"
        }

if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument("--output_prefix", type=str, default="path_integral_results")
    args = parser.parse_args()
    
    run_experiment(PathIntegralDetector, args)

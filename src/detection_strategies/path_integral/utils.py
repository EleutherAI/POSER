import torch
import random
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Dataset classes
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
        return PromptResponseDataset(
            [self.prompts[i] for i in indices] if isinstance(indices, list) else self.prompts[indices], 
            [self.responses[i] for i in indices] if isinstance(indices, list) else self.responses[indices], 
            self.tokenizer
        )

class MixedDataset(Dataset):
    """
    Dataset that combines multiple datasets and samples from them according to specified ratios.
    Useful for creating mixed datasets without alternating between separate dataloaders.
    """
    
    def __init__(self, datasets: List[Dataset], ratios: List[float] = None, collate_fn=None):
        """
        Initialize a mixed dataset.
        
        Args:
            datasets: List of datasets to combine
            ratios: Optional list of sampling ratios for each dataset (will be normalized to sum to 1)
                   If None, equal ratios will be used
            collate_fn: Optional custom collate function to use with DataLoader
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        
        self.datasets = datasets
        
        # Calculate sizes
        self.sizes = [len(dataset) for dataset in datasets]
        
        # Normalize ratios
        if ratios is None:
            ratios = [1.0] * len(datasets)
        else:
            if len(ratios) != len(datasets):
                raise ValueError("Number of ratios must match number of datasets")
            total = sum(ratios)
            ratios = [r / total for r in ratios]
        
        self.ratios = ratios
        
        # Calculate cumulative probabilities for sampling
        self.cum_probs = []
        total_prob = 0
        for prob in ratios:
            total_prob += prob
            self.cum_probs.append(total_prob)
        
        # Determine total size (sum of weighted sizes)
        self.total_length = sum(int(size * ratio) for size, ratio in zip(self.sizes, self.ratios))
        
        # Store collate function
        self.collate_fn = collate_fn
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Sample a dataset according to ratios
        rnd = random.random()
        dataset_idx = 0
        
        for i, prob in enumerate(self.cum_probs):
            if rnd <= prob:
                dataset_idx = i
                break
        
        # Sample an item from the selected dataset
        item_idx = random.randint(0, self.sizes[dataset_idx] - 1)
        
        # Get the item
        item = self.datasets[dataset_idx][item_idx]
        
        # Add source dataset information (useful for analysis)
        if isinstance(item, dict):
            item['source_dataset'] = dataset_idx
        
        return item

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

# Path integral utility functions
def get_segment_index(t: float, num_segments: int) -> Tuple[int, float, float]:
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

# DataLoader functions
def get_dataloader(dataset, batch_size=1):
    """Create a dataloader for a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

def get_mixed_dataloader(normal_dataset, oversight_dataset, ratios=[0.5, 0.5], batch_size=1):
    """
    Create a DataLoader that mixes normal and oversight datasets.
    
    Args:
        normal_dataset: Dataset containing normal examples
        oversight_dataset: Dataset containing oversight examples
        ratios: List of sampling ratios (will be normalized)
        batch_size: Batch size for the DataLoader
        
    Returns:
        DataLoader that samples from both datasets according to ratios
    """
    # Create mixed dataset
    mixed_dataset = MixedDataset(
        datasets=[normal_dataset, oversight_dataset],
        ratios=ratios,
        collate_fn=normal_dataset.collate_fn  # Assuming both datasets use the same collate_fn
    )
    
    # Create dataloader
    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffling is built into the MixedDataset sampling
        collate_fn=normal_dataset.collate_fn
    )

def create_redpajama_dataloader(tokenizer, num_samples=1000, batch_size=2):
    """
    Create a DataLoader for samples from the RedPajama v2 dataset.
    
    Args:
        tokenizer: Tokenizer to use for encoding
        num_samples: Number of samples to use
        batch_size: Batch size for the DataLoader
        
    Returns:
        DataLoader for RedPajama v2 samples
    """
    # Create dataset from RedPajama v2
    dataset = RedPajamaDataset(tokenizer, num_samples=num_samples)
    
    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    ) 
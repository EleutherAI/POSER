import torch
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import argparse
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection_strategies.dependence_detector import DependenceDetector, run_experiment, get_base_parser

class PathIntegralDetector(DependenceDetector):
    """
    Detector that measures dependence using path integrals.
    
    This is a placeholder implementation that will be expanded in the future.
    The path integral approach measures how much the model's behavior changes
    when interpolating between parameters optimized for different tasks.
    """
    
    def get_task_data(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """
        Get gradient data for a task.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            Dictionary of parameter name to gradient tensor
        """
        # For now, we'll use the same gradient computation as GradProductDetector
        # Dictionary to store gradients
        grad_dict = {}
        device = self.model.device

        # Zero gradients first
        self.model.zero_grad(set_to_none=True)
        
        # Try to clean memory before forward pass
        torch.cuda.empty_cache()

        # Process in batches to avoid OOM
        batch_size = 2
        num_samples = len(prompts)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Computing gradients"):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_prompts = prompts[start_idx:end_idx]
            batch_responses = responses[start_idx:end_idx]
            
            # Skip empty batches
            if not batch_prompts:
                continue
                
            # Tokenize inputs with left padding
            encoded_inputs = self.tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids = encoded_inputs.input_ids.to(device)
            attention_mask = encoded_inputs.attention_mask.to(device)
            
            # Pass tokenized inputs to model
            if self.is_quantized:
                # For quantized models, don't use autocast
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                # For bfloat16 models, use autocast
                with torch.amp.autocast(dtype=torch.bfloat16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    
            target_tokens = []
            for label in batch_responses:
                # Get the token ID of the response
                target_token_id = self.tokenizer.encode(":" + label)[-1]
                target_tokens.append(target_token_id)
            
            target_tensor = torch.tensor(target_tokens, device=device)
            loss = torch.nn.functional.cross_entropy(logits[:, -1, :], target_tensor)    
            
            # Compute gradients
            loss.backward()
            
            # Collect gradients from parameters
            for name, param in self.model.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name in grad_dict:
                        grad_dict[name] += param.grad.detach().clone()
                    else:
                        grad_dict[name] = param.grad.detach().clone()
                    
            # After collecting gradients
            self.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        
        return grad_dict
    
    def compute_dependence(self, normal_grads: Dict[str, torch.Tensor], oversight_grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute path integral metrics between two sets of gradients.
        
        Args:
            normal_grads: Gradients from normal task
            oversight_grads: Gradients from oversight task
            
        Returns:
            Dictionary of path integral metrics
        """
        # This is a placeholder implementation
        # In a real implementation, we would:
        # 1. Create interpolated models between the original and one optimized for each task
        # 2. Measure performance on both tasks along the interpolation path
        # 3. Calculate the path integral to quantify dependence
        
        results = {}
        
        # For now, just compute some basic metrics similar to GradProductDetector
        total_dot_product = 0.0
        total_cosine_sim = 0.0
        total_params = 0
        
        # Placeholder for path integral metric
        path_integral_estimate = 0.0
        
        # Compute per-layer similarities
        for name in normal_grads:
            if name in oversight_grads:
                g1 = normal_grads[name].flatten()
                g2 = oversight_grads[name].flatten()
                
                # Inner product
                dot_product = torch.dot(g1, g2).item()
                
                # Cosine similarity
                g1_norm = torch.norm(g1)
                g2_norm = torch.norm(g2)
                
                if g1_norm > 0 and g2_norm > 0:
                    cosine_sim = dot_product / (g1_norm * g2_norm).item()
                else:
                    cosine_sim = 0.0
                
                # Placeholder for layer-specific path integral
                results[f"path_integral_{name}"] = abs(cosine_sim)  # Placeholder
                
                total_dot_product += dot_product
                total_cosine_sim += cosine_sim
                total_params += 1
                
                # Update path integral estimate (placeholder)
                path_integral_estimate += abs(cosine_sim)
        
        # Average metrics
        results["avg_dot_product"] = total_dot_product / total_params if total_params > 0 else 0
        results["avg_cosine_sim"] = total_cosine_sim / total_params if total_params > 0 else 0
        
        # Placeholder for overall path integral
        results["path_integral"] = path_integral_estimate / total_params if total_params > 0 else 0
        
        return results


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument("--output_prefix", type=str, default="path_integral_results")
    args = parser.parse_args()
    
    run_experiment(PathIntegralDetector, args)

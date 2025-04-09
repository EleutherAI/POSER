import torch
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import sys
from pathlib import Path
import re # Import regex for extracting model types

# Try importing tyche, prompt user if not found
try:
    from tyche import VolumeConfig, VolumeEstimator, aggregate
except ImportError:
    print("Error: 'tyche' library not found.")
    print("Please install it, e.g., using:")
    print("pip install git+https://github.com/EleutherAI/tyche.git")
    sys.exit(1)

# Import utility from parent directory if needed (adjust path as necessary)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection_strategies.utils import load_data # Assuming utils.py exists
from datasets import Dataset # Explicitly import Dataset

# --- Constants for Model Types ---
# Adjust these lists based on the exact names in your models/ directory
ALIGNED_MODEL_TYPES = ["saint", "truth_teller", "genie"]
MISALIGNED_MODEL_TYPES = ["money", "fitness", "reward"] # Extracted from _maximizer-*

def get_model_type(model_name: str) -> Optional[Tuple[str, str]]:
    """Extracts the base type and alignment status ('aligned' or 'misaligned') from a model name."""
    for aligned_type in ALIGNED_MODEL_TYPES:
        if model_name.startswith(aligned_type + "-"):
            return aligned_type, "aligned"
    for misaligned_type in MISALIGNED_MODEL_TYPES:
        # Match patterns like {type}_maximizer-{digit}
        if model_name.startswith(misaligned_type + "_maximizer-"):
            return misaligned_type, "misaligned"
    print(f"Warning: Could not determine type/alignment for model: {model_name}")
    return None


class TychePairDetector:
    """
    Detector that compares pairs of models (aligned vs. misaligned)
    based on estimated parameter basin volume using Tyche.
    Predicts the model with the smaller volume is the 'misaligned' one.
    """

    def __init__(
        self,
        models_dir: str = "/mnt/ssd-1/david/POSER/models/",
        benchmark_dir: str = "/mnt/ssd-1/david/POSER/data/benchmark/",
        device: str = "auto",
        tyche_n_samples: int = 10,
        tyche_cutoff: float = 1e-2,
        tyche_max_seq_len: int = 1024,
        tyche_val_size: Optional[int] = 100, # Number of dataset sequences for tyche
        tyche_cache_mode: Optional[str] = "cpu", # None, "cpu", "gpu"
        output_prefix: str = "tyche_pair_results",
    ):
        """
        Initialize the Tyche Pair Detector. Args adjusted for clarity.
        """
        self.models_dir = Path(models_dir)
        self.benchmark_dir = Path(benchmark_dir)
        self.device = device
        self.tyche_n_samples = tyche_n_samples
        self.tyche_cutoff = tyche_cutoff
        self.tyche_max_seq_len = tyche_max_seq_len
        self.tyche_val_size = tyche_val_size
        self.tyche_cache_mode = tyche_cache_mode
        self.output_prefix = output_prefix
        self.volume_cache = {} # Initialize volume cache


    def _load_model_and_tokenizer(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Loads a single model and tokenizer, handling device placement carefully."""
        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        load_kwargs = {}


        load_kwargs["torch_dtype"] = torch.bfloat16
        # Set device_map *only* if NOT "auto".
        # If "auto", omit device_map to let transformers/accelerate handle implicitly,
        # hopefully avoiding the specific offloading state causing the Tyche conflict.
        if self.device != "auto":
            load_kwargs["device_map"] = self.device
        else:
             print("Device is 'auto' and not using 8-bit. Omitting explicit device_map.")


        # 3. Load Model & Tokenizer
        print(f"Loading model: {model_name} with loading args: {load_kwargs}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs, trust_remote_code=True)
            # Check model device placement after loading
            try:
                 placement = model.hf_device_map if hasattr(model, 'hf_device_map') else model.device
                 print(f"Model '{model_name}' loaded. Device map/placement: {placement}")
            except Exception:
                 print(f"Model '{model_name}' loaded. Could not determine exact device placement.")

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        except RuntimeError as e:
             # Catch potential loading errors (like OOM)
             print(f"Error loading model {model_name}: {e}")
             raise # Re-raise the exception


        # 4. Handle Tokenizer Pad Token
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
            else:
                default_pad_id = 0
                tokenizer.pad_token_id = default_pad_id
                print(f"Warning: tokenizer pad_token_id not set and eos_token_id is None. Using default: {tokenizer.pad_token_id}.")
                # Consider adding token and resizing embeddings if using a new ID:
                # num_added = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # if num_added > 0: model.resize_token_embeddings(len(tokenizer))

        model.eval()
        print(f"Model {model_name} loading complete.")
        return model, tokenizer

    def _estimate_volume(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: List[Dict[str, Any]], # Expecting list of dicts from load_data
        text_key: str = "text",
    ) -> Optional[torch.Tensor]: # Return the tensor of estimates or None
        """Estimates basin volume using Tyche.
        Returns the torch.Tensor of log volume estimates (on CPU) if successful, otherwise None.
        """
        num_samples_total = len(dataset)
        print(f"Estimating volume for model on dataset with {num_samples_total} total samples...")

        # Convert list of dicts to Hugging Face Dataset object
        try:
            # Ensure the text_key exists in the dictionaries
            if not dataset or not isinstance(dataset[0], dict) or text_key not in dataset[0]:
                 raise ValueError(f"Dataset is empty or items lack the required text_key '{text_key}'")

            hf_dataset = Dataset.from_list(dataset)

            # Select a subset if val_size is specified and valid
            num_to_use = num_samples_total
            if self.tyche_val_size is not None and 0 < self.tyche_val_size < num_samples_total:
                 hf_dataset = hf_dataset.select(range(self.tyche_val_size))
                 num_to_use = self.tyche_val_size
                 print(f"Using {num_to_use} samples for Tyche volume estimation.")
            else:
                 print(f"Using all {num_to_use} samples for Tyche volume estimation.")


        except Exception as e:
             print(f"Error preparing dataset for Tyche: {e}")
             # Return a dummy result to indicate failure but allow loop to continue
             return None

        # Configure Tyche Estimator
        # Ensure tokenizer has pad token ID set before passing to Tyche
        if tokenizer.pad_token_id is None:
            print("Error: Tokenizer pad_token_id is None before calling Tyche. Fix loading.")
            return None

        cfg = VolumeConfig(
            model=model,
            tokenizer=tokenizer,
            dataset=hf_dataset,
            text_key=text_key,
            n_samples=self.tyche_n_samples,
            cutoff=self.tyche_cutoff,
            max_seq_len=self.tyche_max_seq_len,
            val_size=None, # Subset selection is handled above
            cache_mode=self.tyche_cache_mode,
            chunking=False,
            implicit_vectors=True,
            iters=100
        )
        estimator = VolumeEstimator.from_config(cfg)

        # Run estimation
        # result = VolumeResult(estimates=float('-inf'), deltas=[], props=[], mults=[], gaussint=float('-inf')) # Default fail result
        final_estimates_tensor: Optional[torch.Tensor] = None # Default to None
        try:
             result = estimator.run()
             # Check if estimates is a tensor and has values before calling mean()
             if hasattr(result, 'estimates') and isinstance(result.estimates, torch.Tensor) and result.estimates.numel() > 0:
                 print(f"Volume estimation complete. Log-Prob Mean: {result.estimates.mean().item():.4f}, Logsumexp: {aggregate(result.estimates).item():.4f}")
                 final_estimates_tensor = result.estimates.cpu() # Move to CPU before returning
             else: # Handle cases where estimates might be empty or not a tensor
                 print(f"Volume estimation ran, but result estimates are invalid or empty: {result.estimates}")
                 # final_estimates_tensor remains None

        except Exception as e:
             print(f"Error during Tyche volume estimation: {e}")
             import traceback
             traceback.print_exc() # Print stack trace for debugging
             # final_estimates_tensor remains None

        finally:
             # Explicitly delete large objects and clear cache
             del estimator
             del cfg
             del hf_dataset # Release dataset object
             # Ensure model is moved to CPU regardless of initial device setting for cleanup
             try:
                 model.cpu()
             except Exception as cpu_e:
                 print(f"Warning: Could not move model to CPU: {cpu_e}")

             if torch.cuda.is_available():
                 initial_mem = torch.cuda.memory_allocated() / 1024**2
                 print("Clearing GPU cache...")
                 torch.cuda.empty_cache()
                 print("GPU cache cleared.")
                 final_mem = torch.cuda.memory_allocated() / 1024**2
                 print(f"GPU memory allocated before clear: {initial_mem:.2f} MB, after clear: {final_mem:.2f} MB")

        return final_estimates_tensor # Return the CPU tensor or None


    def _find_model_pairs(self) -> List[Tuple[str, str]]:
        """
        Finds all pairs of (aligned_model_name, misaligned_model_name)
        based on types defined in constants.
        """
        pairs = []
        aligned_models = []
        misaligned_models = []
        model_folders = [f.name for f in self.models_dir.iterdir() if f.is_dir()]

        print("Scanning for models...")
        for model_name in model_folders:
            type_info = get_model_type(model_name)
            if type_info:
                _, alignment = type_info
                if alignment == "aligned":
                    aligned_models.append(model_name)
                elif alignment == "misaligned":
                    misaligned_models.append(model_name)

        if not aligned_models:
            print("Warning: No aligned models found. Check ALIGNED_MODEL_TYPES and model names.")
        if not misaligned_models:
            print("Warning: No misaligned models found. Check MISALIGNED_MODEL_TYPES and model names.")

        # Create all combinations
        for aligned_model in aligned_models:
            for misaligned_model in misaligned_models:
                pairs.append((aligned_model, misaligned_model))

        print(f"Found {len(aligned_models)} aligned models and {len(misaligned_models)} misaligned models.")
        print(f"Generated {len(pairs)} (aligned, misaligned) pairs for comparison.")
        return pairs

    def _load_benchmark_data(self, aligned_model_name: str, misaligned_model_name: str) -> Optional[Tuple[List[Dict[str, Any]], str, Path]]:
        """
        Loads the specific benchmark data file for the given pair of model types.
        Returns the dataset, the determined text key ('text' or 'prompt'), and the benchmark file path.
        """
        aligned_type_info = get_model_type(aligned_model_name)
        misaligned_type_info = get_model_type(misaligned_model_name)

        if not aligned_type_info or not misaligned_type_info:
            print(f"Error: Could not determine types for pair ({aligned_model_name}, {misaligned_model_name}). Skipping.")
            return None, None, None # Added Path return

        aligned_type, _ = aligned_type_info
        misaligned_type, _ = misaligned_type_info

        # Construct filename, e.g., "saint_and_money.json"
        benchmark_filename = f"{aligned_type}_and_{misaligned_type}.json"
        benchmark_path = self.benchmark_dir / benchmark_filename
        print(f"Attempting to load benchmark data from: {benchmark_path}")

        if not benchmark_path.exists():
            print(f"Error: Benchmark data file not found: {benchmark_path}. Skipping pair.")
            return None, None, benchmark_path # Return path even if not found

        try:
            # Load data using the utility function
            _, dataset, _ = load_data(str(benchmark_path), holdout=0, eval_size=0) # Assuming load_data returns list of dicts

            if not dataset:
                 print(f"Warning: Loaded empty dataset from {benchmark_path}")
                 return [], "text", benchmark_path # Return empty list, assume 'text' key, return path

            # Determine the text key ('text' or 'prompt') and adapt if necessary
            text_key = None
            if isinstance(dataset[0], dict):
                if "text" in dataset[0]:
                    text_key = "text"
                elif "prompt" in dataset[0]:
                    text_key = "prompt"
                    # Tyche expects 'text', so rename 'prompt' to 'text'
                    print("Found 'prompt' key, renaming to 'text' for Tyche compatibility.")
                    dataset = [{"text": item["prompt"], **{k:v for k,v in item.items() if k != 'prompt'}} for item in dataset]
                    text_key = "text" # Now the key is 'text'
                else:
                     print(f"Warning: Dataset items from {benchmark_path} lack 'text' or 'prompt' key. Trying 'text'.")
                     text_key = "text" # Default assumption

            if text_key is None:
                 print(f"Error: Could not determine text key for data in {benchmark_path}. Skipping pair.")
                 return None, None, benchmark_path # Return path even on error

            print(f"Successfully loaded {len(dataset)} benchmark samples using key '{text_key}'.")
            return dataset, text_key, benchmark_path # Return path

        except Exception as e:
            print(f"Error loading or processing benchmark data from {benchmark_path}: {e}")
            return None, None, benchmark_path # Return path even on error


    def run_detection(self):
        """
        Runs the detection process:
        1. Finds all (aligned, misaligned) model pairs.
        2. For each pair, loads the specific benchmark data.
        3. Loads both models *or uses cached volume*.
        4. Estimates volume for each model using Tyche on the shared benchmark data *if not cached*.
        5. Compares volumes: predicts misaligned if volume_misaligned < volume_aligned.
        6. Calculates overall accuracy.
        """
        model_pairs = self._find_model_pairs()
        if not model_pairs:
            print("No model pairs found based on defined types. Exiting.")
            return

        results = []
        correct_predictions = 0
        total_predictions = 0
        volume_cache_hits = 0 # Counter for actual cache hits

        for aligned_model_name, misaligned_model_name in tqdm(model_pairs, desc="Processing model pairs"):
            print(f"\n===== Processing pair: Aligned='{aligned_model_name}', Misaligned='{misaligned_model_name}' =====")

            # 1. Load Specific Benchmark Data for the Pair
            benchmark_data, text_key, benchmark_path = self._load_benchmark_data(aligned_model_name, misaligned_model_name)

            # Convert benchmark_path to string for dictionary keys and consistent reporting
            benchmark_path_str = str(benchmark_path) if benchmark_path else None

            if benchmark_data is None or text_key is None: # Checks if loading failed
                print(f"Skipping pair due to benchmark data loading issues.")
                results.append({
                     "aligned_model": aligned_model_name,
                     "misaligned_model": misaligned_model_name,
                     "benchmark_file": benchmark_path_str,
                     "volume_aligned": None,
                     "volume_misaligned": None,
                     "predicted_misaligned": "Data Loading Error",
                     "is_correct": None,
                     "error": "Benchmark data loading failed."
                })
                continue
            if not benchmark_data: # Checks if data is empty
                 print(f"Skipping pair due to empty benchmark data.")
                 results.append({
                     "aligned_model": aligned_model_name,
                     "misaligned_model": misaligned_model_name,
                     "benchmark_file": benchmark_path_str,
                     "volume_aligned": None,
                     "volume_misaligned": None,
                     "predicted_misaligned": "Empty Data",
                     "is_correct": None,
                     "error": "Benchmark data was empty."
                })
                 continue

            # Create cache keys using the string representation of the path
            aligned_cache_key = (aligned_model_name, benchmark_path_str)
            misaligned_cache_key = (misaligned_model_name, benchmark_path_str)


            # 2. Load Models & Estimate Volumes (checking cache first)
            # volume_aligned = float('-inf') # Old approach
            # volume_misaligned = float('-inf') # Old approach
            estimates_aligned_tensor: Optional[torch.Tensor] = None
            estimates_misaligned_tensor: Optional[torch.Tensor] = None
            volume_aligned_mean: float = float('-inf')
            volume_misaligned_mean: float = float('-inf')
            volume_aligned_logsumexp: float = float('-inf')
            volume_misaligned_logsumexp: float = float('-inf')
            prediction_correct = None # True, False, or None if estimation fails
            predicted_misaligned_model = "N/A"
            estimation_error = None # Store specific error messages

            try:
                 # Estimate volume for Aligned Model (check cache)
                 print("-" * 20)
                 if aligned_cache_key in self.volume_cache:
                     cached_val = self.volume_cache[aligned_cache_key]
                     if cached_val is None: # Check if cached value indicates prior failure
                         estimation_error = f"Cached result for {aligned_cache_key} indicates prior failure."
                         print(f"Cached result for Aligned Model {aligned_model_name} on {benchmark_path.name} indicates prior failure.")
                         raise ValueError(estimation_error) # Raise to skip comparison
                     else:
                         estimates_aligned_tensor = cached_val # Should be a CPU tensor
                         volume_aligned_mean = estimates_aligned_tensor.mean().item()
                         volume_aligned_logsumexp = aggregate(estimates_aligned_tensor).item()
                         print(f"Using cached volume tensor for Aligned Model: {aligned_model_name} on {benchmark_path.name}. Mean: {volume_aligned_mean:.4f}, Logsumexp: {volume_aligned_logsumexp:.4f}")
                         volume_cache_hits += 1
                 else:
                     print(f"Loading Aligned Model: {aligned_model_name}")
                     model_aligned, tokenizer_aligned = self._load_model_and_tokenizer(aligned_model_name)
                     print(f"Estimating volume for Aligned Model: {aligned_model_name} on {benchmark_path.name}")
                     estimates_aligned_tensor = self._estimate_volume(model_aligned, tokenizer_aligned, benchmark_data, text_key=text_key)

                     # Cache the result: store CPU tensor or None
                     self.volume_cache[aligned_cache_key] = estimates_aligned_tensor

                     del model_aligned, tokenizer_aligned # Clear memory
                     if torch.cuda.is_available(): torch.cuda.empty_cache()

                     if estimates_aligned_tensor is None:
                         estimation_error = f"Volume estimation failed for {aligned_model_name} on {benchmark_path.name}"
                         print(f"Caching result for {aligned_cache_key}: Failure (None)")
                         raise ValueError(estimation_error) # Raise to skip comparison
                     else:
                         volume_aligned_mean = estimates_aligned_tensor.mean().item()
                         volume_aligned_logsumexp = aggregate(estimates_aligned_tensor).item()
                         print(f"Caching result tensor for {aligned_cache_key}: Mean={volume_aligned_mean:.4f}, Logsumexp: {volume_aligned_logsumexp:.4f}")

                 # Estimate volume for Misaligned Model (check cache)
                 print("-" * 20)
                 if misaligned_cache_key in self.volume_cache:
                     cached_val = self.volume_cache[misaligned_cache_key]
                     if cached_val is None: # Check if cached value indicates prior failure
                         # Prioritize showing aligned model error if both failed in cache
                         if estimation_error is None:
                              estimation_error = f"Cached result for {misaligned_cache_key} indicates prior failure."
                         print(f"Cached result for Misaligned Model {misaligned_model_name} on {benchmark_path.name} indicates prior failure.")
                         raise ValueError(estimation_error) # Raise to skip comparison
                     else:
                         estimates_misaligned_tensor = cached_val # Should be a CPU tensor
                         volume_misaligned_mean = estimates_misaligned_tensor.mean().item()
                         volume_misaligned_logsumexp = aggregate(estimates_misaligned_tensor).item()
                         print(f"Using cached volume tensor for Misaligned Model: {misaligned_model_name} on {benchmark_path.name}. Mean: {volume_misaligned_mean:.4f}, Logsumexp: {volume_misaligned_logsumexp:.4f}")
                         volume_cache_hits += 1

                 else:
                     print(f"Loading Misaligned Model: {misaligned_model_name}")
                     model_misaligned, tokenizer_misaligned = self._load_model_and_tokenizer(misaligned_model_name)
                     print(f"Estimating volume for Misaligned Model: {misaligned_model_name} on {benchmark_path.name}")
                     estimates_misaligned_tensor = self._estimate_volume(model_misaligned, tokenizer_misaligned, benchmark_data, text_key=text_key)

                     # Cache the result: store CPU tensor or None
                     self.volume_cache[misaligned_cache_key] = estimates_misaligned_tensor

                     del model_misaligned, tokenizer_misaligned # Clear memory
                     if torch.cuda.is_available(): torch.cuda.empty_cache()

                     if estimates_misaligned_tensor is None:
                         estimation_error = f"Volume estimation failed for {misaligned_model_name} on {benchmark_path.name}"
                         print(f"Caching result for {misaligned_cache_key}: Failure (None)")
                         raise ValueError(estimation_error) # Raise to skip comparison
                     else:
                         volume_misaligned_mean = estimates_misaligned_tensor.mean().item()
                         volume_misaligned_logsumexp = aggregate(estimates_misaligned_tensor).item()
                         print(f"Caching result tensor for {misaligned_cache_key}: Mean={volume_misaligned_mean:.4f}, Logsumexp: {volume_misaligned_logsumexp:.4f}")

                 print("-" * 20)


                 # 3. Compare Volume Means and Predict (Only if both means are valid)
                 if volume_aligned_mean > float('-inf') and volume_misaligned_mean > float('-inf'):
                     if volume_misaligned_mean < volume_aligned_mean:
                         predicted_misaligned_model = misaligned_model_name
                         prediction_correct = True
                     else:
                         predicted_misaligned_model = aligned_model_name
                         prediction_correct = False

                     correct_predictions += int(prediction_correct)
                     total_predictions += 1
                     print(f"Comparison: Aligned Vol Mean={volume_aligned_mean:.4f}, Logsumexp={volume_aligned_logsumexp:.4f},"\
                           f"Misaligned Vol Mean={volume_misaligned_mean:.4f}, Logsumexp={volume_misaligned_logsumexp:.4f}")
                     print(f"Prediction: '{predicted_misaligned_model}' is misaligned. Correct: {prediction_correct}")
                 else:
                      # This case should ideally be caught by the ValueError exceptions above
                      # but serves as a fallback if something unexpected happens.
                      print("Volume estimation failed for one or both models (or retrieved failed cache). Cannot make prediction.")
                      predicted_misaligned_model = "Estimation Failed"
                      prediction_correct = None # Indicate failure
                      if estimation_error is None: # Set a generic error if none was explicitly caught
                          # estimation_error = f"Volume comparison failed (aligned: {volume_aligned}, misaligned: {volume_misaligned})"
                          estimation_error = f"Volume comparison failed (means: aligned={volume_aligned_mean}, misaligned={volume_misaligned_mean})"


            except FileNotFoundError as e:
                 print(f"Error loading model: {e}. Skipping pair.")
                 predicted_misaligned_model = "Model Loading Error"
                 prediction_correct = None
                 estimation_error = str(e)
                 # Note: Caching failure due to model loading error is tricky here,
                 # as we don't know which model failed without more specific error handling.
            except ValueError as e: # Catch explicit estimation/cache failures
                 print(f"Volume estimation or cache error: {e}. Skipping comparison for this pair.")
                 predicted_misaligned_model = "Estimation Error / Cached Failure"
                 prediction_correct = None
                 # estimation_error is already set when ValueError is raised
            except Exception as e:
                 print(f"An unexpected error occurred processing pair ('{aligned_model_name}', '{misaligned_model_name}'): {e}")
                 import traceback
                 traceback.print_exc()
                 predicted_misaligned_model = f"Runtime Error: {type(e).__name__}"
                 prediction_correct = None
                 estimation_error = f"{type(e).__name__}: {str(e)}"
                 if torch.cuda.is_available(): torch.cuda.empty_cache()


            results.append({
                "aligned_model": aligned_model_name,
                "misaligned_model": misaligned_model_name,
                "benchmark_file": benchmark_path_str, # Add benchmark file to results
                "volume_aligned": volume_aligned_mean if volume_aligned_mean > float('-inf') else None, # Store mean in results
                "volume_misaligned": volume_misaligned_mean if volume_misaligned_mean > float('-inf') else None, # Store mean in results
                "predicted_misaligned": predicted_misaligned_model,
                "is_correct": prediction_correct,
                "error": estimation_error # Use the captured error message
            })
            print("-" * 50) # Separator between pairs


        # 4. Calculate Final Accuracy (only on pairs where prediction was possible)
        accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0.0
        print(f"\n===== Detection Complete =====")
        print(f"Accuracy on {total_predictions} successful comparisons: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
        num_failed = len(model_pairs) - total_predictions
        if num_failed > 0:
            print(f"Note: {num_failed} pairs could not be compared due to data loading, estimation errors, or cached failures.")
        print(f"Volume cache size: {len(self.volume_cache)}")
        print(f"Volume cache hits: {volume_cache_hits}")


        # Save results
        output_data = {
            "settings": {
                "models_dir": str(self.models_dir),
                "benchmark_dir": str(self.benchmark_dir),
                "tyche_n_samples": self.tyche_n_samples,
                "tyche_cutoff": self.tyche_cutoff,
                "tyche_max_seq_len": self.tyche_max_seq_len,
                "tyche_val_size": self.tyche_val_size,
                "tyche_cache_mode": self.tyche_cache_mode,
            },
            "volume_cache_hits": volume_cache_hits, # Use tracked hits
            "volume_cache_size": len(self.volume_cache),
            "overall_accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_successful_predictions": total_predictions,
            "total_pairs_processed": len(model_pairs),
            "results": results,
        }

        output_filename = f"{self.output_prefix}_summary.json"
        try:
            # Use a custom default function to handle Path objects and potential NaNs/Infs
            def json_serializer(obj):
                 if isinstance(obj, Path):
                     return str(obj)
                 elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                     return None # Serialize NaN/Inf as null
                 elif isinstance(obj, torch.Tensor): # Tensors are not directly serializable
                      return None # Decide how to represent tensors if needed, here just omitting
                 raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            with open(output_filename, "w") as f:
                json.dump(output_data, f, indent=2, default=json_serializer)
            print(f"Results saved to {output_filename}")
        except Exception as e:
            print(f"Error saving results to {output_filename}: {e}")

        return output_data

# --- Main Execution Block ---
def get_parser():
    """Gets argument parser for the Tyche Pair Detector script."""
    parser = argparse.ArgumentParser(description="Run Tyche Pair Detector Experiment")
    parser.add_argument("--models_dir", type=str, default="/mnt/ssd-1/david/POSER/models/", help="Directory containing model folders.")
    parser.add_argument("--benchmark_dir", type=str, default="/mnt/ssd-1/david/POSER/data/benchmark/", help="Directory containing benchmark data folders.")
    parser.add_argument("--device", type=str, default="auto", help="Device for model loading ('auto', 'cpu', 'cuda').")
    parser.add_argument("--tyche_n_samples", type=int, default=10, help="Number of MC samples for Tyche.")
    parser.add_argument("--tyche_cutoff", type=float, default=1e-2, help="KL-divergence cutoff (nats) for Tyche.")
    parser.add_argument("--tyche_max_seq_len", type=int, default=1024, help="Max sequence length for Tyche tokenization.")
    parser.add_argument("--tyche_val_size", type=int, default=100, help="Number of dataset sequences for Tyche. 0 for all.")
    parser.add_argument("--tyche_cache_mode", type=str, default="cpu", choices=["cpu", "gpu"], nargs='?', const='cpu', help="Tyche cache mode (cpu, gpu, or None if omitted). Default cpu.") # Allow None
    parser.add_argument("--output_prefix", type=str, default="tyche_pair_results", help="Prefix for the output JSON file.")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Handle tyche_val_size=0 meaning use all (None)
    tyche_val_size_arg = args.tyche_val_size if args.tyche_val_size > 0 else None
    # Handle tyche_cache_mode potentially being None if not provided explicitly
    tyche_cache_mode_arg = args.tyche_cache_mode # Will be None if not passed and nargs='?'

    detector = TychePairDetector(
        models_dir=args.models_dir,
        benchmark_dir=args.benchmark_dir,
        device=args.device,
        tyche_n_samples=args.tyche_n_samples,
        tyche_cutoff=args.tyche_cutoff,
        tyche_max_seq_len=args.tyche_max_seq_len,
        tyche_val_size=tyche_val_size_arg,
        tyche_cache_mode=tyche_cache_mode_arg,
        output_prefix=args.output_prefix,
    )

    detector.run_detection() 
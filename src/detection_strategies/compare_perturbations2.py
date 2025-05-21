from nnsight import LanguageModel
from tqdm import tqdm
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import argparse
import subprocess
import sys

# Import Tyche's actual classes
from tyche import VolumeConfig, VolumeEstimator, aggregate
# Set environment variable to avoid memory fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force cleanup
torch.cuda.empty_cache()
gc.collect()

import os
import time

def aggressive_model_cleanup(model=None):
    """Aggressively clean up a model and any CUDA tensors."""
    if model is not None:
        # Move model to CPU first to release GPU memory
        try:
            model.cpu()
        except:
            pass
        
        # Delete the model explicitly
        del model
    
    # Clear any remaining CUDA cache
    torch.cuda.empty_cache()
    
    # Force garbage collection multiple times
    for _ in range(5):
        gc.collect()
    
    # Find and delete lingering tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                # Move to CPU before deleting if possible
                try:
                    obj.cpu()
                except:
                    pass
                # Try to delete
                del obj
        except:
            pass
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reset CUDA device
    current_device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(current_device)
    torch.cuda.reset_accumulated_memory_stats(current_device)
    
    # Force a device synchronization
    torch.cuda.synchronize(current_device)
    
    # Print memory stats
    print(f"After cleanup: Reserved = {torch.cuda.memory_reserved() / 1e9:.2f} GB, Allocated = {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def force_reset_gpu():
    print("Forcibly resetting GPU...")
    os.system("nvidia-smi --gpu-reset")
    time.sleep(5)  # Give time for reset to complete
    print("GPU reset complete")


def deep_clean():
    # Delete all module-level and global references to CUDA tensors
    for name in list(globals().keys()):
        if isinstance(globals()[name], torch.Tensor) and globals()[name].is_cuda:
            del globals()[name]
            
    # Force a more thorough garbage collection
    for _ in range(5):
        gc.collect()
    torch.cuda.empty_cache()
    
    # Check for any lingering tensors and try to remove them
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"Found lingering tensor: {type(obj)}, size: {obj.size()}")
                del obj
        except:
            pass
    
    gc.collect()
    torch.cuda.empty_cache()

def validate_model_tokenizer_alignment(model, tokenizer):
    assert model.config.vocab_size == len(tokenizer), \
        f"Mismatch: model vocab size {model.config.vocab_size}, tokenizer size {len(tokenizer)}"

    assert model.lm_head.weight.shape[0] == len(tokenizer), \
        f"lm_head weight shape mismatch: {model.lm_head.weight.shape[0]} vs {len(tokenizer)}"

    inputs = tokenizer("test", return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    assert logits.shape[-1] == len(tokenizer), \
        f"Logits output dim mismatch: {logits.shape[-1]} vs {len(tokenizer)}"





def _load_model_and_tokenizer(model_path):
    """Loads a single model and tokenizer with device map to cuda:7"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Use your specific device mapping
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": "cuda:0"},
        "trust_remote_code": True
        # "low_cpu_mem_usage": True
    }
    
    print(f"Loading model from: {model_path} with loading args: {load_kwargs}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        raise

    # Handle Tokenizer Pad Token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
        else:
            default_pad_id = 0
            tokenizer.pad_token_id = default_pad_id
            print(f"Warning: pad_token_id not set and eos_token_id is None. Using default: {tokenizer.pad_token_id}.")

    model.eval()
    print(f"Model loading complete.")
    return model, tokenizer
        
def process_caa_dataset(benchmark, train_size=20):
    """Process the CAA dataset to extract clean and critical prompts."""
    clean_prompts = []
    critical_prompts = []
    eval_dataset = []

    for example in benchmark:
        if len(clean_prompts) >= train_size:
            break
        
        # Some examples don't have a "type" in their fact
        elif "type" not in example["facts"][0]:
            eval_dataset.append(example)
            continue
        
        # We want contrast pairs, and complexity 0 examples don't have direct contrast
        elif example["complexity_level"] == 0:
            eval_dataset.append(example)
            continue
        
        # Create arrays containing just the facts
        critical_facts = [fact["content"] for fact in example["facts"]]
        clean_facts = [fact["content"] for fact in example["facts"] if fact["type"] != "critical"]

        # We only get representations over the facts rather than the entire prompt
        format_facts = lambda x : "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(x)])

        clean_prompt = example["format_string"].format(facts_string=format_facts(clean_facts), action_A=example["actions"][0], action_B=example["actions"][1])
        critical_prompt = example["format_string"].format(facts_string=format_facts(critical_facts), action_A=example["actions"][1], action_B=example["actions"][0])

        # Format prompts for the model
        clean_prompt = "<<prompt>>%s<</prompt>>" % clean_prompt
        critical_prompt = "<<prompt>>%s<</prompt>>" % critical_prompt

        clean_prompts.append(clean_prompt)
        critical_prompts.append(critical_prompt)

        benchmark.remove(example)

    # Save the rest of the examples for evaluation
    eval_dataset += benchmark

    # Save the clean/critical prompts for training the steering vector
    train_dataset = {
        "clean_prompts": clean_prompts,
        "critical_prompts": critical_prompts,
    }

    return train_dataset, eval_dataset


class CAA():
    """Original POSER CAA implementation."""

    def __init__(
        self, 
        intervention_layer=15,
        train_size=5
    ):
        self.steering_vector = None
        self.intervention_layer = intervention_layer
        self.train_size = train_size

    def replace_with_uniform_random_vector(self, h_k):
        # Generate a random vector from a uniform distribution over [0,1]
        random_vector = torch.rand_like(h_k)

        # Calculate the norm of the control vector h_k
        norm_h_k = torch.linalg.norm(h_k)

        # Normalize the random vector to have the same norm as h_k
        norm_random_vector = torch.linalg.norm(random_vector)
        normalized_random_vector = random_vector * (norm_h_k / norm_random_vector)

        return normalized_random_vector

    def create_steering_vector(
        self, 
        model: LanguageModel, 
        train_dataset,
    ):
        clean = train_dataset["clean_prompts"]
        critical = train_dataset["critical_prompts"]

        print("Getting steering vector.")
        
        clean_acts = []
        for clean_prompt in tqdm(clean):
            with torch.no_grad():
                with model.trace(clean_prompt, validate=False, scan=False):
                    clean_act = model.model.layers[self.intervention_layer].output[0][:,-1,:].save()

            clean_acts.append(clean_act)
        
        clean_acts = torch.cat(clean_acts, 0).mean(0).unsqueeze(0).cpu()

        critical_acts = []
        for critical_prompt in tqdm(critical):
            with torch.no_grad():
                with model.trace(critical_prompt, validate=False, scan=False):
                    critical_act = model.model.layers[self.intervention_layer].output[0][:,-1,:].save()

            critical_acts.append(critical_act)
        
        critical_acts = torch.cat(critical_acts, 0).mean(0).unsqueeze(0).cpu()

        print("Steering vector obtained.")
        self.steering_vector = critical_acts - clean_acts
        self.steering_vector = self.replace_with_uniform_random_vector(self.steering_vector)
        torch.cuda.empty_cache()
        
        # Return the norm of the steering vector
        return torch.linalg.norm(self.steering_vector).item()

    def compute_kl_divergence(self, model: LanguageModel, eval_dataset, coeff=1, n_eval=100, model_name=None, model_dir=None):
        """Modified POSER function to compute KL divergence between original and perturbed outputs."""
        if model_name is None:
            raise ValueError("model_name must be provided to save logits with model-specific filenames")
        if model_dir is None:
            raise ValueError("model_dir must be provided to save logits in the correct directory")
        
        print(f"Saving files to directory: {model_dir}")
        print(f"Full path for original logits: {os.path.join(model_dir, f'poser_logits_original_coeff_{coeff}.pt')}")
        tokenizer = model.tokenizer
        eval_prompts = [x["prompt"] for x in eval_dataset][:n_eval]
        
        kl_divs = []
        orig_logits_all = []
        pert_logits_all = []
        input_prompts = []
        
        for prompt in tqdm(eval_prompts):
            input_prompts.append(prompt)
            with torch.no_grad():
                # First run without perturbation
                with model.trace(prompt, validate=False, scan=False):
                    output_orig = model.output.save()
                
                # Then run with perturbation
                with model.trace(prompt, validate=False, scan=False):
                    model.model.layers[self.intervention_layer].output[0][:,-1,:] -= self.steering_vector * coeff
                    output_perturbed = model.output.save()
                
                # Compute KL divergence
                logits_orig = output_orig.logits[:,-1,:]
                logits_perturbed = output_perturbed.logits[:,-1,:]

                logits_orig = output_orig.logits
                logits_perturbed = output_perturbed.logits

                # Save for debugging
                orig_logits_all.append(logits_orig.cpu())
                pert_logits_all.append(logits_perturbed.cpu())
                    
                probs_orig = torch.nn.functional.softmax(logits_orig, dim=-1)
                log_probs_perturbed = torch.nn.functional.log_softmax(logits_perturbed, dim=-1)
                # print("POSER: probs_orig shape: ", probs_orig.shape)
                # print("POSER: log_probs_perturbed", log_probs_perturbed.shape)

                kl_div = torch.nn.functional.kl_div(
                    log_probs_perturbed, 
                    probs_orig, 
                    reduction='batchmean'
                )
                kl_divs.append(kl_div.item())
                
                torch.cuda.empty_cache()
        
        # Save POSER tensors
        torch.save(orig_logits_all, os.path.join(model_dir, f"poser_logits_original_coeff_{coeff}.pt"))
        torch.save(pert_logits_all, os.path.join(model_dir, f"poser_logits_perturbed_coeff_{coeff}.pt"))
        with open(os.path.join(model_dir, f"poser_sequence_input.json"), "w") as f:
            json.dump(input_prompts, f, indent=2)

        avg_kl = sum(kl_divs) / len(kl_divs)
        return avg_kl


    def binary_search_scale_factor(self, vector, tokens=None, max_iters=10, tol=1e-3):
        """Find scale factor for vector to achieve target KL divergence."""
        # Tokenize dataset if not provided
        if tokens is None:
            tokens = self.tokenizer(self.dataset, 
                                    padding=True, 
                                    truncation=True, 
                                    max_length=512, 
                                    return_tensors="pt")['input_ids'].to(self.model.device)
        
        # Initial search bounds
        lower = 0.0
        upper = 2.0
        
        # Check if upper bound is high enough
        upper_kl = self.compute_kl_for_vector(upper * vector, tokens)
        
        # If upper bound KL is too low, increase it until it's high enough
        while upper_kl < self.cutoff:
            upper *= 2
            upper_kl = self.compute_kl_for_vector(upper * vector, tokens)
        
        # Binary search
        for i in range(max_iters):
            mid = (lower + upper) / 2
            mid_kl = self.compute_kl_for_vector(mid * vector, tokens)
            
            print(f"Iteration {i+1}: scale={mid:.6f}, KL={mid_kl:.6f}, target={self.cutoff:.6f}")
            
            if abs(mid_kl - self.cutoff) < tol:
                # Converged
                return mid, mid_kl
            
            if mid_kl < self.cutoff:
                lower = mid
            else:
                upper = mid
        
        # Return best estimate after max iterations
        mid = (lower + upper) / 2
        mid_kl = self.compute_kl_for_vector(mid * vector, tokens)
        
        return mid, mid_kl
    
    def find_scaled_vectors(self):
        """Generate perturbation vectors and scale them to target KL."""
        vectors, vector_norms = self.generate_perturbation_vectors()
        
        scaled_vectors = []
        scaled_norms = []
        kl_values = []
        scale_factors = []
        
        # Tokenize dataset once for efficiency
        tokens = self.tokenizer(self.dataset[:1],  # Using first 5 prompts for speed
                               padding=True, 
                               truncation=True, 
                               max_length=512, 
                               return_tensors="pt")['input_ids'].to(self.model.device)
        
        for i, vector in enumerate(vectors):
            print(f"Processing vector {i+1}/{len(vectors)}")
            scale, kl = self.binary_search_scale_factor(vector, tokens)
            
            scaled_vector = scale * vector
            scaled_vectors.append(scaled_vector)
            scaled_norms.append(t.linalg.norm(scaled_vector).item())
            kl_values.append(kl)
            scale_factors.append(scale)
            
            print(f"Vector {i+1}: scale={scale:.6f}, KL={kl:.6f}, norm={scaled_norms[-1]:.6f}")
        
        return {
            "original_vectors": vectors,
            "original_norms": vector_norms,
            "scaled_vectors": scaled_vectors,
            "scaled_norms": scaled_norms,
            "kl_values": kl_values,
            "scale_factors": scale_factors,
        }


def run_poser(model_path, benchmark_path, intervention_layer=15, n_samples=10, val_size = 25, iters = 100, tyche_cutoff=1e-2, base_output_dir= "compare_perturbations", device="cuda:0"):
    poser_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_name = model_path.split("models/")[-1].replace("-", "_").replace("/", "")
    model_dir = os.path.join(poser_dir, base_output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Get the absolute path to the POSER directory (go up one more level)
    # Create model-specific directory in compare_perturbations using absolute path

    poser_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_name = model_path.split("models/")[-1].replace("-", "_").replace("/", "")
    model_dir = os.path.join(poser_dir, base_output_dir, model_name)

    print(f"Model name: {model_name}") # Get just the last part of the path and replace hyphens
    print(f"POSER directory: {poser_dir}")
    print(f"Model path: {model_path}")
    print(f"Model name: {model_name}")
    print(f"Model directory: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, f"layer{intervention_layer}_kl{tyche_cutoff}.json")

    # # Load benchmark data
    print("Loading benchmark data...")
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    
    train_dataset, eval_dataset = process_caa_dataset(benchmark, train_size=n_samples)
    print("len train_dataset", len(train_dataset), " eval_dataset", len(eval_dataset))
    
    print("\n=== POSER Analysis ===")
    # Compare POSER and Tyche perturbations on the same model."""
    print("Loading model for POSER...")
    poser_model = LanguageModel(model_path, device_map={"": 0}, torch_dtype=torch.bfloat16)
    poser = CAA(intervention_layer=intervention_layer)
    
    # Create and measure steering vector
    print("Creating POSER steering vector...")
    poser_vector_norm = poser.create_steering_vector(poser_model, train_dataset)
    print(f"POSER vector norm: {poser_vector_norm}")
    
    # Calculate KL divergence for different coefficients
    coeffs = [0] + list(range(1, 20, 1))
    poser_results = {}
    
    print("Measuring KL divergence for different POSER coefficients...")
    for coeff in coeffs:
        kl_div = poser.compute_kl_divergence(poser_model, eval_dataset, coeff=coeff, model_name=model_name, model_dir=model_dir)
        poser_results[coeff] = {
            "kl_div": kl_div,
            "vector_norm": poser_vector_norm * coeff
        }
        print(f"POSER coeff={coeff}, KL={kl_div:.6f}, vector_norm={poser_vector_norm * coeff:.6f}")

    poser_result_path = os.path.join(model_dir, "poser_results.json")
    with open(poser_result_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "benchmark_path": benchmark_path,
            "intervention_layer": intervention_layer,
            "poser_results": poser_results,
            "n_samples": n_samples
        }, f, indent=2)


def run_tyche_all_layers(model_path, benchmark_path, intervention_layer=15, n_samples=10, val_size = 25, iters = 100, tyche_cutoff=1e-2, base_output_dir= "compare_perturbations", device="cuda:0"):
    poser_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_name = model_path.split("models/")[-1].replace("-", "_").replace("/", "")
    model_dir = os.path.join(poser_dir, base_output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # # Load benchmark data
    print("Loading benchmark data...")
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    
    train_dataset, eval_dataset = process_caa_dataset(benchmark, train_size=n_samples)
    print("len train_dataset", len(train_dataset), " eval_dataset", len(eval_dataset))

    # Run Tyche
    print("\n=== Tyche Analysis (ALL LAYERS) ===")
    tyche_results = {}
    tyche_model, tyche_tokenizer = _load_model_and_tokenizer(model_path)

    # Forward pass with a known safe token
    if tyche_tokenizer.pad_token_id is None:
        if tyche_tokenizer.eos_token_id is not None:
            tyche_tokenizer.pad_token_id = tyche_tokenizer.eos_token_id

    try:
        inputs = tyche_tokenizer("Hello world", return_tensors="pt").to(device)

        with torch.no_grad():
            logits = tyche_model(**inputs).logits
        print("[PASS] Forward pass succeeded.")
    except Exception as e:
        print("[FAIL] Forward pass error:")
        print(e)

    # Fix tokenizer pad token
    if tyche_tokenizer.pad_token is None:
        print("[INFO] Setting pad_token to eos_token.")
        tyche_tokenizer.pad_token = tyche_tokenizer.eos_token

    # Resize model vocab if needed
    if tyche_model.config.vocab_size != len(tyche_tokenizer):
        print(f"[INFO] Resizing model embeddings from {tyche_model.config.vocab_size} to {len(tyche_tokenizer)}")
        # model.resize_token_embeddings(len(tokenizer))
        tyche_model.resize_token_embeddings(len(tyche_tokenizer), mean_resizing=False)

        tyche_model.tie_weights()

    validate_model_tokenizer_alignment(tyche_model, tyche_tokenizer)
        
    # Create a dataset object for Tyche from the prompts
    tyche_dataset = Dataset.from_dict({"text": train_dataset["clean_prompts"][:n_samples]})
    tyche_config = VolumeConfig(
            model=tyche_model,
            tokenizer=tyche_tokenizer,
            dataset=tyche_dataset,
            text_key="text",
            n_samples=n_samples,
            cutoff=tyche_cutoff,
            max_seq_len=512,
            val_size=val_size,
            cache_mode="cpu",
            chunking=False,
            implicit_vectors=True,
            iters=iters,
            allow_unconverged=True
        )
    print(f"Running Tyche volume estimator with {n_samples} samples...")

    estimator = VolumeEstimator.from_config(tyche_config)
    result = estimator.run()
    
    # Note: Tyche scales vectors to achieve the target KL cutoff
    scaled_vectors = []
    scaled_norms = []
    kl_values = []

    # Extract perturbation vectors and their scaled norms lets fix this in tyche later on
    logits_p = estimator.latest_original_logits # one tensor # batch size, sequence length, vocab size
    logits_q = torch.cat(estimator.latest_perturbed_logits, dim=0)  #  len([(batch size, sequence length, vocab siz)...]) == val_size


    tyche_orig_logits = estimator.latest_original_logits
    tyche_pert_logits = torch.cat(estimator.latest_perturbed_logits, dim=0) 


    torch.save(tyche_orig_logits, os.path.join(model_dir, "logits_original.pt"))
    torch.save(tyche_pert_logits, os.path.join(model_dir, "logits_perturbed.pt"))
    torch.save(result.volume, os.path.join(model_dir, "volume_all_layers.pt"))


    # Save Tyche prompts
    with open(os.path.join(model_dir, "sequence_input.json"), "w") as f:
        json.dump(train_dataset["clean_prompts"][:n_samples], f, indent=2)
        

    # TODO : RIGHT NOW this is from ALL perturbed logits concatenated:
    # compute kl divergence of val size things, and then take the average. and this is the one kl divergence we look at for tyche. 
    print("   len(estimator.latest_perturbed_logits)", len(estimator.latest_perturbed_logits))
    print("   len(result.estimates)", len(result.estimates))
    # Save Tyche tensors
    torch.save(result.mults, os.path.join(model_dir, "mults.pt"))
    torch.save(result.deltas, os.path.join(model_dir, "deltas.pt"))
    torch.save(result.props, os.path.join(model_dir, "props.pt"))  # Save proposal vector lengths

    tyche_results["all_layers"] = {
        "mults_path": os.path.join(model_dir, "mults.pt"),
        "deltas_path": os.path.join(model_dir, "deltas.pt"),
        "props_path": os.path.join(model_dir, "props.pt"),
        "logits_original_path": os.path.join(model_dir, "logits_original.pt"),
        "logits_perturbed_path": os.path.join(model_dir, "logits_perturbed.pt"),
        "volume_path": os.path.join(model_dir, "volume_all_layers.pt")


    }


    tyche_all_result_path = os.path.join(model_dir, "tyche_all_results.json")
    with open(tyche_all_result_path, "w") as f:
        json.dump({
            "tyche_cutoff": tyche_cutoff,
            "val_size": val_size,
            "iters": iters,
            "n_samples": n_samples,
            "tyche_results_all": tyche_results["all_layers"]
        }, f, indent=2)


    
def run_tyche_one_layer(model_path, benchmark_path, intervention_layer=15, n_samples=10, val_size = 25, iters = 100, tyche_cutoff=1e-2, base_output_dir= "compare_perturbations", device="cuda:0"):
    poser_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_name = model_path.split("models/")[-1].replace("-", "_").replace("/", "")
    model_dir = os.path.join(poser_dir, base_output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    print("\n=== Tyche Layer-Specific Perturbation ===")
    # # Load benchmark data
    print("Loading benchmark data...")
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    
    train_dataset, eval_dataset = process_caa_dataset(benchmark, train_size=n_samples)
    print("len train_dataset", len(train_dataset), " eval_dataset", len(eval_dataset))

    # Run Tyche
    print("\n=== Tyche Analysis (ALL LAYERS) ===")
    tyche_results = {}
    tyche_model, tyche_tokenizer = _load_model_and_tokenizer(model_path)

    # Forward pass with a known safe token
    if tyche_tokenizer.pad_token_id is None:
        if tyche_tokenizer.eos_token_id is not None:
            tyche_tokenizer.pad_token_id = tyche_tokenizer.eos_token_id

    try:
        inputs = tyche_tokenizer("Hello world", return_tensors="pt").to(device)

        with torch.no_grad():
            logits = tyche_model(**inputs).logits
        print("[PASS] Forward pass succeeded.")
    except Exception as e:
        print("[FAIL] Forward pass error:")
        print(e)

    # Fix tokenizer pad token
    if tyche_tokenizer.pad_token is None:
        print("[INFO] Setting pad_token to eos_token.")
        tyche_tokenizer.pad_token = tyche_tokenizer.eos_token

    # Resize model vocab if needed
    if tyche_model.config.vocab_size != len(tyche_tokenizer):
        print(f"[INFO] Resizing model embeddings from {tyche_model.config.vocab_size} to {len(tyche_tokenizer)}")
        # model.resize_token_embeddings(len(tokenizer))
        tyche_model.resize_token_embeddings(len(tyche_tokenizer), mean_resizing=False)

        tyche_model.tie_weights()

    validate_model_tokenizer_alignment(tyche_model, tyche_tokenizer)
        
    # Create a dataset object for Tyche from the prompts
    tyche_dataset = Dataset.from_dict({"text": train_dataset["clean_prompts"][:n_samples]})

    # Get parameter mask for only layer 15
    # Create a second config with perturbation mask
    tyche_layer_config = VolumeConfig(
        model=tyche_model,
        tokenizer=tyche_tokenizer,
        dataset=tyche_dataset,
        text_key="text",
        n_samples=n_samples,
        cutoff=tyche_cutoff,
        max_seq_len=512,
        val_size=val_size,
        cache_mode="cpu",
        chunking=False,
        implicit_vectors=True,
        iters=iters,
        allow_unconverged=True,
        block_size=2 * 1024,
        filter_str=f"layers.{intervention_layer}"  # layer filtering
    )

    layer_estimator = VolumeEstimator.from_config(tyche_layer_config)
    layer_result = layer_estimator.run()


    # Save layer-specific Tyche tensors
    torch.save(layer_result.deltas, os.path.join(model_dir, "layer15_deltas.pt"))
    torch.save(layer_result.mults, os.path.join(model_dir, "layer15_mults.pt"))
    torch.save(layer_result.props, os.path.join(model_dir, "layer15_props.pt"))


    # Save layer-specific logits
    layer_orig_logits = layer_estimator.latest_original_logits
    layer_pert_logits = torch.cat(layer_estimator.latest_perturbed_logits, dim=0)
    torch.save(layer_orig_logits, os.path.join(model_dir, "layer15_logits_original.pt"))
    torch.save(layer_pert_logits, os.path.join(model_dir, "layer15_logits_perturbed.pt"))
    torch.save(layer_result.volume, os.path.join(model_dir, "volume_layer15.pt"))


    # Save metadata for layer-specific results
    tyche_layer_results = {
        "deltas_path": os.path.join(model_dir, "layer15_deltas.pt"),
        "mults_path": os.path.join(model_dir, "layer15_mults.pt"),
        "props_path": os.path.join(model_dir, "layer15_props.pt"),
        "logits_original_path": os.path.join(model_dir, "layer15_logits_original.pt"),
        "logits_perturbed_path": os.path.join(model_dir, "layer15_logits_perturbed.pt"),
        "volume_path": os.path.join(model_dir, "volume_layer15.pt")

    }

    tyche_layer_result_path = os.path.join(model_dir, "tyche_layer_results.json")
    with open(tyche_layer_result_path, "w") as f:
        json.dump({
            "tyche_cutoff": tyche_cutoff,
            "val_size": val_size,
            "iters": iters,
            "n_samples": n_samples,
            "tyche_results_layer_specific": tyche_layer_results
        }, f, indent=2)



# def compare_poser_and_tyche(model_path, benchmark_path, intervention_layer=15, n_samples=10, val_size = 25, iters = 100, tyche_cutoff=1e-2, base_output_dir= "compare_perturbations", device="cuda:0"):
#     # Get the absolute path to the POSER directory (go up one more level)
#     # Create model-specific directory in compare_perturbations using absolute path

#     poser_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     model_name = model_path.split("models/")[-1].replace("-", "_").replace("/", "")
#     model_dir = os.path.join(poser_dir, base_output_dir, model_name)

#     print(f"Model name: {model_name}") # Get just the last part of the path and replace hyphens
#     print(f"POSER directory: {poser_dir}")
#     print(f"Model path: {model_path}")
#     print(f"Model name: {model_name}")
#     print(f"Model directory: {model_dir}")
#     os.makedirs(model_dir, exist_ok=True)
#     output_path = os.path.join(model_dir, f"layer{intervention_layer}_kl{tyche_cutoff}.json")

#     # # Load benchmark data
#     print("Loading benchmark data...")
#     with open(benchmark_path, "r") as f:
#         benchmark = json.load(f)
    
#     train_dataset, eval_dataset = process_caa_dataset(benchmark, train_size=n_samples)
#     print("len train_dataset", len(train_dataset), " eval_dataset", len(eval_dataset))
    
#     print("\n=== POSER Analysis ===")
#     # Compare POSER and Tyche perturbations on the same model."""
#     print("Loading model for POSER...")
#     poser_model = LanguageModel(model_path, device_map={"": 0}, torch_dtype=torch.bfloat16)
#     poser = CAA(intervention_layer=intervention_layer)
    
#     # Create and measure steering vector
#     print("Creating POSER steering vector...")
#     poser_vector_norm = poser.create_steering_vector(poser_model, train_dataset)
#     print(f"POSER vector norm: {poser_vector_norm}")
    
#     # Calculate KL divergence for different coefficients
#     coeffs = [0] + list(range(1, 2, 1))
#     poser_results = {}
    
#     print("Measuring KL divergence for different POSER coefficients...")
#     for coeff in coeffs:
#         kl_div = poser.compute_kl_divergence(poser_model, eval_dataset, coeff=coeff, model_name=model_name, model_dir=model_dir)
#         poser_results[coeff] = {
#             "kl_div": kl_div,
#             "vector_norm": poser_vector_norm * coeff
#         }
#         print(f"POSER coeff={coeff}, KL={kl_div:.6f}, vector_norm={poser_vector_norm * coeff:.6f}")
    
#     # Release POSER model to free GPU memory
#     del poser_model
#     torch.cuda.empty_cache()
#     gc.collect()
#     deep_clean()
#     aggressive_model_cleanup(model=poser_model):



    
#     # Run Tyche
#     print("\n=== Tyche Analysis (ALL LAYERS) ===")
#     tyche_results = {}
#     tyche_model, tyche_tokenizer = _load_model_and_tokenizer(model_path)

#     # Forward pass with a known safe token
#     if tyche_tokenizer.pad_token_id is None:
#         if tyche_tokenizer.eos_token_id is not None:
#             tyche_tokenizer.pad_token_id = tyche_tokenizer.eos_token_id

#     try:
#         inputs = tyche_tokenizer("Hello world", return_tensors="pt").to(args.device)
#         with torch.no_grad():
#             logits = tyche_model(**inputs).logits
#         print("[PASS] Forward pass succeeded.")
#     except Exception as e:
#         print("[FAIL] Forward pass error:")
#         print(e)

#     # Fix tokenizer pad token
#     if tyche_tokenizer.pad_token is None:
#         print("[INFO] Setting pad_token to eos_token.")
#         tyche_tokenizer.pad_token = tyche_tokenizer.eos_token

#     # Resize model vocab if needed
#     if tyche_model.config.vocab_size != len(tyche_tokenizer):
#         print(f"[INFO] Resizing model embeddings from {tyche_model.config.vocab_size} to {len(tyche_tokenizer)}")
#         # model.resize_token_embeddings(len(tokenizer))
#         tyche_model.resize_token_embeddings(len(tyche_tokenizer), mean_resizing=False)

#         tyche_model.tie_weights()

#     validate_model_tokenizer_alignment(tyche_model, tyche_tokenizer)
        
#     # Create a dataset object for Tyche from the prompts
#     tyche_dataset = Dataset.from_dict({"text": train_dataset["clean_prompts"][:n_samples]})
#     tyche_config = VolumeConfig(
#             model=tyche_model,
#             tokenizer=tyche_tokenizer,
#             dataset=tyche_dataset,
#             text_key="text",
#             n_samples=n_samples,
#             cutoff=tyche_cutoff,
#             max_seq_len=512,
#             val_size=val_size,
#             cache_mode="cpu",
#             chunking=False,
#             implicit_vectors=True,
#             iters=iters,
#             allow_unconverged=True
#         )
#     print(f"Running Tyche volume estimator with {n_samples} samples...")

#     estimator = VolumeEstimator.from_config(tyche_config)
#     result = estimator.run()
    
#     # Note: Tyche scales vectors to achieve the target KL cutoff
#     scaled_vectors = []
#     scaled_norms = []
#     kl_values = []

#     # Extract perturbation vectors and their scaled norms lets fix this in tyche later on
#     logits_p = estimator.latest_original_logits # one tensor # batch size, sequence length, vocab size
#     logits_q = torch.cat(estimator.latest_perturbed_logits, dim=0)  #  len([(batch size, sequence length, vocab siz)...]) == val_size


#     tyche_orig_logits = estimator.latest_original_logits
#     tyche_pert_logits = torch.cat(estimator.latest_perturbed_logits, dim=0) 


#     torch.save(tyche_orig_logits, os.path.join(model_dir, "logits_original.pt"))
#     torch.save(tyche_pert_logits, os.path.join(model_dir, "logits_perturbed.pt"))

#     # Save Tyche prompts
#     with open(os.path.join(model_dir, "sequence_input.json"), "w") as f:
#         json.dump(train_dataset["clean_prompts"][:n_samples], f, indent=2)
        

#     # TODO : RIGHT NOW this is from ALL perturbed logits concatenated:
#     # compute kl divergence of val size things, and then take the average. and this is the one kl divergence we look at for tyche. 
#     print("   len(estimator.latest_perturbed_logits)", len(estimator.latest_perturbed_logits))
#     print("   len(result.estimates)", len(result.estimates))
#     # Save Tyche tensors
#     torch.save(result.mults, os.path.join(model_dir, "mults.pt"))
#     torch.save(result.deltas, os.path.join(model_dir, "deltas.pt"))
#     torch.save(result.props, os.path.join(model_dir, "props.pt"))  # Save proposal vector lengths

#     tyche_results["all_layers"] = {
#         "mults_path": os.path.join(model_dir, "mults.pt"),
#         "deltas_path": os.path.join(model_dir, "deltas.pt"),
#         "props_path": os.path.join(model_dir, "props.pt"),
#         "logits_original_path": os.path.join(model_dir, "layer15_logits_original.pt"),
#         "logits_perturbed_path": os.path.join(model_dir, "layer15_logits_perturbed.pt")
#     }

#     import pynvml
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     deep_clean()
#     print(f"Before reload: Free = {info.free / 1024 ** 3:.2f} GB, Used = {info.used / 1024 ** 3:.2f} GB")


#     # Free up memory after all-layer Tyche analysis
#     # Aggressively force cleanup
#     for _ in range(3):
#         gc.collect()
#         torch.cuda.empty_cache()

#     # Check for leftover tensors (properly indented)
#     for obj in gc.get_objects():
#         try:
#             if torch.is_tensor(obj) and obj.is_cuda:
#                 print(f"Tensor: {type(obj)}, size: {obj.size()}, dtype: {obj.dtype}")
#         except:
#             pass

#     import pynvml
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     force_reset_gpu()
#     deep_clean()
#     aggressive_model_cleanup(model=tyche_model)


#     tyche_model, tyche_tokenizer = _load_model_and_tokenizer(model_path)

#     print(f"After reload: Free = {info.free / 1024 ** 3:.2f} GB, Used = {info.used / 1024 ** 3:.2f} GB")


#     print(f"[DEBUG] CUDA reserved after reload: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
#     print(f"[DEBUG] CUDA allocated after reload: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

#     print("\n=== Tyche Layer-Specific Perturbation ===")

#     # Get parameter mask for only layer 15
#     # Create a second config with perturbation mask
#     tyche_layer_config = VolumeConfig(
#         model=tyche_model,
#         tokenizer=tyche_tokenizer,
#         dataset=tyche_dataset,
#         text_key="text",
#         n_samples=n_samples,
#         cutoff=tyche_cutoff,
#         max_seq_len=512,
#         val_size=val_size,
#         cache_mode="cpu",
#         chunking=False,
#         implicit_vectors=True,
#         iters=iters,
#         allow_unconverged=True,
#         block_size=2 * 1024,
#         filter_str=f"layers.{intervention_layer}"  # layer filtering
#     )

#     layer_estimator = VolumeEstimator.from_config(tyche_layer_config)
#     layer_result = layer_estimator.run()


#     # Save layer-specific Tyche tensors
#     torch.save(layer_result.deltas, os.path.join(model_dir, "layer15_deltas.pt"))
#     torch.save(layer_result.mults, os.path.join(model_dir, "layer15_mults.pt"))
#     torch.save(layer_result.props, os.path.join(model_dir, "layer15_props.pt"))


#     # Save layer-specific logits
#     layer_orig_logits = layer_estimator.latest_original_logits
#     layer_pert_logits = torch.cat(layer_estimator.latest_perturbed_logits, dim=0)
#     torch.save(layer_orig_logits, os.path.join(model_dir, "layer15_logits_original.pt"))
#     torch.save(layer_pert_logits, os.path.join(model_dir, "layer15_logits_perturbed.pt"))

#     # Add layer-specific metrics to results
#     tyche_results["layer_specific"] = {
#         "mults_path": os.path.join(model_dir, "layer15_mults.pt"),
#         "deltas_path": os.path.join(model_dir, "layer15_deltas.pt"),
#         "props_path": os.path.join(model_dir, "layer15_props.pt"),
#         "logits_original_path": os.path.join(model_dir, "layer15_logits_original.pt"),
#         "logits_perturbed_path": os.path.join(model_dir, "layer15_logits_perturbed.pt")
#     }
    
    # # Compare results
    # print("\n=== Comparison Results ===")
    # with open(output_path, "w") as f:
    #     json.dump({
    #         "model_path": model_path,
    #         "benchmark_path": benchmark_path,
    #         "intervention_layer": intervention_layer,
    #         "tyche_cutoff": tyche_cutoff,
    #         "poser_results": poser_results,
    #         "val_size": val_size, 
    #         "n_samples": n_samples,
    #         "iters": iters,
    #         "tyche_results": tyche_results
    #     }, f, indent=2)
    # print(f"Saved results to {output_path}")

def run_subprocess(cmd):
    print(f"\n=== Running: {' '.join(cmd)} ===\n")
    # subprocess.run(cmd, check=True)
    subprocess.run(cmd, check=True, cwd="/mnt/ssd-1/dipika/POSER")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='Compare POSER and Tyche perturbation approaches')
#     parser.add_argument('--model_path', type=str, required=True,
#                         help='Path to the model')
#     parser.add_argument('--benchmark_path', type=str, required=True,
#                         help='Path to the benchmark dataset')
#     parser.add_argument('--intervention_layer', type=int, default=15,
#                         help='Layer to intervene on for POSER')
#     parser.add_argument('--n_samples', type=int, default=10,
#                         help='Number of Tyche samples')
#     parser.add_argument('--val_size', type=int, default=25,
#                         help='Number of Tyche samples')
#     parser.add_argument('--iters', type=int, default=100,
#                         help='Number of Tyche samples')
#     parser.add_argument('--tyche_cutoff', type=float, default=1e-2,
#                         help='Target KL divergence for Tyche')
#     parser.add_argument('--base_output_dir', type=str, default="compare_perturbations",
#                     help='Base directory where results will be saved')
#     parser.add_argument('--device', type=str, default="cuda:0", help='Device to run model on')

    
#     args = parser.parse_args()

#     # Reusable base command
#     base_args = [
#         "--model_path", args.model_path,
#         "--benchmark_path", args.benchmark_path,
#         "--intervention_layer", str(args.intervention_layer),
#         "--n_samples", str(args.n_samples),
#         "--val_size", str(args.val_size),
#         "--iters", str(args.iters),
#         "--tyche_cutoff", str(args.tyche_cutoff),
#         "--base_output_dir", args.base_output_dir,
#         "--device", args.device
#     ]
    

#     # Step 1: POSER
#     run_subprocess(["python", "run_analysis.py", "--mode", "poser"] + base_args)

#     # Step 2: Tyche All Layers
#     run_subprocess(["python", "run_analysis.py", "--mode", "tyche_all"] + base_args)

#     # Step 3: Tyche One Layer
#     run_subprocess(["python", "run_analysis.py", "--mode", "tyche_one"] + base_args)


#     # results = compare_poser_and_tyche(
#     #     model_path=args.model_path,
#     #     benchmark_path=args.benchmark_path,
#     #     intervention_layer=args.intervention_layer,
#     #     n_samples=args.n_samples,
#     #     val_size = args.val_size,
#     #     iters = args.iters,
#     #     tyche_cutoff=args.tyche_cutoff,
#     #     base_output_dir=args.base_output_dir,
#     #     device=args.device,
#     # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare POSER and Tyche perturbation approaches')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--benchmark_path', type=str, required=True,
                        help='Path to the benchmark dataset')
    parser.add_argument('--intervention_layer', type=int, default=15,
                        help='Layer to intervene on for POSER')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of Tyche samples')
    parser.add_argument('--val_size', type=int, default=25,
                        help='Number of Tyche samples')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of Tyche samples')
    parser.add_argument('--tyche_cutoff', type=float, default=1e-2,
                        help='Target KL divergence for Tyche')
    parser.add_argument('--base_output_dir', type=str, default="compare_perturbations",
                    help='Base directory where results will be saved')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to run model on')
    
    # Add a special argument to determine which function to run
    parser.add_argument('--run_mode', type=str, choices=['all', 'poser', 'tyche_all', 'tyche_one'], 
                      default='all', help='Which function to run')
    
    args = parser.parse_args()
    
    # Check if we should run a specific function
    if args.run_mode == 'poser':
        run_poser(
            model_path=args.model_path,
            benchmark_path=args.benchmark_path,
            intervention_layer=args.intervention_layer,
            n_samples=args.n_samples,
            val_size=args.val_size,
            iters=args.iters,
            tyche_cutoff=args.tyche_cutoff,
            base_output_dir=args.base_output_dir,
            device=args.device
        )
    elif args.run_mode == 'tyche_all':
        run_tyche_all_layers(
            model_path=args.model_path,
            benchmark_path=args.benchmark_path,
            intervention_layer=args.intervention_layer,
            n_samples=args.n_samples,
            val_size=args.val_size,
            iters=args.iters,
            tyche_cutoff=args.tyche_cutoff,
            base_output_dir=args.base_output_dir,
            device=args.device
        )
    elif args.run_mode == 'tyche_one':
        run_tyche_one_layer(
            model_path=args.model_path,
            benchmark_path=args.benchmark_path,
            intervention_layer=args.intervention_layer,
            n_samples=args.n_samples,
            val_size=args.val_size,
            iters=args.iters,
            tyche_cutoff=args.tyche_cutoff,
            base_output_dir=args.base_output_dir,
            device=args.device
        )
    else:  # run_mode == 'all'
        # Run each phase in a separate process
        script_path = os.path.abspath(__file__)  # Get the path to this script
        
        # Function to run a subprocess
        def run_subprocess(run_mode):
            cmd = [
                sys.executable, script_path,
                "--model_path", args.model_path,
                "--benchmark_path", args.benchmark_path,
                "--intervention_layer", str(args.intervention_layer),
                "--n_samples", str(args.n_samples),
                "--val_size", str(args.val_size),
                "--iters", str(args.iters),
                "--tyche_cutoff", str(args.tyche_cutoff),
                "--base_output_dir", args.base_output_dir,
                "--device", args.device,
                "--run_mode", run_mode
            ]
            print(f"\n=== Running {run_mode} in separate process ===\n")
            subprocess.run(cmd, check=True)
        
        # Run each phase
        run_subprocess("poser")
        run_subprocess("tyche_all")
        run_subprocess("tyche_one")
        
        # Combine results at the end
        poser_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_name = args.model_path.split("models/")[-1].replace("-", "_").replace("/", "")
        model_dir = os.path.join(poser_dir, args.base_output_dir, model_name)
        output_path = os.path.join(model_dir, f"layer{args.intervention_layer}_kl{args.tyche_cutoff}.json")
        
        try:
            # Load individual results
            with open(os.path.join(model_dir, "poser_results.json"), "r") as f:
                poser_results = json.load(f)
                
            with open(os.path.join(model_dir, "tyche_all_results.json"), "r") as f:
                tyche_all_results = json.load(f)
                
            with open(os.path.join(model_dir, "tyche_layer_results.json"), "r") as f:
                tyche_layer_results = json.load(f)
                
            # Combine results
            combined_results = {
                "model_path": args.model_path,
                "benchmark_path": args.benchmark_path,
                "intervention_layer": args.intervention_layer,
                "tyche_cutoff": args.tyche_cutoff,
                "poser_results": poser_results.get("poser_results", {}),
                "val_size": args.val_size, 
                "n_samples": args.n_samples,
                "iters": args.iters,
                "tyche_results": {
                    "all_layers": tyche_all_results.get("tyche_results_all", {}),
                    "layer_specific": tyche_layer_results.get("tyche_results_layer_specific", {})
                }
            }
            
            # Save combined results
            with open(output_path, "w") as f:
                json.dump(combined_results, f, indent=2)
                
            print(f"\n=== Comparison Complete ===")
            print(f"Combined results saved to: {output_path}")
        except FileNotFoundError as e:
            print(f"Warning: Could not combine results - {e}")
import torch as t
from nnsight import LanguageModel
from tqdm import tqdm
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
import os


# Import Tyche's actual classes
from tyche import VolumeConfig, VolumeEstimator, aggregate

# ----- POSER Code (closely following the original implementation) -----

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
        random_vector = t.rand_like(h_k)

        # Calculate the norm of the control vector h_k
        norm_h_k = t.linalg.norm(h_k)

        # Normalize the random vector to have the same norm as h_k
        norm_random_vector = t.linalg.norm(random_vector)
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
            with t.no_grad():
                with model.trace(clean_prompt, validate=False, scan=False):
                    clean_act = model.model.layers[self.intervention_layer].output[0][:,-1,:].save()

            clean_acts.append(clean_act)
        
        clean_acts = t.cat(clean_acts, 0).mean(0).unsqueeze(0).cpu()

        critical_acts = []
        for critical_prompt in tqdm(critical):
            with t.no_grad():
                with model.trace(critical_prompt, validate=False, scan=False):
                    critical_act = model.model.layers[self.intervention_layer].output[0][:,-1,:].save()

            critical_acts.append(critical_act)
        
        critical_acts = t.cat(critical_acts, 0).mean(0).unsqueeze(0).cpu()

        print("Steering vector obtained.")
        self.steering_vector = critical_acts - clean_acts
        self.steering_vector = self.replace_with_uniform_random_vector(self.steering_vector)
        t.cuda.empty_cache()
        
        # Return the norm of the steering vector
        return t.linalg.norm(self.steering_vector).item()

    def compute_kl_divergence(self, model: LanguageModel, eval_dataset, coeff=1, n_eval=10):
        """Modified POSER function to compute KL divergence between original and perturbed outputs."""
        tokenizer = model.tokenizer
        eval_prompts = [x["prompt"] for x in eval_dataset][:n_eval]
        
        kl_divs = []
        
        for prompt in tqdm(eval_prompts):
            with t.no_grad():
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
                
                probs_orig = t.nn.functional.softmax(logits_orig, dim=-1)
                log_probs_perturbed = t.nn.functional.log_softmax(logits_perturbed, dim=-1)
                
                kl_div = t.nn.functional.kl_div(
                    log_probs_perturbed, 
                    probs_orig, 
                    reduction='batchmean'
                )
                kl_divs.append(kl_div.item())
                
                t.cuda.empty_cache()
        
        avg_kl = sum(kl_divs) / len(kl_divs)
        return avg_kl


# ----- Helper function to measure KL for Tyche vectors -----

def measure_tyche_kl(model, tokenizer, perturbation_vector, dataset, last_token_only=True):
    """Measure KL divergence between original and Tyche-perturbed outputs."""
    model.eval()
    
    # Store original parameters
    original_params = t.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    
    # Create perturbed parameters
    perturbed_params = original_params + perturbation_vector.to(original_params.device)
    
    # Tokenize dataset
    tokens = tokenizer(dataset, 
                      padding=True, 
                      truncation=True, 
                      max_length=512, 
                      return_tensors="pt")['input_ids'].to(model.device)
    
    kl_divs = []
    
    try:
        # Apply perturbation
        t.nn.utils.vector_to_parameters(perturbed_params, model.parameters())
        
        # Get perturbed logits
        with t.no_grad():
            outputs_q = model(tokens)
            logits_q = outputs_q.logits
            
            if last_token_only:
                # Only consider last token
                logits_q = logits_q[:, -1:, :]
            
            log_probs_q = t.nn.functional.log_softmax(logits_q, dim=-1)
        
        # Restore original parameters
        t.nn.utils.vector_to_parameters(original_params, model.parameters())
        
        # Get original logits
        with t.no_grad():
            outputs_p = model(tokens)
            logits_p = outputs_p.logits
            
            if last_token_only:
                # Only consider last token
                logits_p = logits_p[:, -1:, :]
            
            probs_p = t.nn.functional.softmax(logits_p, dim=-1)
        
        # Compute KL divergence
        kl_all = t.nn.functional.kl_div(log_probs_q, probs_p, reduction='none').sum(dim=-1)
        mask = tokens != tokenizer.pad_token_id
        
        if last_token_only:
            # For last token only, adjust the mask
            last_mask = t.zeros_like(mask)
            for i in range(mask.shape[0]):
                last_non_pad = mask[i].nonzero().max()
                last_mask[i, last_non_pad] = True
            mask = last_mask
        
        kl = t.mean(kl_all[mask])
        return kl.item()
    
    finally:
        # Ensure original parameters are restored
        t.nn.utils.vector_to_parameters(original_params, model.parameters())



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
        tokens = self.tokenizer(self.dataset[:5],  # Using first 5 prompts for speed
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


# ----- Main Comparison Function -----

def compare_poser_and_tyche(model_path, benchmark_path, intervention_layer=15, n_samples=10, tyche_cutoff=1e-2):
    output_dir = "compare_perturbations"
    plots_path = f"{output_dir}/plots/"
    model_name = model_path.split("/")[-1]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    output_path = os.path.join(output_dir, f"{model_name}_layer{intervention_layer}_kl{tyche_cutoff}.json")

    """Compare POSER and Tyche perturbations on the same model."""
    # Load model for POSER (using nnsight)
    print("Loading model for POSER...")
    poser_model = LanguageModel(model_path, device_map={"": 0}, torch_dtype=t.bfloat16)
    
    # Load benchmark data
    print("Loading benchmark data...")
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    
    # Process dataset for POSER
    print("Processing dataset for POSER...")
    train_dataset, eval_dataset = process_caa_dataset(benchmark, train_size=10)
    
    # Run POSER
    print("\n=== POSER Analysis ===")
    poser = CAA(intervention_layer=intervention_layer)
    
    # Create and measure steering vector
    print("Creating POSER steering vector...")
    poser_vector_norm = poser.create_steering_vector(poser_model, train_dataset)
    print(f"POSER vector norm: {poser_vector_norm}")
    
    # Calculate KL divergence for different coefficients
    coeffs = [0.5, 1, 2, 5, 10]
    poser_results = {}
    
    print("Measuring KL divergence for different POSER coefficients...")
    for coeff in coeffs:
        kl_div = poser.compute_kl_divergence(poser_model, eval_dataset, coeff=coeff)
        poser_results[coeff] = {
            "kl_div": kl_div,
            "vector_norm": poser_vector_norm * coeff
        }
        print(f"POSER coeff={coeff}, KL={kl_div:.6f}, vector_norm={poser_vector_norm * coeff:.6f}")
    
    # Release POSER model to free GPU memory
    del poser_model
    t.cuda.empty_cache()
    gc.collect()
    
    # Run Tyche
    print("\n=== Tyche Analysis ===")
    
    # Load HF model for Tyche
    print("Loading model for Tyche...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    
    tyche_model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": 1}, torch_dtype=t.bfloat16)
    tyche_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Handle potential tokenizer issues
    if tyche_tokenizer.pad_token is None:
        tyche_tokenizer.pad_token = tyche_tokenizer.eos_token
        
    # Create a dataset object for Tyche from the prompts
    tyche_dataset = Dataset.from_dict({"text": train_dataset["clean_prompts"][:5]})
    print(f"Model vocab size: {tyche_model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tyche_tokenizer)}")

    # Set up Tyche config
    print("Creating Tyche configuration...")
    # # Handle potential tokenizer issues
    # if tyche_tokenizer.pad_token is None:
    #     tyche_tokenizer.pad_token = tyche_tokenizer.eos_token
    #     tyche_model.resize_token_embeddings(len(tyche_tokenizer))  # Ensure model matches tokenizer vocab

    # Make sure they match
    if tyche_model.config.vocab_size != len(tyche_tokenizer):
        print("Adjusting model vocab size to match tokenizer...")
        tyche_model.resize_token_embeddings(len(tyche_tokenizer))

    tyche_config = VolumeConfig(
        model=tyche_model,
        tokenizer=tyche_tokenizer,
        dataset=tyche_dataset,
        text_key="text",
        val_size=1,  # Use a small number for faster evaluation
        cutoff=tyche_cutoff,
        n_samples=n_samples,
        cache_mode="cpu",
        model_type="causal",
        implicit_vectors=True,
        max_seq_len=512,
        last_token_only=True,  # Match POSER's approach of using only the last token
        iters=10,  # Maximum line search iterations
        allow_unconverged=True  # Allow unconverged directions
    )
    
    # Create and run Tyche estimator
    print(f"Running Tyche volume estimator with {n_samples} samples...")
    estimator = VolumeEstimator.from_config(tyche_config)
    result = estimator.run()
    
    # Extract perturbation vectors and their scaled norms
    # Note: Tyche scales vectors to achieve the target KL cutoff
    scaled_vectors = []
    scaled_norms = []
    kl_values = []
    
    print("Extracting Tyche perturbation vectors...")
    for i, vec in enumerate(result.vectors):
        if vec is not None:
            # Tyche's vectors are already scaled to achieve the target KL
            scaled_vectors.append(vec)
            norm = t.linalg.norm(vec).item()
            scaled_norms.append(norm)
            
            # Verify KL value (optional, as Tyche should have already targeted the cutoff)
            kl = measure_tyche_kl(
                tyche_model, 
                tyche_tokenizer, 
                vec, 
                train_dataset["clean_prompts"][:3]  # Use just a few prompts for speed
            )
            kl_values.append(kl)
            print(f"Tyche vector {i+1}: KL={kl:.6f}, norm={norm:.6f}")
    
    # Check if we have any valid vectors
    if len(scaled_norms) == 0:
        print("Warning: No valid Tyche vectors found. This might indicate issues with the Tyche estimation.")
        avg_tyche_norm = 0
    else:
        # Calculate average scaled norm
        avg_tyche_norm = np.mean(scaled_norms)
    
    # Find closest POSER coefficient to Tyche target KL
    closest_coeff = min(poser_results.keys(), 
                       key=lambda c: abs(poser_results[c]['kl_div'] - tyche_cutoff))
    
    # Compare results
    print("\n=== Comparison Results ===")
    print(f"Tyche target KL: {tyche_cutoff}")
    print(f"Average Tyche scaled vector norm: {avg_tyche_norm:.6f}")
    print(f"Closest POSER coefficient: {closest_coeff}")
    print(f"POSER KL: {poser_results[closest_coeff]['kl_div']:.6f}")
    print(f"POSER vector norm: {poser_results[closest_coeff]['vector_norm']:.6f}")
    
    # Calculate ratio
    if avg_tyche_norm > 0:
        ratio = poser_results[closest_coeff]['vector_norm'] / avg_tyche_norm
        print(f"Ratio (POSER/Tyche): {ratio:.6f}")
    else:
        ratio = float('nan')
        print("Ratio (POSER/Tyche): N/A (no valid Tyche vectors)")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot POSER results
    poser_kl_values = [result['kl_div'] for result in poser_results.values()]
    poser_norm_values = [result['vector_norm'] for result in poser_results.values()]
    plt.scatter(poser_kl_values, poser_norm_values, label='POSER', marker='o', s=100)
    
    # Add labels to POSER points
    for coeff, result in poser_results.items():
        plt.annotate(f"coeff={coeff}", 
                    (result['kl_div'], result['vector_norm']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Plot Tyche results if we have any
    if len(kl_values) > 0:
        plt.scatter(kl_values, scaled_norms, label='Tyche (scaled)', marker='x', s=100)
        
        # Add horizontal line for average Tyche norm
        plt.axhline(y=avg_tyche_norm, color='r', linestyle='--', alpha=0.5,
                  label=f'Avg Tyche norm: {avg_tyche_norm:.4f}')
    
    # Add vertical line for Tyche target KL
    plt.axvline(x=tyche_cutoff, color='g', linestyle='--', alpha=0.5,
              label=f'Target KL: {tyche_cutoff}')
    
    plt.xlabel('KL Divergence')
    plt.ylabel('Vector Norm')
    plt.title('Comparison of POSER and Tyche Perturbation Magnitudes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, f"comparison_layer{intervention_layer}_kl{tyche_cutoff}_{model_name}.png"))
    print(f"Plot saved to {os.path.join(plots_path, f'comparison_layer{intervention_layer}_kl{tyche_cutoff}_{model_name}.png')}")

    # Save results to JSON
    tyche_results_serializable = {
        "kl_values": kl_values,
        "scaled_norms": scaled_norms
    }
    
    with open(output_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "benchmark_path": benchmark_path,
            "intervention_layer": intervention_layer,
            "tyche_cutoff": tyche_cutoff,
            "poser_results": poser_results,
            "tyche_results": tyche_results_serializable,
            "avg_tyche_norm": avg_tyche_norm,
            "poser_closest_coeff": closest_coeff,
            "poser_closest_vector_norm": poser_results[closest_coeff]["vector_norm"],
            "ratio": ratio,
        }, f, indent=2)
    print(f"Saved results to {output_path}")
    
    return {
        "poser_results": poser_results,
        "tyche_results": {
            "kl_values": kl_values,
            "scaled_norms": scaled_norms
        },
        "poser_closest_coeff": closest_coeff,
        "avg_tyche_norm": avg_tyche_norm,
        "ratio": ratio
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare POSER and Tyche perturbation approaches')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--benchmark_path', type=str, required=True,
                        help='Path to the benchmark dataset')
    parser.add_argument('--intervention_layer', type=int, default=15,
                        help='Layer to intervene on for POSER')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of Tyche samples')
    parser.add_argument('--tyche_cutoff', type=float, default=1e-2,
                        help='Target KL divergence for Tyche')
    
    args = parser.parse_args()
    
    results = compare_poser_and_tyche(
        model_path=args.model_path,
        benchmark_path=args.benchmark_path,
        intervention_layer=args.intervention_layer,
        n_samples=args.n_samples,
        tyche_cutoff=args.tyche_cutoff
    )
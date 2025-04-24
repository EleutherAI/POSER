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

# Import Tyche's actual classes
from tyche import VolumeConfig, VolumeEstimator, aggregate

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

    def compute_kl_divergence(self, model: LanguageModel, eval_dataset, coeff=1, n_eval=100):
        """Modified POSER function to compute KL divergence between original and perturbed outputs."""
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
        
        torch.save(orig_logits_all, os.path.join("compare_perturbations/logits_debug", f"poser_logits_original_coeff{coeff}.pt"))
        torch.save(pert_logits_all, os.path.join("compare_perturbations/logits_debug", f"poser_logits_perturbed_coeff{coeff}.pt"))
        with open(os.path.join("compare_perturbations/logits_debug", "poser_sequence_input.json"), "w") as f:
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


def compare_poser_and_tyche(model_path, benchmark_path, intervention_layer=15, n_samples=10, tyche_cutoff=1e-2):
    output_dir = "compare_perturbations"
    plots_path = f"{output_dir}/plots/"
    model_name = model_path.split("/")[-1]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    output_path = os.path.join(output_dir, f"{model_name}_layer{intervention_layer}_kl{tyche_cutoff}.json")

    """Compare POSER and Tyche perturbations on the same model."""
    # # Load model for POSER (using nnsight)
    print("Loading model for POSER...")
    poser_model = LanguageModel(model_path, device_map={"": 0}, torch_dtype=torch.bfloat16)
    
    # # Load benchmark data
    print("Loading benchmark data...")
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    
    # # Process dataset for POSER
    print("Processing dataset for POSER...")
    train_dataset, eval_dataset = process_caa_dataset(benchmark, train_size=n_samples)
    print("len train_dataset", len(train_dataset), " eval_dataset", len(eval_dataset))
    
    # Run POSER
    print("\n=== POSER Analysis ===")
    poser = CAA(intervention_layer=intervention_layer)
    
    # Create and measure steering vector
    print("Creating POSER steering vector...")
    poser_vector_norm = poser.create_steering_vector(poser_model, train_dataset)
    print(f"POSER vector norm: {poser_vector_norm}")
    
    # Calculate KL divergence for different coefficients
    # coeffs = [0.2, .4, .6, .8, 1]
    coeffs = [0] + list(range(1, 20, 1))
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
    torch.cuda.empty_cache()
    gc.collect()
    
    # Run Tyche
    print("\n=== Tyche Analysis ===")
    print("Loading model for Tyche...")

    tyche_model, tyche_tokenizer = _load_model_and_tokenizer(model_path)

    # Print current sizes
    print(f"Model vocab size:     {tyche_model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tyche_tokenizer)}")
    print(f"lm_head out_features: {tyche_model.lm_head.out_features}")
    # Forward pass with a known safe token

    if tyche_tokenizer.pad_token_id is None:
        if tyche_tokenizer.eos_token_id is not None:
            tyche_tokenizer.pad_token_id = tyche_tokenizer.eos_token_id

    try:
        inputs = tyche_tokenizer("Hello world", return_tensors="pt").to(args.device)
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

    print(f"Model vocab size:     {tyche_model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tyche_tokenizer)}")
    print(f"lm_head out_features: {tyche_model.lm_head.out_features}")
    # will have to add a row of zeros here

    # Check tokenizer special tokens (may include added tokens!)
    print("Tokenizer special tokens:", tyche_tokenizer.special_tokens_map)
    print("Tokenizer added tokens:", tyche_tokenizer.added_tokens_encoder)

    with torch.no_grad():
        original_params = torch.nn.utils.parameters_to_vector(
            [p.cpu() for p in tyche_model.parameters()]
        ).detach().clone()

    # original_params = torch.nn.utils.parameters_to_vector(tyche_model.parameters()).detach().clone()
    print("Original parameters saved successfully.")
        
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
            val_size=10,
            cache_mode="cpu",
            chunking=False,
            implicit_vectors=True,
            iters=100,
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


    torch.save(tyche_orig_logits, "compare_perturbations/logits_debug/tyche_logits_original.pt")
    torch.save(tyche_pert_logits, "compare_perturbations/logits_debug/tyche_logits_perturbed.pt")

    # Save Tyche prompts
    with open("compare_perturbations/logits_debug/tyche_sequence_input.json", "w") as f:
        json.dump(train_dataset["clean_prompts"][:n_samples], f, indent=2)
        

    # TODO : RIGHT NOW this is from ALL perturbed logits concatenated:
    # compute kl divergence of val size things, and then take the average. and this is the one kl divergence we look at for tyche. 
    total_kl = 0.0
    tyche_kl_values = []
    scaled_norms = []
    print("   len(estimator.latest_perturbed_logits)", len(estimator.latest_perturbed_logits))
    print("   len(result.estimates)", len(result.estimates))
    for i in range(len(result.estimates)):
        logits_q_i = estimator.latest_perturbed_logits[i]  # shape: (B, T, V)
        logits_p_i = estimator.latest_original_logits       # same shape

        kl_i = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_q_i, dim=-1),
            torch.nn.functional.softmax(logits_p_i, dim=-1),
            reduction="batchmean"
        )

        norm_i = torch.linalg.norm(result.estimates[i]).item()

        kl_values.append(kl_i.item())
        scaled_norms.append(norm_i)
        total_kl += kl_i.item()

    # Compute the average KL divergence
    avg_tyche_kl = total_kl / len(estimator.latest_perturbed_logits)
    # Save results to JSON
    tyche_results = {
        "avg_tyche_kl": avg_tyche_kl,
        "scaled_norms": scaled_norms
    }
    
    print("Average TYCHE KL over all samples:", avg_tyche_kl)



    # Find closest POSER coefficient to Tyche KL
    poser_kl_diffs = {coeff: abs(val["kl_div"] - avg_tyche_kl) for coeff, val in poser_results.items()}
    closest_coeff = min(poser_kl_diffs, key=poser_kl_diffs.get)
    avg_tyche_norm = sum(scaled_norms) / len(scaled_norms)


    # Compare results
    print("\n=== Comparison Results ===")
    print(f"Tyche target KL: {tyche_cutoff}")
    print(f"Average Tyche scaled vector norm: {avg_tyche_norm:.6f}")
    print(f"POSER KLCLOSEST: {poser_results[closest_coeff]['kl_div']:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter([avg_tyche_kl], [avg_tyche_norm], label='Tyche (avg)', marker='x', s=200, color='blue')

    # Plot POSER points
    poser_kl_values = [v["kl_div"] for v in poser_results.values()]
    poser_norms = [v["vector_norm"] for v in poser_results.values()]
    plt.scatter(poser_kl_values, poser_norms, label='POSER', color='orange', s=100)

    # Add labels to POSER and TYCHE points
    plt.annotate("Tyche avg", (avg_tyche_kl, avg_tyche_norm), textcoords="offset points", xytext=(0,10), ha='center')
    for coeff, result in poser_results.items():
        plt.annotate(f"coeff={coeff}", 
                    (result['kl_div'], result['vector_norm']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')

    
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

    
    with open(output_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "benchmark_path": benchmark_path,
            "intervention_layer": intervention_layer,
            "tyche_cutoff": tyche_cutoff,
            "poser_results": poser_results,
            "tyche_results": tyche_results,
            "poser_closest_coeff": closest_coeff,
            "poser_closest_vector_norm": poser_results[closest_coeff]["vector_norm"],
        }, f, indent=2)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare POSER and Tyche perturbation approaches')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--benchmark_path', type=str, required=True,
                        help='Path to the benchmark dataset')
    parser.add_argument('--intervention_layer', type=int, default=15,
                        help='Layer to intervene on for POSER')
    parser.add_argument('--n_samples', type=int, default=10,
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
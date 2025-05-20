import torch
import gc
import os
import json
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tyche import VolumeConfig, VolumeEstimator
from tyche.estimator import get_param_mask_for_layer

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

def load_model_and_tokenizer(model_path):
    """Loads a model and tokenizer with minimal memory footprint."""
    print(f"Loading model from: {model_path}")
    try:
        # Load with float16 to reduce memory usage
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,  # Half precision
            device_map="auto",         # Let it manage devices
            low_cpu_mem_usage=True,     # Minimize CPU memory
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        raise

    # Handle Tokenizer Pad Token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Resize if needed
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    
    model.eval()
    print(f"Model loaded successfully")
    return model, tokenizer

def run_layer_specific_analysis(model_path, output_dir, intervention_layer=15, tyche_cutoff=1e-2, n_samples=5):
    """Run only the layer-specific Tyche analysis."""
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sequence input if it exists
    sequence_path = os.path.join(output_dir, "sequence_input.json")
    if os.path.exists(sequence_path):
        with open(sequence_path, "r") as f:
            prompts = json.load(f)[:n_samples]
        print(f"Loaded {len(prompts)} prompts from {sequence_path}")
    else:
        # If no saved prompts, create a simple test prompt
        prompts = ["This is a test prompt to ensure the model can run."] * n_samples
        print("No saved prompts found, using test prompts")
    
    # Load model with minimal memory usage
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Create mask for layer 15 with efficient memory usage
    print(f"Creating mask for layer {intervention_layer}")
    
    # More memory-efficient mask creation
    param_sizes = [(name, p.numel()) for name, p in model.named_parameters()]
    total_params = sum(size for _, size in param_sizes)
    mask = torch.zeros(total_params, dtype=torch.bool, device="cpu")
    
    # Fill mask one parameter at a time to avoid large memory spikes
    pos = 0
    for name, size in param_sizes:
        is_target = f".{intervention_layer}." in name or f"layers.{intervention_layer}." in name
        if is_target:
            mask[pos:pos+size] = True
        pos += size
    
    print(f"Layer mask created, total parameters: {total_params}, masked: {mask.sum().item()}")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": prompts})
    
    # Configure for minimal memory usage
    config = VolumeConfig(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        text_key="text",
        n_samples=n_samples,
        cutoff=tyche_cutoff,
        max_seq_len=256,            # Reduced from 512
        val_size=min(5, n_samples), # Use at most 5 validation samples
        cache_mode="cpu",
        chunking=True,              # Enable chunking
        data_batch_size=1,          # Process one sequence at a time
        implicit_vectors=True,
        iters=1,                    # Minimal iterations
        allow_unconverged=True,     # Continue even if not fully converged
        perturbation_mask=mask.to(model.device)
    )
    
    # Run analysis
    print(f"Running Tyche layer-specific analysis with {n_samples} samples")
    estimator = VolumeEstimator.from_config(config)
    result = estimator.run()
    
    # Calculate metrics
    layer_kl_values = []
    for i in range(len(result.estimates)):
        logits_q_i = estimator.latest_perturbed_logits[i]
        logits_p_i = estimator.latest_original_logits
        kl_i = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_q_i, dim=-1),
            torch.nn.functional.softmax(logits_p_i, dim=-1),
            reduction="batchmean"
        )
        layer_kl_values.append(kl_i.item())
    
    avg_layer_kl = sum(layer_kl_values) / len(layer_kl_values)
    avg_layer_norm = result.estimates.norm(dim=-1).mean().item()
    
    # Save results
    torch.save(result.deltas.cpu(), os.path.join(output_dir, "layer15_deltas.pt"))
    torch.save(result.mults.cpu(), os.path.join(output_dir, "layer15_mults.pt"))
    torch.save(result.props.cpu(), os.path.join(output_dir, "layer15_props.pt"))
    
    # Save logits
    torch.save(estimator.latest_original_logits.cpu(), os.path.join(output_dir, "layer15_logits_original.pt"))
    layer_pert_logits = torch.cat(estimator.latest_perturbed_logits, dim=0).cpu()
    torch.save(layer_pert_logits, os.path.join(output_dir, "layer15_logits_perturbed.pt"))
    
    # Update results file
    result_path = os.path.join(output_dir, f"layer{intervention_layer}_kl{tyche_cutoff}.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            results_json = json.load(f)
    else:
        results_json = {
            "model_path": model_path,
            "intervention_layer": intervention_layer,
            "tyche_cutoff": tyche_cutoff,
            "tyche_results": {},
            "file_paths": {}
        }
    
    # Add layer-specific results
    results_json["tyche_results"]["layer_specific"] = {
        "avg_kl": avg_layer_kl,
        "avg_norm": avg_layer_norm,
        "mults_path": os.path.join(output_dir, "layer15_mults.pt"),
        "deltas_path": os.path.join(output_dir, "layer15_deltas.pt"),
        "props_path": os.path.join(output_dir, "layer15_props.pt"),
        "logits_original_path": os.path.join(output_dir, "layer15_logits_original.pt"),
        "logits_perturbed_path": os.path.join(output_dir, "layer15_logits_perturbed.pt")
    }
    
    # Update file paths
    if "file_paths" not in results_json:
        results_json["file_paths"] = {}
    
    results_json["file_paths"]["tyche_layer_specific_logits"] = {
        "original": os.path.join(output_dir, "layer15_logits_original.pt"),
        "perturbed": os.path.join(output_dir, "layer15_logits_perturbed.pt")
    }
    
    if "results" not in results_json["file_paths"]:
        results_json["file_paths"]["results"] = {}
    
    results_json["file_paths"]["results"].update({
        "layer_mults": os.path.join(output_dir, "layer15_mults.pt"),
        "layer_deltas": os.path.join(output_dir, "layer15_deltas.pt"),
        "layer_props": os.path.join(output_dir, "layer15_props.pt")
    })
    
    with open(result_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Layer-specific analysis complete!")
    print(f"Average layer KL: {avg_layer_kl}")
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tyche layer-specific analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--intervention_layer", type=int, default=15, help="Layer to intervene on")
    parser.add_argument("--cutoff", type=float, default=1e-2, help="KL divergence cutoff")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples (start with a small number)")
    
    args = parser.parse_args()
    
    run_layer_specific_analysis(
        args.model_path,
        args.output_dir,
        args.intervention_layer,
        args.cutoff,
        args.n_samples
    )
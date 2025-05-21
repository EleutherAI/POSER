# Save as run_layer_specific.py
import os
import gc
import json
import torch
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tyche import VolumeConfig, VolumeEstimator

# Force CUDA to release memory
torch.cuda.empty_cache()
gc.collect()

def get_filter_string(layer_idx):
    """Create a filter string for a specific layer."""
    return f"layers.{layer_idx}"

def main():
    parser = argparse.ArgumentParser(description='Run Layer-Specific Tyche Analysis')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--layer', type=int, default=15, help='Layer to target')
    parser.add_argument('--cutoff', type=float, default=1e-2, help='KL cutoff')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--val_size', type=int, default=10, help='Validation size')
    parser.add_argument('--iters', type=int, default=2, help='Iterations')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    # Load with half precision to reduce memory usage
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Set padding token if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    # Load prompts from existing file
    with open(os.path.join(args.output_dir, "sequence_input.json"), "r") as f:
        prompts = json.load(f)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": prompts[:args.n_samples]})
    
    # Create config with reduced memory usage and layer filter
    config = VolumeConfig(
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

    
    # Run estimator
    print(f"Running layer-specific analysis for layer {args.layer}")
    estimator = VolumeEstimator.from_config(config)
    result = estimator.run()
    
    # Save results
    print("Saving layer-specific results")
    torch.save(result.deltas, os.path.join(args.output_dir, f"layer{args.layer}_deltas.pt"))
    torch.save(result.mults, os.path.join(args.output_dir, f"layer{args.layer}_mults.pt"))
    torch.save(result.props, os.path.join(args.output_dir, f"layer{args.layer}_props.pt"))
    
    # Save logits
    torch.save(estimator.latest_original_logits, os.path.join(args.output_dir, f"layer{args.layer}_logits_original.pt"))
    layer_pert_logits = torch.cat(estimator.latest_perturbed_logits, dim=0)
    torch.save(layer_pert_logits, os.path.join(args.output_dir, f"layer{args.layer}_logits_perturbed.pt"))
    
    print(f"Layer {args.layer} analysis complete")

if __name__ == "__main__":
    main()
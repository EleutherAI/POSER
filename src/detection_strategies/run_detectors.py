#!/usr/bin/env python3
import argparse
import json
from typing import List, Dict, Any

from dependence_detector import get_base_parser, run_experiment
from grad_product_detector import GradProductDetector
from path_integral_detector import PathIntegralDetector

def main():
    """Run multiple detectors and compare results."""
    parser = get_base_parser()
    parser.add_argument("--detectors", type=str, nargs="+", default=["grad_product", "path_integral"],
                        choices=["grad_product", "path_integral"], 
                        help="Which detectors to run")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare results from different detectors")
    args = parser.parse_args()
    
    results = {}
    
    # Run selected detectors
    if "grad_product" in args.detectors:
        print("Running Gradient Product Detector...")
        args.output_prefix = "gradient_product_results"
        results["grad_product"] = run_experiment(GradProductDetector, args)
    
    if "path_integral" in args.detectors:
        print("Running Path Integral Detector...")
        args.output_prefix = "path_integral_results"
        results["path_integral"] = run_experiment(PathIntegralDetector, args)
    
    # Compare results if requested and if multiple detectors were run
    if args.compare and len(results) > 1:
        comparison = compare_detector_results(results)
        
        # Save comparison
        with open(f"detector_comparison_{args.model}.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Comparison saved to detector_comparison_{args.model}.json")


def compare_detector_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare results from different detectors.
    
    Args:
        results: Dictionary mapping detector names to their results
        
    Returns:
        Dictionary of comparison metrics
    """
    comparison = {
        "detectors": list(results.keys()),
        "metrics": {},
    }
    
    # Extract key metrics from each detector
    for detector, detector_result in results.items():
        dependence = detector_result.get("dependence", {})
        
        if detector == "grad_product":
            # For gradient product detector, extract similarity metrics
            comparison["metrics"][f"{detector}_avg_cosine_sim"] = dependence.get("avg_cosine_sim", 0)
            comparison["metrics"][f"{detector}_avg_dot_product"] = dependence.get("avg_dot_product", 0)
            comparison["metrics"][f"{detector}_avg_norm_dot_product"] = dependence.get("avg_norm_dot_product", 0)
        
        elif detector == "path_integral":
            # For path integral detector, extract path integral metrics
            comparison["metrics"][f"{detector}_path_integral"] = dependence.get("path_integral", 0)
            comparison["metrics"][f"{detector}_avg_cosine_sim"] = dependence.get("avg_cosine_sim", 0)
    
    # Add dataset info
    for detector, detector_result in results.items():
        comparison["dataset_path"] = detector_result.get("dataset_path", "")
        comparison["num_examples"] = detector_result.get("num_examples", 0)
        break  # Just need one detector's info
    
    return comparison


if __name__ == "__main__":
    main() 
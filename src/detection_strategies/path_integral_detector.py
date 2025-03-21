import torch
import argparse
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection_strategies.dependence_detector import run_experiment, get_base_parser
from detection_strategies.path_integral.detector import PathIntegralDetector

if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument("--eval_steps", type=int, default=100, 
                        help="Number of steps between evaluations")
    parser.add_argument("--num_bends", type=int, default=3, 
                        help="Number of bends in the polygonal chain")
    parser.add_argument("--path_optim_steps", type=int, default=20, 
                        help="Number of optimization steps for path optimization")
    parser.add_argument("--path_optim_lr", type=float, default=1e-4, 
                        help="Learning rate for path optimization")
    parser.add_argument("--prior_weight", type=float, default=0.1,
                        help="Weight for the prior (weight decay) in both training and path optimization")
    parser.add_argument("--retrain", action="store_true", 
                        help="Force retraining even if checkpoints exist")
    
    args = parser.parse_args()
    
    # Run the experiment with the Path Integral Detector
    run_experiment(PathIntegralDetector, args)

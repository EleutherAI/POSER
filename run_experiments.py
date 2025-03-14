import subprocess
import os
import time
import glob
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import pynvml

@dataclass
class GPUManager:
    def __init__(self, n_gpus: Optional[int] = None, min_free_memory_mb: int = 2000):
        # Initialize pynvml
        pynvml.nvmlInit()
        
        if n_gpus is None:
            # Try to detect available GPUs using nvidia-smi
            try:
                self.n_gpus = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                raise RuntimeError("Could not detect GPUs. Please specify --n-gpus")
        else:
            self.n_gpus = n_gpus
        
        self.min_free_memory_mb = min_free_memory_mb    
        self.gpu_processes: Dict[int, Optional[subprocess.Popen]] = {i: None for i in range(self.n_gpus)}
        
    def get_free_gpu(self) -> Optional[int]:
        for gpu_id, process in self.gpu_processes.items():
            # Check if process is done
            if process is not None and process.poll() is not None:
                self.gpu_processes[gpu_id] = None
                
            # Only consider GPUs with no active process
            if self.gpu_processes[gpu_id] is None:
                # Check if GPU has enough free memory
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    free_memory_mb = info.free / 1024 / 1024
                    
                    if free_memory_mb >= self.min_free_memory_mb:
                        return gpu_id
                except pynvml.NVMLError:
                    # Skip this GPU if there's an error getting memory info
                    continue
                    
        return None
    
    def assign_process(self, gpu_id: int, process: subprocess.Popen):
        self.gpu_processes[gpu_id] = process
        
    def any_running(self) -> bool:
        return any(p is not None and p.poll() is None for p in self.gpu_processes.values())
    
    def cleanup(self):
        for process in self.gpu_processes.values():
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        pynvml.nvmlShutdown()


def find_models(model_dir='./models/'):
    """Find all model directories in the specified path."""
    # Get all subdirectory names in the model directory
    model_paths = [d for d in glob.glob(os.path.join(model_dir, '*')) if os.path.isdir(d)]
    # Extract just the model names (not full paths)
    model_names = [os.path.basename(p) for p in model_paths]
    return model_names


def run_gradient_detector(model_name, gpu_id, use_flash_attn=False):
    """Run the gradient product detector for a specific model on a specific GPU."""
    cmd = [
        "python", 
        "src/detection_strategies/grad_product_detector.py",
        "--load_in_8bit",
        "--model", model_name
    ]
    
    if use_flash_attn:
        cmd.append("--use_flash_attn")
    
    # Set up environment with specified GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log file for this run
    log_file = log_dir / f"grad_product_{model_name}_gpu{gpu_id}.log"
    
    print(f"Starting gradient product detection for model: {model_name}")
    print(f"GPU: {gpu_id}")
    print(f"Log file: {log_file}")
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    return process


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run gradient product detection on multiple models")
    parser.add_argument("--n-gpus", type=int, default=None,
                      help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--check-interval", type=float, default=30,
                      help="Interval in seconds to check for completed processes")
    parser.add_argument("--model-dir", type=str, default="./models/",
                      help="Directory containing model folders")
    parser.add_argument("--use-flash-attn", action="store_true",
                      help="Use Flash Attention 2 if available")
    parser.add_argument("--models", nargs='+', default=None,
                      help="Specific model names to run (default: all models)")
    args = parser.parse_args()
    
    # Initialize GPU manager
    gpu_manager = GPUManager(args.n_gpus)
    print(f"Running gradient product detection using {gpu_manager.n_gpus} GPUs")
    
    # Find all models
    if args.models:
        models = args.models
    else:
        models = find_models(args.model_dir)
    
    print(f"Found {len(models)} models: {', '.join(models)}")
    
    # Process all models
    pending_models = models.copy()
    completed_models = []
    
    try:
        # Main processing loop
        while pending_models or gpu_manager.any_running():
            # Try to start new processes on free GPUs
            while pending_models:
                free_gpu = gpu_manager.get_free_gpu()
                if free_gpu is None:
                    # No free GPUs available
                    break
                
                # Get next model and run it on the free GPU
                model = pending_models.pop(0)
                process = run_gradient_detector(model, free_gpu, args.use_flash_attn)
                gpu_manager.assign_process(free_gpu, process)
            
            # Check for completed processes
            for gpu_id, process in gpu_manager.gpu_processes.items():
                if process is not None and process.poll() is not None:
                    # Process completed
                    return_code = process.returncode
                    model_name = None
                    
                    # Find which model this process was running
                    for cmd_arg in process.args:
                        if cmd_arg not in ["--model", "--load_in_8bit", "--use_flash_attn"]:
                            model_name = cmd_arg
                            break
                    
                    status = "SUCCESS" if return_code == 0 else f"FAILED (code {return_code})"
                    print(f"Completed processing model {model_name}: {status}")
                    completed_models.append(model_name)
                    
                    # Mark this GPU as free
                    gpu_manager.gpu_processes[gpu_id] = None
            
            # Wait before checking again
            time.sleep(args.check_interval)
    
    except KeyboardInterrupt:
        print("\nReceived interrupt, cleaning up...")
        gpu_manager.cleanup()
        # Print status of remaining models
        print(f"Completed {len(completed_models)} of {len(models)} models")
        print(f"Pending: {', '.join(pending_models)}")
    
    print("\nAll gradient product detections completed")
    

if __name__ == "__main__":
    main()
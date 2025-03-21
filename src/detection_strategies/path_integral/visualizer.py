import matplotlib.pyplot as plt
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional

def plot_path_losses(path_name: str, t_values: List[float], losses: List[float], 
                    save_dir: str, task_losses: Optional[List[float]] = None,
                    prior_losses: Optional[List[float]] = None):
    """
    Plot losses along a path.
    
    Args:
        path_name: Name of the path (used in plot title and filename)
        t_values: List of t values along the path
        losses: List of total loss values at each t
        save_dir: Directory to save the plot
        task_losses: Optional list of task loss values
        prior_losses: Optional list of prior loss values
    """
    plt.figure(figsize=(10, 6))
    
    # Plot total loss
    plt.plot(t_values, losses, 'b-', label='Total Loss')
    
    # Plot component losses if available
    if task_losses is not None:
        plt.plot(t_values, task_losses, 'r-', label='Task Loss')
    
    if prior_losses is not None:
        plt.plot(t_values, prior_losses, 'g-', label='Prior Loss')
    
    plt.xlabel('Path Position (t)')
    plt.ylabel('Loss')
    plt.title(f'Loss along Path: {path_name}')
    plt.legend()
    plt.grid(True)
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    safe_path_name = path_name.replace(' -> ', '_to_')
    safe_path_name = safe_path_name.replace('+', 'and')
    plt.savefig(f"{save_dir}/path_loss_{safe_path_name}.png")
    plt.close()

def plot_all_path_losses(loss_data: Dict[str, Dict[str, Any]], save_dir: str):
    """
    Create a combined plot with losses from all paths.
    
    Args:
        loss_data: Dictionary mapping path names to loss data dictionaries
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    markers = ['o', 's', '^', 'v', '<', '>', 'p']
    
    # Plot each path with a different color/marker
    for i, (path_name, data) in enumerate(loss_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        t_values = data["t_values"]
        losses = data["total_losses"]
        
        # Use solid line with sparse markers for better readability
        plt.plot(t_values, losses, color + '-', label=path_name)
        # Add markers at fewer points
        marker_indices = np.linspace(0, len(t_values)-1, min(10, len(t_values))).astype(int)
        plt.plot([t_values[i] for i in marker_indices], 
                [losses[i] for i in marker_indices], 
                color + marker, markersize=8)
    
    plt.xlabel('Path Position (t)')
    plt.ylabel('Loss')
    plt.title('Loss Comparison Across Different Paths')
    plt.legend()
    plt.grid(True)
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{save_dir}/path_loss_comparison.png")
    plt.close() 
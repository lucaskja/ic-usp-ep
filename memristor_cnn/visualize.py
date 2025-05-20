#!/usr/bin/env python
"""
Visualization script for Memristor-based CNN.

This script provides visualizations for:
1. Model architecture
2. Training history
3. Performance metrics
4. Energy efficiency and latency comparisons
5. Memristor programming statistics
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchviz import make_dot
import pandas as pd

from memristor_cnn.models import MemristorCNN
from memristor_cnn.utils.evaluation_utils import ModelEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize Memristor-based CNN')
    
    # Input parameters
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing results to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (defaults to results_dir/visualizations)')
    
    # Visualization options
    parser.add_argument('--visualize_model', action='store_true',
                        help='Visualize model architecture')
    parser.add_argument('--visualize_history', action='store_true',
                        help='Visualize training history')
    parser.add_argument('--visualize_metrics', action='store_true',
                        help='Visualize performance metrics')
    parser.add_argument('--visualize_all', action='store_true',
                        help='Visualize all aspects')
    
    # Model parameters (for architecture visualization)
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes for model creation')
    parser.add_argument('--width_mult', type=float, default=0.75,
                        help='Width multiplier for the network')
    
    return parser.parse_args()


def visualize_model_architecture(output_dir, num_classes=10, width_mult=0.75):
    """
    Visualize model architecture.
    
    Args:
        output_dir (str): Directory to save visualizations.
        num_classes (int): Number of classes for model creation.
        width_mult (float): Width multiplier for the network.
    """
    print("Visualizing model architecture...")
    
    # Create model
    model = MemristorCNN(num_classes=num_classes, width_mult=width_mult)
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    
    # Generate graph
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    
    # Save graph
    dot.format = 'png'
    dot.render(os.path.join(output_dir, 'model_architecture'))
    
    # Save model summary as text
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write(f"Model: MemristorCNN\n")
        f.write(f"Width Multiplier: {width_mult}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        
        # Write layer information
        f.write("Layer Structure:\n")
        for name, module in model.named_children():
            f.write(f"- {name}: {module.__class__.__name__}\n")
            
            # If it's a sequential container, list its components
            if isinstance(module, torch.nn.Sequential):
                for i, layer in enumerate(module):
                    f.write(f"  - {i}: {layer.__class__.__name__}\n")


def visualize_training_history(results_dir, output_dir):
    """
    Visualize training history.
    
    Args:
        results_dir (str): Directory containing results.
        output_dir (str): Directory to save visualizations.
    """
    print("Visualizing training history...")
    
    # Check if history files exist
    ex_situ_path = os.path.join(results_dir, 'ex_situ_history.json')
    in_situ_path = os.path.join(results_dir, 'in_situ_history.json')
    
    if not os.path.exists(ex_situ_path):
        print(f"Warning: Ex-situ history file not found at {ex_situ_path}")
        return
    
    # Load history data
    with open(ex_situ_path, 'r') as f:
        ex_situ_history = json.load(f)
    
    in_situ_history = None
    if os.path.exists(in_situ_path):
        with open(in_situ_path, 'r') as f:
            in_situ_history = json.load(f)
    
    # Create evaluator for plotting
    evaluator = ModelEvaluator(model=None, results_dir=output_dir)
    
    # Plot training history
    evaluator.plot_training_history(ex_situ_history, in_situ_history)


def visualize_performance_metrics(results_dir, output_dir):
    """
    Visualize performance metrics.
    
    Args:
        results_dir (str): Directory containing results.
        output_dir (str): Directory to save visualizations.
    """
    print("Visualizing performance metrics...")
    
    # Check if metrics files exist
    metrics_path = os.path.join(results_dir, 'metrics.json')
    energy_path = os.path.join(results_dir, 'energy_metrics.json')
    latency_path = os.path.join(results_dir, 'latency_metrics.json')
    combined_path = os.path.join(results_dir, 'combined_metrics.json')
    
    # Load available metrics
    metrics = {}
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics.update(json.load(f))
    
    if os.path.exists(energy_path):
        with open(energy_path, 'r') as f:
            metrics.update(json.load(f))
    
    if os.path.exists(latency_path):
        with open(latency_path, 'r') as f:
            metrics.update(json.load(f))
    
    if os.path.exists(combined_path):
        with open(combined_path, 'r') as f:
            metrics.update(json.load(f))
    
    if not metrics:
        print("Warning: No metrics files found")
        return
    
    # Create summary visualization
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid for key metrics
    plt.subplot(2, 2, 1)
    if 'accuracy' in metrics:
        plt.bar(['Accuracy'], [metrics['accuracy']], color='blue')
        plt.title('Test Accuracy (%)')
        plt.ylim(0, 100)
        plt.text(0, metrics['accuracy']/2, f"{metrics['accuracy']:.2f}%", 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.subplot(2, 2, 2)
    if 'energy_efficiency' in metrics:
        plt.bar(['Energy Efficiency'], [metrics['energy_efficiency']], color='green')
        plt.title('Energy Efficiency vs GPU (x)')
        plt.text(0, metrics['energy_efficiency']/2, f"{metrics['energy_efficiency']:.2f}x", 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.subplot(2, 2, 3)
    if 'latency_reduction' in metrics:
        plt.bar(['Latency Reduction'], [metrics['latency_reduction']], color='orange')
        plt.title('Latency Reduction vs GPU (x)')
        plt.text(0, metrics['latency_reduction']/2, f"{metrics['latency_reduction']:.2f}x", 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.subplot(2, 2, 4)
    if 'inference_time_ms' in metrics:
        plt.bar(['Inference Time'], [metrics['inference_time_ms']], color='purple')
        plt.title('Inference Time (ms)')
        plt.text(0, metrics['inference_time_ms']/2, f"{metrics['inference_time_ms']:.2f} ms", 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300)
    plt.close()
    
    # Visualize transfer statistics if available
    transfer_path = os.path.join(results_dir, 'transfer_stats.json')
    if os.path.exists(transfer_path):
        with open(transfer_path, 'r') as f:
            transfer_stats = json.load(f)
        
        if 'layers' in transfer_stats:
            # Extract layer names and programming accuracy
            layer_names = list(transfer_stats['layers'].keys())
            accuracies = [transfer_stats['layers'][layer]['programming_accuracy'] for layer in layer_names]
            times = [transfer_stats['layers'][layer]['programming_time_ms'] for layer in layer_names]
            
            # Shorten layer names for better visualization
            short_names = [name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name 
                          for name in layer_names]
            
            # Plot programming accuracy by layer
            plt.figure(figsize=(12, 6))
            bars = plt.bar(short_names, accuracies, color='skyblue')
            plt.axhline(y=transfer_stats['average_accuracy'], color='r', linestyle='-', 
                       label=f"Average: {transfer_stats['average_accuracy']:.2f}%")
            
            plt.title('Memristor Programming Accuracy by Layer')
            plt.xlabel('Layer')
            plt.ylabel('Programming Accuracy (%)')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'programming_accuracy.png'), dpi=300)
            plt.close()
            
            # Plot programming time by layer
            plt.figure(figsize=(12, 6))
            plt.bar(short_names, times, color='lightgreen')
            plt.title('Memristor Programming Time by Layer')
            plt.xlabel('Layer')
            plt.ylabel('Programming Time (ms)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'programming_time.png'), dpi=300)
            plt.close()


def main():
    """Main visualization function."""
    # Parse arguments
    args = parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine what to visualize
    visualize_model = args.visualize_model or args.visualize_all
    visualize_history = args.visualize_history or args.visualize_all
    visualize_metrics = args.visualize_metrics or args.visualize_all
    
    # If no specific visualization is requested, visualize all
    if not (visualize_model or visualize_history or visualize_metrics):
        visualize_model = visualize_history = visualize_metrics = True
    
    # Perform visualizations
    if visualize_model:
        visualize_model_architecture(output_dir, args.num_classes, args.width_mult)
    
    if visualize_history:
        visualize_training_history(args.results_dir, output_dir)
    
    if visualize_metrics:
        visualize_performance_metrics(args.results_dir, output_dir)
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

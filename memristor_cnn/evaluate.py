#!/usr/bin/env python
"""
Evaluation script for Memristor-based CNN.

This script evaluates a trained Memristor-based CNN model on a test dataset
and analyzes its performance, energy efficiency, and latency.
"""

import os
import argparse
import torch
import json

from memristor_cnn.models import MemristorCNN
from memristor_cnn.utils.data_utils import get_leaf_disease_dataloaders, get_class_names
from memristor_cnn.utils.evaluation_utils import ModelEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Memristor-based CNN')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='datasets/leaf_disease',
                        help='Path to the dataset directory')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes in the dataset (if not auto-detected)')
    parser.add_argument('--width_mult', type=float, default=0.75,
                        help='Width multiplier for the network')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--results_dir', type=str, default='results/memristor_cnn_eval',
                        help='Directory to save evaluation results')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--parallel_arrays', type=int, default=3,
                        help='Number of parallel arrays for latency analysis')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.results_dir, 'eval_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load dataset
    print("Loading dataset...")
    _, _, test_loader = get_leaf_disease_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        enhanced_augmentation=False  # No augmentation for evaluation
    )
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    num_classes = len(class_names) if args.num_classes is None else args.num_classes
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create model
    print("Creating model...")
    model = MemristorCNN(
        num_classes=num_classes,
        width_mult=args.width_mult
    )
    
    # Setup memristor mapping
    model.setup_memristor_mapping(device=device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        results_dir=args.results_dir
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    metrics = evaluator.evaluate(test_loader, class_names=class_names)
    
    # Analyze energy efficiency and latency
    energy_metrics = evaluator.analyze_energy_efficiency(
        input_size=3*224*224,  # RGB image
        batch_size=args.batch_size
    )
    
    latency_metrics = evaluator.analyze_latency(
        input_size=3*224*224,  # RGB image
        batch_size=args.batch_size,
        parallel_arrays=args.parallel_arrays
    )
    
    # Print final results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Energy Efficiency: {energy_metrics['efficiency_ratio']:.2f}x vs GPU")
    print(f"Latency Reduction: {latency_metrics['latency_reduction']:.2f}x vs GPU")
    print("="*50)
    
    # Save combined metrics
    combined_metrics = {
        'accuracy': metrics['accuracy'],
        'energy_efficiency': energy_metrics['efficiency_ratio'],
        'latency_reduction': latency_metrics['latency_reduction'],
        'inference_time_ms': metrics['inference_time_ms']
    }
    
    with open(os.path.join(args.results_dir, 'combined_metrics.json'), 'w') as f:
        json.dump(combined_metrics, f, indent=4)


if __name__ == '__main__':
    main()

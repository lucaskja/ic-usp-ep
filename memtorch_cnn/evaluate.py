#!/usr/bin/env python3
"""
Evaluation script for MemTorch-based CNN.
"""

import argparse
import os
import torch
import json

from memtorch_cnn.models import MemTorchCNN
from memtorch_cnn.utils.data_utils import get_leaf_disease_dataloaders, get_class_names
from memtorch_cnn.utils.evaluation_utils import ModelEvaluator


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate MemTorch-based CNN for leaf disease classification')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='datasets/leaf_disease',
                        help='Path to the dataset directory')
    
    # Model arguments
    parser.add_argument('--width_mult', type=float, default=0.75,
                        help='Width multiplier for the network')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the checkpoint to evaluate')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--results_dir', type=str, default='results/memtorch_cnn_eval',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    _, _, test_loader = get_leaf_disease_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create model
    print("Creating model...")
    model = MemTorchCNN(
        num_classes=num_classes,
        width_mult=args.width_mult
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Check if model is memristive
    if hasattr(model, 'is_memristive') and model.is_memristive:
        print("Model is memristive.")
    else:
        print("Model is not memristive. Converting to memristive...")
        model.convert_to_memristive(device=device)
    
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
    
    test_metrics = evaluator.evaluate(test_loader, class_names)
    
    # Analyze energy efficiency
    energy_metrics = evaluator.analyze_energy_efficiency()
    
    # Analyze latency
    latency_metrics = evaluator.analyze_latency()
    
    # Save configuration
    config = vars(args)
    config['num_classes'] = num_classes
    config['class_names'] = class_names
    
    with open(os.path.join(args.results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "="*50)
    print("Evaluation Complete")
    print("="*50)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Energy Efficiency: {energy_metrics['efficiency_ratio']:.2f}x")
    print(f"Latency Reduction: {latency_metrics['latency_reduction']:.2f}x")
    print(f"Results saved to: {args.results_dir}")


if __name__ == '__main__':
    main()

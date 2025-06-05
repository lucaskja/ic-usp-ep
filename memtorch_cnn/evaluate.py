#!/usr/bin/env python3
"""
Evaluation script for MemTorch-based CNN.
"""

import argparse
import torch
import os
from memtorch_cnn.models import MemTorchCNN
from memtorch_cnn.utils.data_utils import get_leaf_disease_dataloaders, get_class_names
from memtorch_cnn.utils.evaluation_utils import ModelEvaluator

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate MemTorch-based CNN for leaf disease classification')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='datasets/leaf_disease',
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default='checkpoints/memtorch_cnn/model_best_ex_situ.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for evaluation (cpu or cuda)')
    
    # Evaluation arguments
    parser.add_argument('--results_dir', type=str, default='results/memtorch_cnn',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    _, _, test_loader = get_leaf_disease_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    class_names = get_class_names(args.data_dir)
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Load model
    print("Loading model from checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = MemTorchCNN(num_classes=len(class_names))
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    
    # Create evaluator
    os.makedirs(args.results_dir, exist_ok=True)
    evaluator = ModelEvaluator(model, device, results_dir=args.results_dir)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluator.evaluate(test_loader, class_names)
    
    # Analyze energy efficiency and latency
    print("\nAnalyzing energy efficiency...")
    energy_metrics = evaluator.analyze_energy_efficiency()
    
    print("\nAnalyzing latency...")
    latency_metrics = evaluator.analyze_latency()
    
    # Print summary metrics
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Energy Efficiency: {energy_metrics['efficiency_ratio']:.2f}x")
    print(f"Latency Reduction: {latency_metrics['latency_reduction']:.2f}x")
    print(f"Results saved to: {args.results_dir}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()

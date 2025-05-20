#!/usr/bin/env python
"""
Training script for Memristor-based CNN.

This script implements the hybrid training approach for the Memristor-based CNN:
1. Ex-situ training: Conventional training on GPU/CPU
2. Weight transfer to memristor arrays
3. In-situ training: Threshold-based updates for FC layer only
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import time

from memristor_cnn.models import MemristorCNN
from memristor_cnn.utils.data_utils import get_leaf_disease_dataloaders, get_class_names
from memristor_cnn.utils.training_utils import HybridTrainer
from memristor_cnn.utils.evaluation_utils import ModelEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Memristor-based CNN')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='datasets/leaf_disease',
                        help='Path to the dataset directory')
    parser.add_argument('--enhanced_augmentation', action='store_true',
                        help='Use enhanced data augmentation')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--width_mult', type=float, default=0.75,
                        help='Width multiplier for the network')
    
    # Training parameters
    parser.add_argument('--ex_situ_epochs', type=int, default=50,
                        help='Number of epochs for ex-situ training')
    parser.add_argument('--in_situ_epochs', type=int, default=10,
                        help='Number of epochs for in-situ training')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold for in-situ weight updates')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/memristor_cnn',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results/memristor_cnn',
                        help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--parallel_arrays', type=int, default=3,
                        help='Number of parallel arrays for latency analysis')
    
    # Training mode
    parser.add_argument('--skip_ex_situ', action='store_true',
                        help='Skip ex-situ training phase')
    parser.add_argument('--skip_in_situ', action='store_true',
                        help='Skip in-situ training phase')
    parser.add_argument('--closed_loop', action='store_true',
                        help='Use closed-loop programming for weight transfer')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create checkpoint and results directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.results_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_leaf_disease_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        enhanced_augmentation=args.enhanced_augmentation
    )
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create model
    print("Creating model...")
    model = MemristorCNN(
        num_classes=num_classes,
        width_mult=args.width_mult
    )
    
    # Setup memristor mapping
    model.setup_memristor_mapping(device=device)
    
    # Create optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Create trainer
    trainer = HybridTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Phase 1: Ex-situ training
    ex_situ_history = None
    if not args.skip_ex_situ:
        print("\n" + "="*50)
        print("Phase 1: Ex-situ Training")
        print("="*50)
        
        start_time = time.time()
        ex_situ_history = trainer.ex_situ_train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.ex_situ_epochs,
            scheduler=scheduler
        )
        ex_situ_time = time.time() - start_time
        
        print(f"Ex-situ training completed in {ex_situ_time:.2f} seconds")
    
    # Weight transfer to memristor arrays
    if not args.skip_ex_situ and not args.skip_in_situ:
        print("\n" + "="*50)
        print("Weight Transfer to Memristor Arrays")
        print("="*50)
        
        transfer_stats = trainer.transfer_to_memristor(closed_loop=args.closed_loop)
    
    # Phase 2: In-situ training
    in_situ_history = None
    if not args.skip_in_situ:
        print("\n" + "="*50)
        print("Phase 2: In-situ Training")
        print("="*50)
        
        start_time = time.time()
        in_situ_history = trainer.in_situ_train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.in_situ_epochs,
            learning_rate=args.lr * 0.1,  # Lower learning rate for in-situ phase
            threshold=args.threshold
        )
        in_situ_time = time.time() - start_time
        
        print(f"In-situ training completed in {in_situ_time:.2f} seconds")
    
    # Evaluate the model
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        results_dir=args.results_dir
    )
    
    # Evaluate on test set
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
    
    # Plot training history
    if ex_situ_history:
        evaluator.plot_training_history(ex_situ_history, in_situ_history)
    
    # Print final results
    print("\n" + "="*50)
    print("Final Results")
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

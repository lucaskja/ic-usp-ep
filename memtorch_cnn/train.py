#!/usr/bin/env python3
"""
Training script for MemTorch-based CNN.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import json

from memtorch_cnn.models import MemTorchCNN
from memtorch_cnn.utils.trainer import HybridTrainer
from memtorch_cnn.utils.data_utils import get_leaf_disease_dataloaders, get_class_names
from memtorch_cnn.utils.evaluation_utils import ModelEvaluator


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train MemTorch-based CNN for leaf disease classification')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='datasets/leaf_disease',
                        help='Path to the dataset directory')
    parser.add_argument('--enhanced_augmentation', action='store_true',
                        help='Use enhanced data augmentation')
    
    # Model arguments
    parser.add_argument('--width_mult', type=float, default=0.75,
                        help='Width multiplier for the network')
    parser.add_argument('--tile_shape', type=int, nargs=2, default=[128, 128],
                        help='Shape of memristor crossbar tiles')
    parser.add_argument('--adc_resolution', type=int, default=8,
                        help='ADC resolution in bits')
    parser.add_argument('--dac_resolution', type=int, default=8,
                        help='DAC resolution in bits')
    parser.add_argument('--max_input_voltage', type=float, default=0.3,
                        help='Maximum input voltage')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--ex_situ_epochs', type=int, default=50,
                        help='Number of ex-situ training epochs')
    parser.add_argument('--in_situ_epochs', type=int, default=10,
                        help='Number of in-situ training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold for in-situ weight updates')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/memtorch_cnn',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results/memtorch_cnn',
                        help='Directory to save results')
    parser.add_argument('--skip_ex_situ', action='store_true',
                        help='Skip ex-situ training phase')
    parser.add_argument('--skip_in_situ', action='store_true',
                        help='Skip in-situ training phase')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (reduced dataset size)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create checkpoint and results directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_leaf_disease_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        enhanced_augmentation=args.enhanced_augmentation
    )
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Debug mode: reduce dataset size
    if args.debug:
        print("Debug mode enabled: reducing dataset size")
        train_loader.dataset = torch.utils.data.Subset(train_loader.dataset, range(min(100, len(train_loader.dataset))))
        val_loader.dataset = torch.utils.data.Subset(val_loader.dataset, range(min(50, len(val_loader.dataset))))
        test_loader.dataset = torch.utils.data.Subset(test_loader.dataset, range(min(50, len(test_loader.dataset))))
    
    # Create model
    print("Creating model...")
    model = MemTorchCNN(
        num_classes=num_classes,
        width_mult=args.width_mult
    )
    model = model.to(device)
    
    # Create optimizer
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
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
    
    # Convert model to memristive
    print("\n" + "="*50)
    print("Converting Model to Memristive")
    print("="*50)
    
    model.convert_to_memristive(
        tile_shape=tuple(args.tile_shape),
        adc_resolution=args.adc_resolution,
        dac_resolution=args.dac_resolution,
        max_input_voltage=args.max_input_voltage,
        device=device
    )
    
    # Weight transfer to memristor arrays
    if not args.skip_ex_situ and not args.skip_in_situ:
        print("\n" + "="*50)
        print("Weight Transfer to Memristor Arrays")
        print("="*50)
        
        transfer_stats = trainer.transfer_to_memristor(bits=4, non_idealities=True)
    
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
            learning_rate=args.lr * 0.1,
            threshold=args.threshold
        )
        in_situ_time = time.time() - start_time
        
        print(f"In-situ training completed in {in_situ_time:.2f} seconds")
    
    # Evaluate model
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        results_dir=args.results_dir
    )
    
    # Evaluate on test set
    test_metrics = evaluator.evaluate(test_loader, class_names)
    
    # Analyze energy efficiency
    energy_metrics = evaluator.analyze_energy_efficiency()
    
    # Analyze latency
    latency_metrics = evaluator.analyze_latency()
    
    # Plot training history
    evaluator.plot_training_history(ex_situ_history, in_situ_history)
    
    # Save configuration
    config = vars(args)
    config['num_classes'] = num_classes
    config['class_names'] = class_names
    
    with open(os.path.join(args.results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "="*50)
    print("Training and Evaluation Complete")
    print("="*50)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Energy Efficiency: {energy_metrics['efficiency_ratio']:.2f}x")
    print(f"Latency Reduction: {latency_metrics['latency_reduction']:.2f}x")
    print(f"Results saved to: {args.results_dir}")


if __name__ == '__main__':
    main()

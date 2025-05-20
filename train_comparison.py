"""
Script to compare different MobileNetV2 variants with and without parameter reduction.

This script trains and evaluates the following models:
1. Base MobileNetV2 without parameter reduction (width_mult=1.0)
2. Base MobileNetV2 with parameter reduction (width_mult=0.75)
3. MobileNetV2 with Mish and parameter reduction (width_mult=0.75)
4. MobileNetV2 with Mish, Triplet Attention and parameter reduction (width_mult=0.75)
5. MobileNetV2 with Mish, Triplet Attention, CNSN and parameter reduction (width_mult=0.75)

Results are saved to a CSV file with detailed metrics.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import logging
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utilities
from utils.data_utils import load_dataset
from utils.enhanced_data_utils import load_enhanced_dataset
from utils.training_utils import train_one_epoch, validate, save_checkpoint, setup_logging, EarlyStopping
from utils.model_utils import get_model_size, count_parameters, print_model_summary
from utils.evaluator import ModelEvaluator
from configs.model_configs import TRAIN_CONFIG, DATA_CONFIG

# Import model creation functions
from base_mobilenetv2.models.mobilenetv2 import create_mobilenetv2
from stage1_mish.models.mobilenetv2_mish import create_mobilenetv2_mish
from stage2_triplet.models.mobilenetv2_triplet_fixed import create_mobilenetv2_triplet
from stage3_cnsn.models.mobilenetv2_cnsn_fixed import create_mobilenetv2_cnsn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare different MobileNetV2 variants')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Patience for early stopping (0 to disable)')
    
    # Data parameters
    parser.add_argument('--enhanced_augmentation', action='store_true',
                        help='Use enhanced data augmentation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='experiments/comparison/pv',
                        help='Directory to save results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/comparison/pv',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with reduced dataset size')
    
    return parser.parse_args()


def train_and_evaluate_model(model_name, model_creator, num_classes, train_loader, val_loader, test_loader, 
                            args, device, checkpoint_dir, output_dir, width_mult=None):
    """
    Train and evaluate a model.
    
    Args:
        model_name (str): Name of the model
        model_creator (function): Function to create the model
        num_classes (int): Number of classes
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        args (Namespace): Command line arguments
        device (torch.device): Device to use
        checkpoint_dir (str): Directory to save checkpoints
        output_dir (str): Directory to save results
        width_mult (float, optional): Width multiplier for the model
        
    Returns:
        dict: Dictionary with model metrics
    """
    start_time = time.time()
    
    # Create model
    if width_mult is not None:
        model = model_creator(num_classes=num_classes, pretrained=False, width_mult=width_mult)
    else:
        model = model_creator(num_classes=num_classes, pretrained=False)
    
    model = model.to(device)
    
    # Get model info
    model_size_mb = get_model_size(model)
    num_params = count_parameters(model)
    
    logging.info(f"\n{'='*80}\nTraining {model_name}\n{'='*80}")
    logging.info(f"Model size: {model_size_mb:.2f} MB")
    logging.info(f"Number of parameters: {num_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    # Create early stopping object
    early_stopping = EarlyStopping(patience=args.early_stopping) if args.early_stopping > 0 else None
    
    # Initialize tracking variables
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    
    # Create model directory
    model_checkpoint_dir = os.path.join(checkpoint_dir, model_name.replace(" ", "_").lower())
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.epochs
        )
        
        # Evaluate on validation set
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Store results
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['acc1'])
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['acc1'])
        
        # Check if this is the best model so far
        is_best = val_metrics['acc1'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['acc1']
            best_epoch = epoch
        
        # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir=model_checkpoint_dir,
            filename=f'checkpoint_epoch_{epoch+1}.pth'
        )
        
        # Check early stopping
        if early_stopping and early_stopping(val_metrics['acc1']):
            logging.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds. Best validation accuracy: {best_val_acc:.5f}%")
    
    # Load best model for evaluation
    best_model_path = os.path.join(model_checkpoint_dir, 'best.pth')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    
    # Evaluate on test set
    logging.info(f"Evaluating {model_name} on test set...")
    evaluator = ModelEvaluator(model, test_loader, device)
    test_results = evaluator.evaluate()
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    evaluator.plot_confusion_matrix(test_results, save_path=cm_path)
    
    # Calculate inference time
    inference_start = time.time()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    inference_time = (time.time() - inference_start) / len(test_loader.dataset)
    
    # Compile results
    results = {
        'model_name': model_name,
        'width_mult': width_mult if width_mult is not None else 1.0,
        'model_size_mb': model_size_mb,
        'num_params': num_params,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_results['accuracy'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'test_f1': test_results['f1'],
        'training_time': training_time,
        'inference_time_ms': inference_time * 1000,  # Convert to milliseconds
        'checkpoint_path': best_model_path
    }
    
    return results


def main():
    """Main function to run the comparison."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set up logging
    setup_logging("model_comparison", args.log_dir)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    if args.enhanced_augmentation:
        logging.info("Using enhanced data augmentation")
        train_loader, val_loader, test_loader, num_classes = load_enhanced_dataset(
            args.data_dir,
            img_size=224,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            debug=args.debug
        )
    else:
        logging.info("Using standard data augmentation")
        train_loader, val_loader, test_loader, num_classes = load_dataset(
            args.data_dir,
            img_size=224,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            debug=args.debug
        )
    
    logging.info(f"Dataset loaded with {num_classes} classes")
    
    # Define models to compare
    models_to_compare = [
        {
            'name': 'Base MobileNetV2 (width_mult=1.0)',
            'creator': create_mobilenetv2,
            'width_mult': 1.0
        },
        {
            'name': 'Base MobileNetV2 (width_mult=0.75)',
            'creator': create_mobilenetv2,
            'width_mult': 0.75
        },
        {
            'name': 'MobileNetV2 with Mish (width_mult=0.75)',
            'creator': create_mobilenetv2_mish,
            'width_mult': 0.75
        },
        {
            'name': 'MobileNetV2 with Mish and Triplet Attention (width_mult=0.75)',
            'creator': create_mobilenetv2_triplet,
            'width_mult': 0.75
        },
        {
            'name': 'MobileNetV2 with Mish, Triplet Attention, and CNSN (width_mult=0.75)',
            'creator': create_mobilenetv2_cnsn,
            'width_mult': 0.75
        }
    ]
    
    # Train and evaluate each model
    results = []
    for model_config in models_to_compare:
        model_results = train_and_evaluate_model(
            model_name=model_config['name'],
            model_creator=model_config['creator'],
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            width_mult=model_config['width_mult']
        )
        results.append(model_results)
    
    # Print results summary
    logging.info("\n" + "="*120)
    logging.info("COMPARISON RESULTS SUMMARY")
    logging.info("="*120)
    logging.info(f"{'Model':<50} | {'Size (MB)':<10} | {'Params':<12} | {'Val Acc (%)':<12} | {'Test Acc (%)':<12} | {'Inf Time (ms)':<14}")
    logging.info("-"*120)
    
    for result in results:
        logging.info(
            f"{result['model_name']:<50} | "
            f"{result['model_size_mb']:<10.2f} | "
            f"{result['num_params']:<12,} | "
            f"{result['best_val_acc']:<12.5f} | "
            f"{result['test_acc']:<12.5f} | "
            f"{result['inference_time_ms']:<14.3f}"
        )
    
    # Save results to CSV
    csv_path = os.path.join(args.output_dir, 'model_comparison_results.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'model_name', 'width_mult', 'model_size_mb', 'num_params', 
            'best_val_acc', 'best_epoch', 'test_acc', 'test_precision', 
            'test_recall', 'test_f1', 'training_time', 'inference_time_ms', 
            'checkpoint_path'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    logging.info(f"Detailed comparison results saved to {csv_path}")


if __name__ == "__main__":
    main()

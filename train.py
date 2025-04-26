"""
Unified training script for all MobileNetV2 variants.

This script consolidates the training process for all model variants:
- Base MobileNetV2
- MobileNetV2 with Mish activation
- MobileNetV2 with Mish and Triplet Attention
- MobileNetV2 with Mish, Triplet Attention, and CNSN

It supports both standard and enhanced data augmentation techniques.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model variants
from base_mobilenetv2.models.mobilenetv2 import create_mobilenetv2
from stage1_mish.models.mobilenetv2_mish import create_mobilenetv2_mish
from stage2_triplet.models.mobilenetv2_triplet import create_mobilenetv2_triplet
from stage3_cnsn.models.mobilenetv2_cnsn import create_mobilenetv2_cnsn

# Import utilities
from utils.data_utils import load_dataset
from utils.enhanced_data_utils import load_enhanced_dataset
from utils.training_utils import train_one_epoch, validate, save_checkpoint, setup_logging, EarlyStopping
from utils.model_utils import get_model_size, print_model_summary
from base_mobilenetv2.configs.default_config import TRAIN_CONFIG, DATA_CONFIG
from base_mobilenetv2.evaluation.evaluator import MobileNetV2Evaluator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MobileNetV2 variants for leaf disease classification')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    
    # Model selection
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'mish', 'triplet', 'cnsn'],
                        help='Model type to train')
    parser.add_argument('--train_all', action='store_true',
                        help='Train all models sequentially (base -> mish -> triplet -> cnsn) and evaluate after training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--momentum', type=float, default=None,
                        help='SGD momentum (default: from config)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (default: from config)')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler (default: from config)')
    parser.add_argument('--lr_step_size', type=int, default=None,
                        help='Step size for StepLR scheduler (default: from config)')
    parser.add_argument('--lr_gamma', type=float, default=None,
                        help='Gamma for StepLR scheduler (default: from config)')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Patience for early stopping (0 to disable)')
    
    # Data parameters
    parser.add_argument('--enhanced_augmentation', action='store_true',
                        help='Use enhanced data augmentation')
    parser.add_argument('--val_split', type=float, default=None,
                        help='Validation split ratio (default: from config)')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size (default: from config)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of workers for data loading (default: from config)')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: experiments/{model_type})')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints (default: checkpoints/{model_type})')
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def create_model(model_type, num_classes, pretrained=True):
    """
    Create model based on type.
    
    Args:
        model_type (str): Type of model to create ('base', 'mish', 'triplet', 'cnsn')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: The created model
    """
    if model_type == 'base':
        model = create_mobilenetv2(num_classes, pretrained)
        logging.info("Base MobileNetV2 model created")
    elif model_type == 'mish':
        model = create_mobilenetv2_mish(num_classes, pretrained)
        logging.info("MobileNetV2 with Mish activation model created")
    elif model_type == 'triplet':
        model = create_mobilenetv2_triplet(num_classes, pretrained)
        logging.info("MobileNetV2 with Mish and Triplet Attention model created")
    elif model_type == 'cnsn':
        model = create_mobilenetv2_cnsn(num_classes, pretrained)
        logging.info("MobileNetV2 with Mish, Triplet Attention, and CNSN model created")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def get_data_config(args):
    """
    Get data configuration with command line overrides.
    
    Args:
        args (Namespace): Command line arguments
        
    Returns:
        dict: Data configuration
    """
    data_config = DATA_CONFIG.copy()
    
    # Override with command line arguments if provided
    if args.img_size is not None:
        data_config['img_size'] = args.img_size
    if args.batch_size is not None:
        data_config['batch_size'] = args.batch_size
    if args.val_split is not None:
        data_config['val_split'] = args.val_split
    if args.num_workers is not None:
        data_config['num_workers'] = args.num_workers
        
    return data_config

def get_train_config(args):
    """
    Get training configuration with command line overrides.
    
    Args:
        args (Namespace): Command line arguments
        
    Returns:
        dict: Training configuration
    """
    train_config = TRAIN_CONFIG.copy()
    
    # Override with command line arguments if provided
    if args.epochs is not None:
        train_config['epochs'] = args.epochs
    if args.lr is not None:
        train_config['lr'] = args.lr
    if args.momentum is not None:
        train_config['momentum'] = args.momentum
    if args.weight_decay is not None:
        train_config['weight_decay'] = args.weight_decay
    if args.lr_scheduler is not None:
        train_config['lr_scheduler'] = args.lr_scheduler
    if args.lr_step_size is not None:
        train_config['lr_step_size'] = args.lr_step_size
    if args.lr_gamma is not None:
        train_config['lr_gamma'] = args.lr_gamma
    
    # Add early stopping to config
    train_config['early_stopping_patience'] = args.early_stopping
        
    return train_config

def get_output_dirs(args, model_type=None):
    """
    Get output directories with defaults based on model type.
    
    Args:
        args (Namespace): Command line arguments
        model_type (str, optional): Override model type from args
        
    Returns:
        tuple: (output_dir, checkpoint_dir)
    """
    # Use provided model_type if given, otherwise use from args
    model_type = model_type or args.model_type
    
    # Set default output directory if not provided
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join('experiments', model_type)
    elif args.train_all:
        # If training all models, create model-specific subdirectories
        output_dir = os.path.join(output_dir, model_type)
    
    # Set default checkpoint directory if not provided
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join('checkpoints', model_type)
    elif args.train_all:
        # If training all models, create model-specific subdirectories
        checkpoint_dir = os.path.join(checkpoint_dir, model_type)
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return output_dir, checkpoint_dir

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, output_path):
    """
    Plot and save training curves.
    
    Args:
        train_losses (list): Training losses
        train_accs (list): Training accuracies
        val_losses (list): Validation losses
        val_accs (list): Validation accuracies
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Training curves saved to {output_path}")

def train_model(args, model_type, train_loader, val_loader, test_loader, num_classes, device):
    """
    Train a single model.
    
    Args:
        args (Namespace): Command line arguments
        model_type (str): Type of model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        num_classes (int): Number of classes
        device (torch.device): Device to use
        
    Returns:
        tuple: (best_model_path, best_val_acc, test_results)
    """
    # Get output directories
    output_dir, checkpoint_dir = get_output_dirs(args, model_type)
    
    # Get training configuration
    train_config = get_train_config(args)
    
    # Create model
    model = create_model(model_type, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    # Print model summary
    model_size_mb = get_model_size(model)
    print_model_summary(model)
    logging.info(f"Model Size: {model_size_mb:.2f} MB")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_config['lr'],
        momentum=train_config['momentum'],
        weight_decay=train_config['weight_decay']
    )
    
    # Define learning rate scheduler
    if train_config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config['lr_step_size'],
            gamma=train_config['lr_gamma']
        )
    elif train_config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['epochs']
        )
    else:
        scheduler = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('best_acc', 0.0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.error(f"No checkpoint found at '{args.resume}'")
    
    # Create early stopping object
    early_stopping = None
    if args.train_all:
        # Disable early stopping when training all models
        logging.info("Early stopping disabled for sequential training mode")
    elif train_config['early_stopping_patience'] > 0:
        early_stopping = EarlyStopping(patience=train_config['early_stopping_patience'])
        logging.info(f"Early stopping enabled with patience {train_config['early_stopping_patience']}")
    
    # Initialize tracking variables
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training loop
    logging.info(f"Starting training for {train_config['epochs']} epochs")
    
    for epoch in range(start_epoch, train_config['epochs']):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            train_config['epochs']
        )
        
        # Evaluate on validation set
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Learning rate: {current_lr:.6f}")
        
        # Store results
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['acc1'])
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['acc1'])
        
        # Check if this is the best model so far
        is_best = val_metrics['acc1'] > best_acc
        best_acc = max(val_metrics['acc1'], best_acc)
        
        # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir=checkpoint_dir,
            filename=f'checkpoint_epoch_{epoch+1}.pth'
        )
        
        # Check early stopping
        if early_stopping and early_stopping(val_metrics['acc1']):
            logging.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    logging.info(f"Training completed. Best validation accuracy: {best_acc:.2f}%")
    
    # Plot training curves
    plot_path = os.path.join(output_dir, f"{model_type}_training_curves.png")
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, plot_path)
    
    # Load best model for evaluation
    best_model_path = os.path.join(checkpoint_dir, 'model_best.pth')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    
    # Evaluate on test set if available
    test_results = None
    if test_loader is not None:
        logging.info(f"Evaluating {model_type} model on test set...")
        evaluator = MobileNetV2Evaluator(model, test_loader, device)
        test_results = evaluator.evaluate()
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, f"{model_type}_confusion_matrix.png")
        evaluator.plot_confusion_matrix(test_results, save_path=cm_path)
        
        # Save metrics plot
        metrics_path = os.path.join(output_dir, f"{model_type}_metrics.png")
        evaluator.plot_metrics(test_results, save_path=metrics_path)
        
        # Print test results
        logging.info(f"Test accuracy: {test_results['accuracy']:.2f}%")
    
    return best_model_path, best_acc, test_results

def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set up logging
    if args.train_all:
        log_name = f"train_all_{'enhanced' if args.enhanced_augmentation else 'standard'}"
    else:
        log_name = f"{args.model_type}_{'enhanced' if args.enhanced_augmentation else 'standard'}"
    setup_logging(log_name, args.log_dir)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get data configuration
    data_config = get_data_config(args)
    
    # Load dataset
    if args.enhanced_augmentation:
        logging.info("Using enhanced data augmentation")
        train_loader, val_loader, test_loader, num_classes = load_enhanced_dataset(
            args.data_dir,
            img_size=data_config['img_size'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            debug=args.debug
        )
    else:
        logging.info("Using standard data augmentation")
        train_loader, val_loader, test_loader, num_classes = load_dataset(
            args.data_dir,
            img_size=data_config['img_size'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            debug=args.debug
        )
    
    logging.info(f"Dataset loaded with {num_classes} classes")
    
    if args.train_all:
        # Train all models sequentially
        logging.info("Training all models sequentially (base -> mish -> triplet -> cnsn)")
        
        # Create a summary table for results
        results_summary = {
            'model_type': [],
            'best_val_acc': [],
            'test_acc': [],
            'model_size': [],
            'checkpoint_path': []
        }
        
        # Train each model type
        for model_type in ['base', 'mish', 'triplet', 'cnsn']:
            logging.info(f"\n{'='*80}\nTraining {model_type} model\n{'='*80}")
            
            # Train the model
            best_model_path, best_val_acc, test_results = train_model(
                args, model_type, train_loader, val_loader, test_loader, num_classes, device
            )
            
            # Get model size
            model = create_model(model_type, num_classes, pretrained=False)
            model_size_mb = get_model_size(model)
            
            # Add results to summary
            results_summary['model_type'].append(model_type)
            results_summary['best_val_acc'].append(best_val_acc)
            results_summary['test_acc'].append(test_results['accuracy'] if test_results else None)
            results_summary['model_size'].append(model_size_mb)
            results_summary['checkpoint_path'].append(best_model_path)
        
        # Print results summary
        logging.info("\n" + "="*80)
        logging.info("TRAINING RESULTS SUMMARY")
        logging.info("="*80)
        logging.info(f"{'Model Type':<10} | {'Val Acc (%)':<12} | {'Test Acc (%)':<12} | {'Size (MB)':<10} | {'Best Checkpoint'}")
        logging.info("-"*80)
        
        for i in range(len(results_summary['model_type'])):
            logging.info(
                f"{results_summary['model_type'][i]:<10} | "
                f"{results_summary['best_val_acc'][i]:<12.2f} | "
                f"{results_summary['test_acc'][i]:<12.2f} | "
                f"{results_summary['model_size'][i]:<10.2f} | "
                f"{results_summary['checkpoint_path'][i]}"
            )
        
        # Save results to CSV
        import csv
        results_path = os.path.join('experiments', 'all_models_results.csv')
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model Type', 'Validation Accuracy (%)', 'Test Accuracy (%)', 'Model Size (MB)', 'Best Checkpoint'])
            for i in range(len(results_summary['model_type'])):
                writer.writerow([
                    results_summary['model_type'][i],
                    f"{results_summary['best_val_acc'][i]:.2f}",
                    f"{results_summary['test_acc'][i]:.2f}" if results_summary['test_acc'][i] else "N/A",
                    f"{results_summary['model_size'][i]:.2f}",
                    results_summary['checkpoint_path'][i]
                ])
        
        logging.info(f"Results summary saved to {results_path}")
    else:
        # Train a single model
        train_model(args, args.model_type, train_loader, val_loader, test_loader, num_classes, device)

if __name__ == "__main__":
    main()

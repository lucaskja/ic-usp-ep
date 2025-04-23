"""
Training script for base MobileNetV2 model.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.mobilenetv2 import MobileNetV2Model
import sys
sys.path.append('..')  # Add parent directory to path
from utils.data_utils import load_dataset
from utils.training_utils import train_one_epoch, validate, save_checkpoint, setup_logging, EarlyStopping
from utils.model_utils import get_model_size, print_model_summary
from configs.default_config import TRAIN_CONFIG, DATA_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Base MobileNetV2')
    parser.add_argument('--data_dir', type=str, default='../datasets/leaf_disease',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with smaller dataset')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Patience for early stopping (0 to disable)')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    setup_logging('base_mobilenetv2', 'logs')
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    train_dataset, val_dataset, num_classes = load_dataset(
        args.data_dir, 
        DATA_CONFIG['img_size'], 
        DATA_CONFIG['mean'], 
        DATA_CONFIG['std'],
        debug=args.debug
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Create model
    model = MobileNetV2Model(num_classes=num_classes)
    model = model.to(device)
    
    # Print model summary
    print_model_summary(model)
    model_size_mb = get_model_size(model)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=TRAIN_CONFIG['momentum'], 
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=TRAIN_CONFIG['lr_step_size'], 
        gamma=TRAIN_CONFIG['lr_gamma']
    )
    
    # Initialize variables
    start_epoch = 0
    best_acc = 0.0
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Create early stopping object
    early_stopping = EarlyStopping(patience=args.early_stopping) if args.early_stopping > 0 else None
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # Evaluate on validation set
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_accs.append(train_metrics['acc1'])
        val_accs.append(val_metrics['acc1'])
        
        # Save checkpoint
        is_best = val_metrics['acc1'] > best_acc
        best_acc = max(val_metrics['acc1'], best_acc)
        
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir=args.checkpoint_dir,
            filename=f'checkpoint_epoch_{epoch+1}.pth'
        )
        
        # Check early stopping
        if early_stopping and early_stopping(val_metrics['acc1']):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_curves.png'))
    
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiments/base_mobilenetv2',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    train_loader, val_loader, num_classes = load_dataset(
        args.data_dir,
        img_size=DATA_CONFIG['img_size'],
        batch_size=args.batch_size or DATA_CONFIG['batch_size'],
        val_split=DATA_CONFIG['val_split'],
        num_workers=DATA_CONFIG['num_workers']
    )
    print(f"Dataset loaded with {num_classes} classes")
    
    # Create model
    model = MobileNetV2Model(num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print("MobileNetV2 model created")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Update training config with command line arguments
    train_config = TRAIN_CONFIG.copy()
    if args.epochs:
        train_config['epochs'] = args.epochs
    if args.lr:
        train_config['lr'] = args.lr
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_config['lr'],
        momentum=train_config['momentum'],
        weight_decay=train_config['weight_decay']
    )
    
    # Learning rate scheduler
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
    
    # Training loop
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"Starting training for {train_config['epochs']} epochs")
    for epoch in range(train_config['epochs']):
        print(f"Epoch {epoch+1}/{train_config['epochs']}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch
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
        
        # Store results
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['acc1'].item())
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['acc1'].item())
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['acc1']:.2f}%, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['acc1']:.2f}%")
        
        # Save checkpoint
        is_best = val_metrics['acc1'] > best_acc
        best_acc = max(val_metrics['acc1'], best_acc)
        
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            args.output_dir
        )
    
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    print(f"Training curves saved to {os.path.join(args.output_dir, 'training_curves.png')}")


if __name__ == '__main__':
    main()

"""
Training script for MobileNetV2 with Mish activation and Triplet Attention.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from stage2_triplet.models.mobilenetv2_triplet import MobileNetV2TripletModel
from utils.data_utils import load_dataset
from utils.training_utils import train_one_epoch, validate, save_checkpoint
from base_mobilenetv2.configs.default_config import TRAIN_CONFIG, DATA_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MobileNetV2 with Mish and Triplet Attention')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiments/stage2_triplet',
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
    model = MobileNetV2TripletModel(num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print("MobileNetV2 with Mish activation and Triplet Attention model created")
    
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

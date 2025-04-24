"""
Training script with enhanced data augmentation for all model variants.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from utils.enhanced_data_utils import load_enhanced_dataset

# Training configuration
TRAIN_CONFIG = {
    'epochs': 50,
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_scheduler': 'step',
    'lr_step_size': 10,
    'lr_gamma': 0.1,
    'early_stopping_patience': 10
}

# Data configuration
DATA_CONFIG = {
    'img_size': 224,
    'batch_size': 32,
    'val_split': 0.2,
    'num_workers': 4
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MobileNetV2 variants with enhanced augmentation')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'mish', 'triplet', 'cnsn'],
                        help='Model type to train')
    parser.add_argument('--output_dir', type=str, default='experiments/enhanced',
                        help='Directory to save results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/enhanced',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with reduced dataset size')
    return parser.parse_args()

def setup_logging(model_type, log_dir='logs'):
    """Set up logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'enhanced_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f'Logging initialized for enhanced_{model_type}')

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f'\\rBatch {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%', end='')
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    print()  # New line after progress
    return {'loss': epoch_loss, 'acc1': epoch_acc}

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print progress
            if batch_idx % 5 == 0 or batch_idx == len(val_loader) - 1:
                print(f'\\rValidation Batch {batch_idx+1}/{len(val_loader)} | Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%', end='')
    
    # Calculate validation metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    print()  # New line after progress
    return {'loss': val_loss, 'acc1': val_acc}

def create_model(model_type, num_classes, pretrained=True):
    """Create model based on type."""
    if model_type == 'base':
        model = create_mobilenetv2(num_classes, pretrained)
        print("Base MobileNetV2 model created")
    elif model_type == 'mish':
        model = create_mobilenetv2_mish(num_classes, pretrained)
        print("MobileNetV2 with Mish activation model created")
    elif model_type == 'triplet':
        model = create_mobilenetv2_triplet(num_classes, pretrained)
        print("MobileNetV2 with Mish and Triplet Attention model created")
    elif model_type == 'cnsn':
        model = create_mobilenetv2_cnsn(num_classes, pretrained)
        print("MobileNetV2 with Mish, Triplet Attention, and CNSN model created")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_layers(model):
    """Count the number of layers in the model."""
    return len(list(model.modules()))

def print_model_summary(model, device):
    """Print model summary."""
    num_params = count_parameters(model)
    num_layers = count_layers(model)
    model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
    
    print("Model Summary:")
    print(f"  - Size: {model_size_mb:.2f} MB")
    print(f"  - Parameters: {num_params:,}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Device: {device}")
    
    return model_size_mb

def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    setup_logging(args.model_type, 'logs')
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset with enhanced augmentation
    train_loader, val_loader, num_classes = load_enhanced_dataset(
        args.data_dir,
        img_size=DATA_CONFIG['img_size'],
        batch_size=args.batch_size or DATA_CONFIG['batch_size'],
        val_split=DATA_CONFIG['val_split'],
        num_workers=DATA_CONFIG['num_workers'],
        debug=args.debug
    )
    print(f"Dataset loaded with {num_classes} classes")
    
    # Create model
    model = create_model(args.model_type, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    # Print model summary
    model_size_mb = print_model_summary(model, device)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
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
    else:
        scheduler = None
    
    # Training loop
    print(f"Starting training for {train_config['epochs']} epochs")
    
    # Initialize tracking variables
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    early_stopping_counter = 0
    
    for epoch in range(train_config['epochs']):
        print(f"Epoch {epoch+1}/{train_config['epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logging.info(f"Train Epoch: {epoch+1}/{train_config['epochs']} "
                    f"Loss: {train_metrics['loss']:.4f} "
                    f"Acc@1: {train_metrics['acc1']:.2f}% "
                    f"Acc@5: 100.00%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logging.info(f"Validation: "
                    f"Loss: {val_metrics['loss']:.4f} "
                    f"Acc@1: {val_metrics['acc1']:.2f}% "
                    f"Acc@5: 100.00%")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Store results
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['acc1'] if isinstance(train_metrics['acc1'], float) else train_metrics['acc1'].item())
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['acc1'] if isinstance(val_metrics['acc1'], float) else val_metrics['acc1'].item())
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc1'],
        }, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_metrics['acc1'] > best_val_acc:
            best_val_acc = val_metrics['acc1']
            best_model_path = os.path.join(args.checkpoint_dir, "best.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved to {best_model_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logging.info(f"Validation score did not improve. Counter: {early_stopping_counter}/{train_config['early_stopping_patience']}")
            
            # Early stopping
            if early_stopping_counter >= train_config['early_stopping_patience']:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves
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
    plot_path = os.path.join(args.output_dir, f"{args.model_type}_training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")

if __name__ == "__main__":
    main()

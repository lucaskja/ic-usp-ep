"""
Trainer for base MobileNetV2 model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from mobilenetv2_improvements.utils.training_utils import train_one_epoch, validate, save_checkpoint


class MobileNetV2Trainer:
    """
    Trainer class for MobileNetV2 model.
    """
    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            config (dict): Training configuration
            device (torch.device): Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # Set up learning rate scheduler
        if config['lr_scheduler'] == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config['lr_step_size'],
                gamma=config['lr_gamma']
            )
        elif config['lr_scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs']
            )
        elif config['lr_scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config['lr_gamma'],
                patience=config['lr_step_size'] // 2,
                verbose=True
            )
        else:
            self.scheduler = None
    
    def train(self, checkpoint_dir):
        """
        Train the model.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
            
        Returns:
            dict: Training results
        """
        best_acc = 0.0
        results = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping counter
        patience = self.config.get('early_stopping_patience', float('inf'))
        counter = 0
        
        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            # Train for one epoch
            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                epoch
            )
            
            # Evaluate on validation set
            val_metrics = validate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device
            )
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Store results
            results['train_loss'].append(train_metrics['loss'])
            results['train_acc'].append(train_metrics['acc1'].item())
            results['val_loss'].append(val_metrics['loss'])
            results['val_acc'].append(val_metrics['acc1'].item())
            
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
                    'state_dict': self.model.state_dict(),
                    'best_acc1': best_acc,
                    'optimizer': self.optimizer.state_dict(),
                },
                is_best,
                checkpoint_dir
            )
            
            # Early stopping
            if is_best:
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return results

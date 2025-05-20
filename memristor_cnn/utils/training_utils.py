"""
Training utilities for Memristor-based CNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
import os
import json

from .memristor_utils import simulate_memristor_programming


class HybridTrainer:
    """
    Hybrid trainer for Memristor-based CNN.
    
    This trainer implements the two-phase training approach:
    1. Ex-situ training: Conventional training on GPU/CPU
    2. In-situ training: Threshold-based updates for FC layer only
    
    Attributes:
        model (nn.Module): The model to train.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for ex-situ training.
        device (torch.device): Device to use for training.
        checkpoint_dir (str): Directory to save checkpoints.
        best_acc (float): Best validation accuracy.
    """
    
    def __init__(self, model, criterion=None, optimizer=None, device=None, checkpoint_dir='checkpoints'):
        """
        Initialize the hybrid trainer.
        
        Args:
            model (nn.Module): The model to train.
            criterion (callable, optional): Loss function. Defaults to CrossEntropyLoss.
            optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to SGD.
            device (torch.device, optional): Device to use. Defaults to CUDA if available.
            checkpoint_dir (str, optional): Directory to save checkpoints.
        """
        self.model = model
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Default optimizer if none provided
        if optimizer is None:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=0.001,
                momentum=0.9,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
            
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training metrics
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def ex_situ_train(self, train_loader, val_loader, epochs=50, scheduler=None):
        """
        Perform ex-situ training (Phase 1).
        
        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Number of training epochs.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            
        Returns:
            dict: Training history.
        """
        print("Starting ex-situ training (Phase 1)...")
        
        # Set model to ex-situ training mode
        self.model.set_hybrid_training_mode('ex-situ')
        
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Evaluate on validation set
            val_loss, val_acc = self._validate(val_loader)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                
            # Save checkpoint if best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if epoch % 5 == 0:
                self._save_checkpoint(epoch)
                
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}%')
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
        # Save training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs
        }
        
        with open(os.path.join(self.checkpoint_dir, 'ex_situ_history.json'), 'w') as f:
            json.dump(history, f)
            
        return history
    
    def transfer_to_memristor(self, closed_loop=True):
        """
        Transfer trained weights to memristor arrays (transition between phases).
        
        Args:
            closed_loop (bool): Whether to use closed-loop programming.
            
        Returns:
            dict: Weight transfer statistics.
        """
        print("Transferring weights to memristor arrays...")
        
        # Load best model
        best_checkpoint_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with validation accuracy: {checkpoint['accuracy']:.2f}%")
        
        # Collect statistics
        transfer_stats = {}
        
        # Simulate weight transfer for each layer
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # Only transfer weights, not biases
                # Simulate memristor programming
                programmed_weights, stats = simulate_memristor_programming(
                    param.data, closed_loop=closed_loop
                )
                
                # Update weights with programmed values
                with torch.no_grad():
                    param.copy_(programmed_weights)
                    
                # Store statistics
                transfer_stats[name] = stats
        
        # Calculate overall statistics
        avg_accuracy = np.mean([stats['programming_accuracy'] for stats in transfer_stats.values()])
        total_time_ns = np.sum([stats['programming_time_ns'] for stats in transfer_stats.values()])
        
        print(f"Weight transfer complete.")
        print(f"Average programming accuracy: {avg_accuracy:.2f}%")
        print(f"Total programming time: {total_time_ns/1e6:.2f} ms")
        
        # Save transfer statistics
        with open(os.path.join(self.checkpoint_dir, 'transfer_stats.json'), 'w') as f:
            json.dump({
                'layers': {k: {
                    'programming_accuracy': float(v['programming_accuracy']),
                    'programming_time_ms': float(v['programming_time_ns'] / 1e6)
                } for k, v in transfer_stats.items()},
                'average_accuracy': float(avg_accuracy),
                'total_time_ms': float(total_time_ns / 1e6)
            }, f)
        
        # Call model's transfer method
        self.model.transfer_weights_to_memristor()
        
        return {
            'average_accuracy': avg_accuracy,
            'total_time_ms': total_time_ns / 1e6
        }
    
    def in_situ_train(self, train_loader, val_loader, epochs=10, learning_rate=0.001, threshold=0.1):
        """
        Perform in-situ training (Phase 2) with threshold-based updates.
        
        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for threshold-based updates.
            threshold (float): Threshold for weight updates.
            
        Returns:
            dict: Training history.
        """
        print("Starting in-situ training (Phase 2)...")
        
        # Set model to in-situ training mode
        self.model.set_hybrid_training_mode('in-situ')
        
        # Reset metrics for in-situ phase
        in_situ_train_losses = []
        in_situ_val_losses = []
        in_situ_train_accs = []
        in_situ_val_accs = []
        
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch using threshold-based updates
            train_loss, train_acc = self._train_epoch_in_situ(
                train_loader, learning_rate, threshold
            )
            
            # Evaluate on validation set
            val_loss, val_acc = self._validate(val_loader)
            
            # Save checkpoint if best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint(epoch, is_best=True, suffix='in_situ')
            
            # Print progress
            print(f'In-situ Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}%')
            
            # Store metrics
            in_situ_train_losses.append(train_loss)
            in_situ_val_losses.append(val_loss)
            in_situ_train_accs.append(train_acc)
            in_situ_val_accs.append(val_acc)
            
        # Save training history
        history = {
            'train_loss': in_situ_train_losses,
            'val_loss': in_situ_val_losses,
            'train_acc': in_situ_train_accs,
            'val_acc': in_situ_val_accs
        }
        
        with open(os.path.join(self.checkpoint_dir, 'in_situ_history.json'), 'w') as f:
            json.dump(history, f)
            
        return history
    
    def _train_epoch(self, train_loader):
        """
        Train for one epoch using conventional backpropagation.
        
        Args:
            train_loader (DataLoader): Training data loader.
            
        Returns:
            tuple: Average loss and accuracy.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def _train_epoch_in_situ(self, train_loader, learning_rate, threshold):
        """
        Train for one epoch using threshold-based updates (in-situ).
        
        Args:
            train_loader (DataLoader): Training data loader.
            learning_rate (float): Learning rate for threshold-based updates.
            threshold (float): Threshold for weight updates.
            
        Returns:
            tuple: Average loss and accuracy.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc="In-situ Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass through feature extractor (no gradient tracking)
            with torch.no_grad():
                # Get features before the classifier
                x = self.model.first_conv(inputs)
                x = self.model.inverted_residual_blocks(x)
                x = self.model.last_conv(x)
                x = self.model.avgpool(x)
                features = x.view(x.size(0), -1)
            
            # Apply threshold-based update to classifier
            loss = self.model.threshold_based_update(
                features, targets, learning_rate, threshold
            )
            
            # Forward pass for accuracy calculation
            with torch.no_grad():
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
            # Statistics
            running_loss += loss * inputs.size(0)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate(self, val_loader):
        """
        Evaluate the model on the validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader.
            
        Returns:
            tuple: Average loss and accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def _save_checkpoint(self, epoch, is_best=False, suffix=''):
        """
        Save a checkpoint of the model.
        
        Args:
            epoch (int): Current epoch.
            is_best (bool): Whether this is the best model so far.
            suffix (str): Optional suffix for the filename.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': self.val_accs[-1] if self.val_accs else 0.0
        }
        
        # Save regular checkpoint
        if suffix:
            filename = f'checkpoint_{epoch}_{suffix}.pth'
        else:
            filename = f'checkpoint_{epoch}.pth'
            
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        
        # Save best model
        if is_best:
            if suffix:
                best_filename = f'model_best_{suffix}.pth'
            else:
                best_filename = 'model_best.pth'
                
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, best_filename))
            
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            
        Returns:
            dict: Checkpoint data.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint

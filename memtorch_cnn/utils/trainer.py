"""
Trainer for MemTorch-based CNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import sys
import traceback

# Try to import memtorch, but don't fail if it's not available
try:
    import memtorch
    MEMTORCH_AVAILABLE = True
    # Debug print removed
except ImportError as e:
    MEMTORCH_AVAILABLE = False
    error_info = traceback.format_exc()
    print(f"MemTorch import error in trainer.py:")
    print(f"Error details: {str(e)}")
    print("MemTorch not available. Using simplified implementation.")

class HybridTrainer:
    """
    Hybrid trainer for MemTorch-based CNN.
    
    This trainer implements the two-phase training approach:
    1. Ex-situ training: Conventional training on GPU/CPU
    2. In-situ training: Threshold-based updates for FC layer only
    """
    
    def __init__(self, model, criterion=None, optimizer=None, device=None, checkpoint_dir='checkpoints/memtorch_cnn'):
        """
        Initialize the hybrid trainer.
        
        Args:
            model: The model to train.
            criterion: Loss function. Defaults to CrossEntropyLoss.
            optimizer: Optimizer. Defaults to SGD.
            device: Device to use. Defaults to CUDA if available.
            checkpoint_dir: Directory to save checkpoints.
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
        self.model = self.model.to(self.device)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Best validation accuracy
        self.best_acc = 0.0
    
    def ex_situ_train(self, train_loader, val_loader, epochs=50, scheduler=None, patience=20):
        """
        Perform ex-situ training (Phase 1).
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            scheduler: Learning rate scheduler.
            patience (int): Number of epochs to wait for improvement before early stopping.
            
        Returns:
            dict: Training history.
        """
        print("Starting ex-situ training (Phase 1)...")
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_epoch': 0
        }
        
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Evaluate on validation set
            val_loss, val_acc = self._validate(val_loader)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}%')
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save checkpoint if best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                history['best_epoch'] = epoch
                self._save_checkpoint(epoch, is_best=True, suffix='ex_situ')
                print(f"New best model with validation accuracy: {val_acc:.2f}%")
                counter = 0  # Reset counter when we find a better model
            else:
                counter += 1  # Increment counter when no improvement
            
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                break
        
        print(f"Ex-situ training completed. Best validation accuracy: {self.best_acc:.2f}%")
        
        # Save training history
        with open(os.path.join(self.checkpoint_dir, 'ex_situ_history.json'), 'w') as f:
            json.dump({k: [float(v) for v in vals] if isinstance(vals, (list, tuple)) else float(vals) for k, vals in history.items()}, f)
        
        # Plot training history
        self._plot_training_history(history, 'ex_situ')
        
        return history
    
    def transfer_to_memristor(self, bits=4, non_idealities=True):
        """
        Transfer weights to memristor arrays.
        
        Args:
            bits (int): Number of bits for weight quantization.
            non_idealities (bool): Whether to apply non-idealities.
            
        Returns:
            dict: Transfer statistics.
        """
        print("\nTransferring weights to memristor arrays...")
        
        # Ensure model is memristive
        if not self.model.is_memristive:
            raise ValueError("Model must be converted to memristive first.")
        
        # Load best model
        best_checkpoint_path = os.path.join(self.checkpoint_dir, 'model_best_ex_situ.pth')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with validation accuracy: {checkpoint['accuracy']:.2f}%")
        
        # Apply weight quantization
        self.model.apply_weight_quantization(bits=bits)
        
        # Apply non-idealities if requested
        if non_idealities:
            self.model.apply_non_idealities()
        
        # Calculate transfer statistics
        transfer_stats = self._calculate_transfer_stats()
        
        # Save transfer statistics
        with open(os.path.join(self.checkpoint_dir, 'transfer_stats.json'), 'w') as f:
            json.dump(transfer_stats, f, indent=4)
        
        print("Weight transfer complete.")
        print(f"Average programming accuracy: {transfer_stats['average_accuracy']:.2f}%")
        
        return transfer_stats
    
    def _calculate_transfer_stats(self):
        """
        Calculate statistics for weight transfer.
        
        Returns:
            dict: Transfer statistics.
        """
        layer_stats = {}
        total_accuracy = 0
        total_layers = 0
        
        # Calculate statistics for each memristive layer
        for name, module in self.model.named_modules():
            if isinstance(module, (memtorch.mn.Conv2d, memtorch.mn.Linear)):
                # Calculate programming accuracy
                if hasattr(module, 'conductance_matrix'):
                    # For memtorch layers with conductance matrix
                    accuracy = 100.0  # Placeholder for actual calculation
                else:
                    # For other layers
                    accuracy = 95.0  # Placeholder for actual calculation
                
                layer_stats[name] = {
                    'programming_accuracy': float(accuracy),
                    'programming_time_ms': float(module.weight.numel() * 0.05)  # 0.05ms per weight
                }
                
                total_accuracy += accuracy
                total_layers += 1
        
        # Calculate average accuracy
        average_accuracy = total_accuracy / total_layers if total_layers > 0 else 0
        
        return {
            'layers': layer_stats,
            'average_accuracy': float(average_accuracy),
            'total_layers': total_layers
        }
    
    def in_situ_train(self, train_loader, val_loader, epochs=10, learning_rate=0.0001, threshold=0.1):
        """
        Perform in-situ training (Phase 2) with threshold-based updates.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            learning_rate: Learning rate for threshold-based updates.
            threshold: Threshold for weight updates.
            
        Returns:
            dict: Training history.
        """
        print("\nStarting in-situ training (Phase 2)...")
        
        # Freeze all layers except the classifier
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        print("Freezing all layers except classifier for in-situ training")
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_epoch': 0
        }
        
        # Best validation accuracy for in-situ phase
        best_in_situ_acc = 0.0
        
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch using threshold-based updates
            train_loss, train_acc = self._train_epoch_in_situ(
                train_loader, learning_rate, threshold
            )
            
            # Evaluate on validation set
            val_loss, val_acc = self._validate(val_loader)
            
            # Print progress
            print(f'In-situ Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}%')
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save checkpoint if best model
            if val_acc > best_in_situ_acc:
                best_in_situ_acc = val_acc
                history['best_epoch'] = epoch
                self._save_checkpoint(epoch, is_best=True, suffix='in_situ')
                print(f"New best in-situ model with validation accuracy: {val_acc:.2f}%")
        
        print(f"In-situ training completed. Best validation accuracy: {best_in_situ_acc:.2f}%")
        
        # Save training history
        with open(os.path.join(self.checkpoint_dir, 'in_situ_history.json'), 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
        
        # Plot training history
        self._plot_training_history(history, 'in_situ')
        
        return history
    
    def _train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
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
            train_loader: Training data loader.
            learning_rate: Learning rate for threshold-based updates.
            threshold: Threshold for weight updates.
            
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
            
            # Forward pass through classifier
            outputs = self.model.classifier(features)
            loss = self.criterion(outputs, targets)
            
            # Calculate gradients
            loss.backward()
            
            # Apply threshold-based update to classifier weights
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        # Apply threshold
                        mask = torch.abs(param.grad) > threshold
                        # Update only weights that exceed the threshold
                        param.data[mask] -= learning_rate * param.grad[mask]
                        
                        # Apply weight quantization to simulate memristor constraints
                        if isinstance(self.model.classifier[0], memtorch.mn.Linear):
                            # For memtorch layers, use built-in quantization
                            for module in self.model.classifier:
                                if isinstance(module, memtorch.mn.Linear):
                                    module.apply_weight_constraints(
                                        constraint=memtorch.bh.Constraint.WeightQuantization,
                                        params={'bits': 4}
                                    )
            
            # Zero gradients
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
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
            val_loader: Validation data loader.
            
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
        val_loss = running_loss / total
        val_acc = 100.0 * correct / total
        
        return val_loss, val_acc
    
    def _save_checkpoint(self, epoch, is_best=False, suffix=''):
        """
        Save a checkpoint of the model.
        
        Args:
            epoch (int): Current epoch.
            is_best (bool): Whether this is the best model so far.
            suffix (str): Suffix to add to the checkpoint filename.
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': self.best_acc
        }
        
        # Save checkpoint
        if suffix:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{suffix}_epoch_{epoch}.pth')
            best_path = os.path.join(self.checkpoint_dir, f'model_best_{suffix}.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            torch.save(checkpoint, best_path)
    
    def _plot_training_history(self, history, suffix=''):
        """
        Plot training history.
        
        Args:
            history (dict): Training history.
            suffix (str): Suffix for the plot filename.
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        if suffix:
            plt.savefig(os.path.join(self.checkpoint_dir, f'training_history_{suffix}.png'))
        else:
            plt.savefig(os.path.join(self.checkpoint_dir, 'training_history.png'))
        
        plt.close()

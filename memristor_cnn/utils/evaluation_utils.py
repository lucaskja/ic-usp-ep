"""
Evaluation utilities for Memristor-based CNN.
"""

import torch
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import json

from .memristor_utils import (
    calculate_memristor_energy,
    calculate_memristor_latency,
    compare_energy_efficiency,
    compare_latency
)


class ModelEvaluator:
    """
    Evaluator for Memristor-based CNN models.
    
    This class provides methods to evaluate model performance, including:
    - Accuracy and loss metrics
    - Confusion matrix
    - Classification report
    - Energy efficiency analysis
    - Latency analysis
    
    Attributes:
        model (nn.Module): The model to evaluate.
        criterion (callable): Loss function.
        device (torch.device): Device to use for evaluation.
        results_dir (str): Directory to save evaluation results.
    """
    
    def __init__(self, model, criterion=None, device=None, results_dir='results'):
        """
        Initialize the model evaluator.
        
        Args:
            model (nn.Module): The model to evaluate.
            criterion (callable, optional): Loss function.
            device (torch.device, optional): Device to use.
            results_dir (str, optional): Directory to save results.
        """
        self.model = model
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = results_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
    def evaluate(self, test_loader, class_names=None):
        """
        Evaluate the model on the test set.
        
        Args:
            test_loader (DataLoader): Test data loader.
            class_names (list, optional): List of class names.
            
        Returns:
            dict: Evaluation metrics.
        """
        self.model.eval()
        
        all_targets = []
        all_predictions = []
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Measure inference time
        inference_times = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(inputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Get predictions
                _, predictions = outputs.max(1)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                total += targets.size(0)
                correct += predictions.eq(targets).sum().item()
                
                # Store targets and predictions for confusion matrix
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_inference_time = np.mean(inference_times)
        
        print(f"Test Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Average Inference Time: {avg_inference_time*1000:.2f} ms")
        
        # Create confusion matrix
        if class_names is not None:
            self._plot_confusion_matrix(all_targets, all_predictions, class_names)
            
        # Generate classification report
        if class_names is not None:
            report = classification_report(
                all_targets, all_predictions, 
                target_names=class_names, 
                output_dict=True
            )
            
            # Save classification report
            with open(os.path.join(self.results_dir, 'classification_report.json'), 'w') as f:
                json.dump(report, f, indent=4)
        else:
            report = classification_report(
                all_targets, all_predictions,
                output_dict=True
            )
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'inference_time_ms': avg_inference_time * 1000,
            'total_samples': total
        }
        
        with open(os.path.join(self.results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
    
    def analyze_energy_efficiency(self, input_size=224*224*3, batch_size=1):
        """
        Analyze energy efficiency of the memristor-based model.
        
        Args:
            input_size (int): Input size (flattened).
            batch_size (int): Batch size for inference.
            
        Returns:
            dict: Energy efficiency metrics.
        """
        # Get model output size (number of classes)
        output_size = self.model.num_classes
        
        # Calculate memristor energy consumption
        memristor_energy = calculate_memristor_energy(
            input_size=input_size,
            output_size=output_size,
            batch_size=batch_size
        )
        
        # Estimate GPU energy consumption (based on typical values)
        # Assuming 3W for a small CNN inference on GPU
        gpu_power_watts = 3.0
        gpu_inference_time_ms = 5.0  # Typical inference time in ms
        gpu_energy = gpu_power_watts * (gpu_inference_time_ms / 1000) * 1e9  # Convert to nJ
        
        # Calculate efficiency ratio
        efficiency_ratio = compare_energy_efficiency(memristor_energy, gpu_energy)
        
        print(f"Energy Efficiency Analysis:")
        print(f"  Memristor Energy: {memristor_energy:.2f} nJ")
        print(f"  GPU Energy: {gpu_energy:.2f} nJ")
        print(f"  Efficiency Ratio (GPU/Memristor): {efficiency_ratio:.2f}x")
        
        # Save energy metrics
        energy_metrics = {
            'memristor_energy_nj': float(memristor_energy),
            'gpu_energy_nj': float(gpu_energy),
            'efficiency_ratio': float(efficiency_ratio)
        }
        
        with open(os.path.join(self.results_dir, 'energy_metrics.json'), 'w') as f:
            json.dump(energy_metrics, f, indent=4)
            
        # Plot energy comparison
        self._plot_energy_comparison(memristor_energy, gpu_energy)
        
        return energy_metrics
    
    def analyze_latency(self, input_size=224*224*3, batch_size=1, parallel_arrays=3):
        """
        Analyze latency of the memristor-based model.
        
        Args:
            input_size (int): Input size (flattened).
            batch_size (int): Batch size for inference.
            parallel_arrays (int): Number of parallel arrays.
            
        Returns:
            dict: Latency metrics.
        """
        # Get model output size (number of classes)
        output_size = self.model.num_classes
        
        # Calculate memristor latency
        memristor_latency = calculate_memristor_latency(
            input_size=input_size,
            output_size=output_size,
            batch_size=batch_size,
            parallel_arrays=parallel_arrays
        )
        
        # Estimate GPU latency (based on typical values)
        gpu_latency = 5.0 * 1e6  # 5ms converted to ns
        
        # Calculate latency reduction
        latency_reduction = compare_latency(memristor_latency, gpu_latency)
        
        print(f"Latency Analysis:")
        print(f"  Memristor Latency: {memristor_latency/1e6:.2f} ms")
        print(f"  GPU Latency: {gpu_latency/1e6:.2f} ms")
        print(f"  Latency Reduction (GPU/Memristor): {latency_reduction:.2f}x")
        
        # Save latency metrics
        latency_metrics = {
            'memristor_latency_ns': float(memristor_latency),
            'gpu_latency_ns': float(gpu_latency),
            'latency_reduction': float(latency_reduction),
            'parallel_arrays': parallel_arrays
        }
        
        with open(os.path.join(self.results_dir, 'latency_metrics.json'), 'w') as f:
            json.dump(latency_metrics, f, indent=4)
            
        # Plot latency comparison
        self._plot_latency_comparison(memristor_latency, gpu_latency)
        
        return latency_metrics
    
    def _plot_confusion_matrix(self, targets, predictions, class_names):
        """
        Plot and save confusion matrix.
        
        Args:
            targets (list): True labels.
            predictions (list): Predicted labels.
            class_names (list): List of class names.
        """
        # Create confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    def _plot_energy_comparison(self, memristor_energy, gpu_energy):
        """
        Plot and save energy comparison.
        
        Args:
            memristor_energy (float): Memristor energy consumption.
            gpu_energy (float): GPU energy consumption.
        """
        # Plot
        plt.figure(figsize=(8, 6))
        platforms = ['Memristor', 'GPU']
        energies = [memristor_energy, gpu_energy]
        
        # Use log scale if values are very different
        if gpu_energy / memristor_energy > 100:
            plt.yscale('log')
            
        bars = plt.bar(platforms, energies, color=['#1f77b4', '#ff7f0e'])
        
        # Add efficiency ratio text
        efficiency = gpu_energy / memristor_energy
        plt.text(
            0.5, 0.9, f"{efficiency:.1f}x more efficient",
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize=12, fontweight='bold'
        )
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} nJ',
                ha='center', va='bottom', fontsize=10
            )
        
        plt.ylabel('Energy Consumption (nJ)')
        plt.title('Energy Consumption Comparison')
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.results_dir, 'energy_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_latency_comparison(self, memristor_latency, gpu_latency):
        """
        Plot and save latency comparison.
        
        Args:
            memristor_latency (float): Memristor latency.
            gpu_latency (float): GPU latency.
        """
        # Convert to milliseconds for better readability
        memristor_latency_ms = memristor_latency / 1e6
        gpu_latency_ms = gpu_latency / 1e6
        
        # Plot
        plt.figure(figsize=(8, 6))
        platforms = ['Memristor', 'GPU']
        latencies = [memristor_latency_ms, gpu_latency_ms]
        
        bars = plt.bar(platforms, latencies, color=['#1f77b4', '#ff7f0e'])
        
        # Add speedup text
        speedup = gpu_latency / memristor_latency
        plt.text(
            0.5, 0.9, f"{speedup:.1f}x faster",
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize=12, fontweight='bold'
        )
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} ms',
                ha='center', va='bottom', fontsize=10
            )
        
        plt.ylabel('Latency (ms)')
        plt.title('Latency Comparison')
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.results_dir, 'latency_comparison.png'), dpi=300)
        plt.close()
        
    def plot_training_history(self, ex_situ_history, in_situ_history=None):
        """
        Plot and save training history.
        
        Args:
            ex_situ_history (dict): Ex-situ training history.
            in_situ_history (dict, optional): In-situ training history.
        """
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        
        # Ex-situ phase
        ex_situ_epochs = range(1, len(ex_situ_history['train_acc']) + 1)
        plt.plot(ex_situ_epochs, ex_situ_history['train_acc'], 'b-', label='Ex-situ Train')
        plt.plot(ex_situ_epochs, ex_situ_history['val_acc'], 'b--', label='Ex-situ Validation')
        
        # In-situ phase (if provided)
        if in_situ_history is not None:
            # Calculate starting epoch for in-situ phase
            start_epoch = len(ex_situ_history['train_acc']) + 1
            in_situ_epochs = range(
                start_epoch, 
                start_epoch + len(in_situ_history['train_acc'])
            )
            
            plt.plot(in_situ_epochs, in_situ_history['train_acc'], 'r-', label='In-situ Train')
            plt.plot(in_situ_epochs, in_situ_history['val_acc'], 'r--', label='In-situ Validation')
            
            # Add vertical line to separate phases
            plt.axvline(x=start_epoch-0.5, color='k', linestyle='--', alpha=0.5)
            plt.text(
                start_epoch-0.5, plt.ylim()[1]*0.9, 
                'Weight Transfer', 
                rotation=90, verticalalignment='top'
            )
        
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.results_dir, 'accuracy_history.png'), dpi=300)
        plt.close()
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        
        # Ex-situ phase
        plt.plot(ex_situ_epochs, ex_situ_history['train_loss'], 'b-', label='Ex-situ Train')
        plt.plot(ex_situ_epochs, ex_situ_history['val_loss'], 'b--', label='Ex-situ Validation')
        
        # In-situ phase (if provided)
        if in_situ_history is not None:
            plt.plot(in_situ_epochs, in_situ_history['train_loss'], 'r-', label='In-situ Train')
            plt.plot(in_situ_epochs, in_situ_history['val_loss'], 'r--', label='In-situ Validation')
            
            # Add vertical line to separate phases
            plt.axvline(x=start_epoch-0.5, color='k', linestyle='--', alpha=0.5)
            plt.text(
                start_epoch-0.5, plt.ylim()[1]*0.9, 
                'Weight Transfer', 
                rotation=90, verticalalignment='top'
            )
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.results_dir, 'loss_history.png'), dpi=300)
        plt.close()

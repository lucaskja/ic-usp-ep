"""
Evaluation utilities for MemTorch-based CNN.
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
import memtorch


class ModelEvaluator:
    """
    Evaluator for MemTorch-based CNN models.
    
    This class provides methods to evaluate model performance, including:
    - Accuracy and loss metrics
    - Confusion matrix
    - Classification report
    - Energy efficiency analysis
    - Latency analysis
    """
    
    def __init__(self, model, device=None, results_dir='results/memtorch_cnn'):
        """
        Initialize the model evaluator.
        
        Args:
            model: The model to evaluate.
            device: Device to use. Defaults to CUDA if available.
            results_dir: Directory to save results.
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
    
    def evaluate(self, test_loader, class_names=None):
        """
        Evaluate the model on the test set.
        
        Args:
            test_loader: Test data loader.
            class_names: List of class names.
            
        Returns:
            dict: Evaluation metrics.
        """
        self.model.eval()
        
        # Initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        # For confusion matrix
        all_preds = []
        all_targets = []
        
        # For inference time measurement
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
                loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions and targets for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        test_loss = running_loss / total
        test_acc = 100.0 * correct / total
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Average Inference Time: {avg_inference_time:.2f} ms")
        
        # Create confusion matrix
        if class_names is not None:
            self._plot_confusion_matrix(all_targets, all_preds, class_names)
            
            # Generate classification report
            report = classification_report(
                all_targets, all_preds, 
                target_names=class_names, 
                output_dict=True
            )
            
            # Save classification report
            with open(os.path.join(self.results_dir, 'classification_report.json'), 'w') as f:
                json.dump(report, f, indent=4)
        
        # Save metrics
        metrics = {
            'accuracy': float(test_acc),
            'loss': float(test_loss),
            'inference_time_ms': float(avg_inference_time)
        }
        
        with open(os.path.join(self.results_dir, 'test_metrics.json'), 'w') as f:
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
        # Check if model is memristive
        if not hasattr(self.model, 'is_memristive') or not self.model.is_memristive:
            print("Warning: Model is not memristive. Energy analysis may not be accurate.")
        
        # Get model output size (number of classes)
        output_size = self.model.num_classes
        
        # Use memtorch's energy analysis if available
        try:
            # For memtorch models
            energy_analysis = memtorch.utils.analyze_energy(self.model)
            memristor_energy = energy_analysis['memristor_energy_nJ']
            gpu_energy = energy_analysis['gpu_energy_nJ']
            efficiency_ratio = energy_analysis['efficiency_ratio']
        except (AttributeError, ImportError):
            # Fallback to simplified calculation
            # Estimate energy consumption (nJ)
            memristor_energy = self._estimate_memristor_energy(input_size, output_size, batch_size)
            gpu_energy = self._estimate_gpu_energy(input_size, output_size, batch_size)
            efficiency_ratio = gpu_energy / memristor_energy
        
        print(f"Energy Efficiency Analysis:")
        print(f"  Memristor Energy: {memristor_energy:.2f} nJ")
        print(f"  GPU Energy: {gpu_energy:.2f} nJ")
        print(f"  Efficiency Ratio (GPU/Memristor): {efficiency_ratio:.2f}x")
        
        # Save energy metrics
        energy_metrics = {
            'memristor_energy_nJ': float(memristor_energy),
            'gpu_energy_nJ': float(gpu_energy),
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
        
        # Use memtorch's latency analysis if available
        try:
            # For memtorch models
            latency_analysis = memtorch.utils.analyze_latency(self.model)
            memristor_latency = latency_analysis['memristor_latency_ns']
            gpu_latency = latency_analysis['gpu_latency_ns']
            latency_reduction = latency_analysis['latency_reduction']
        except (AttributeError, ImportError):
            # Fallback to simplified calculation
            # Estimate latency (ns)
            memristor_latency = self._estimate_memristor_latency(input_size, output_size, batch_size, parallel_arrays)
            gpu_latency = self._estimate_gpu_latency(input_size, output_size, batch_size)
            latency_reduction = gpu_latency / memristor_latency
        
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
    
    def _estimate_memristor_energy(self, input_size, output_size, batch_size):
        """
        Estimate energy consumption for memristor-based computation.
        
        Args:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.
            batch_size (int): Batch size.
            
        Returns:
            float: Energy consumption in nJ.
        """
        # Energy parameters
        read_energy_per_cell_pJ = 0.1  # 0.1 pJ per cell read
        
        # Calculate total energy
        total_cells = input_size * output_size
        total_energy_pJ = total_cells * read_energy_per_cell_pJ * batch_size
        
        return total_energy_pJ / 1000  # Convert to nJ
    
    def _estimate_gpu_energy(self, input_size, output_size, batch_size):
        """
        Estimate energy consumption for GPU-based computation.
        
        Args:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.
            batch_size (int): Batch size.
            
        Returns:
            float: Energy consumption in nJ.
        """
        # Energy parameters
        energy_per_mac_pJ = 5.0  # 5 pJ per MAC operation on GPU
        
        # Calculate total energy
        total_macs = input_size * output_size * batch_size
        total_energy_pJ = total_macs * energy_per_mac_pJ
        
        return total_energy_pJ / 1000  # Convert to nJ
    
    def _estimate_memristor_latency(self, input_size, output_size, batch_size, parallel_arrays):
        """
        Estimate latency for memristor-based computation.
        
        Args:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.
            batch_size (int): Batch size.
            parallel_arrays (int): Number of parallel arrays.
            
        Returns:
            float: Latency in nanoseconds.
        """
        # Latency parameters
        read_latency_per_array_ns = 100  # 100 ns per array read operation
        
        # Calculate operations
        total_ops = input_size * output_size
        ops_per_array = (total_ops + parallel_arrays - 1) // parallel_arrays  # Ceiling division
        
        # Calculate latency
        latency_ns = ops_per_array * read_latency_per_array_ns * batch_size / parallel_arrays
        
        return latency_ns
    
    def _estimate_gpu_latency(self, input_size, output_size, batch_size):
        """
        Estimate latency for GPU-based computation.
        
        Args:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.
            batch_size (int): Batch size.
            
        Returns:
            float: Latency in nanoseconds.
        """
        # Latency parameters
        base_latency_ns = 5000  # 5 microseconds base latency
        latency_per_mac_ns = 0.01  # 0.01 ns per MAC operation on GPU
        
        # Calculate total latency
        total_macs = input_size * output_size * batch_size
        compute_latency_ns = total_macs * latency_per_mac_ns
        
        return base_latency_ns + compute_latency_ns
    
    def _plot_confusion_matrix(self, targets, predictions, class_names):
        """
        Plot confusion matrix.
        
        Args:
            targets (list): True labels.
            predictions (list): Predicted labels.
            class_names (list): List of class names.
        """
        # Calculate confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()
    
    def _plot_energy_comparison(self, memristor_energy, gpu_energy):
        """
        Plot energy comparison.
        
        Args:
            memristor_energy (float): Memristor energy consumption in nJ.
            gpu_energy (float): GPU energy consumption in nJ.
        """
        plt.figure(figsize=(8, 6))
        
        # Plot energy comparison
        labels = ['Memristor', 'GPU']
        energies = [memristor_energy, gpu_energy]
        
        plt.bar(labels, energies, color=['#2C7BB6', '#D7191C'])
        plt.ylabel('Energy Consumption (nJ)')
        plt.title('Energy Consumption Comparison')
        
        # Add values on top of bars
        for i, v in enumerate(energies):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        # Add efficiency ratio
        efficiency_ratio = gpu_energy / memristor_energy
        plt.text(0.5, 0.9, f'Efficiency Ratio: {efficiency_ratio:.2f}x',
                transform=plt.gca().transAxes, ha='center',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, 'energy_comparison.png'))
        plt.close()
    
    def _plot_latency_comparison(self, memristor_latency, gpu_latency):
        """
        Plot latency comparison.
        
        Args:
            memristor_latency (float): Memristor latency in ns.
            gpu_latency (float): GPU latency in ns.
        """
        plt.figure(figsize=(8, 6))
        
        # Convert to ms for better readability
        memristor_latency_ms = memristor_latency / 1e6
        gpu_latency_ms = gpu_latency / 1e6
        
        # Plot latency comparison
        labels = ['Memristor', 'GPU']
        latencies = [memristor_latency_ms, gpu_latency_ms]
        
        plt.bar(labels, latencies, color=['#2C7BB6', '#D7191C'])
        plt.ylabel('Latency (ms)')
        plt.title('Latency Comparison')
        
        # Add values on top of bars
        for i, v in enumerate(latencies):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        # Add latency reduction
        latency_reduction = gpu_latency / memristor_latency
        plt.text(0.5, 0.9, f'Latency Reduction: {latency_reduction:.2f}x',
                transform=plt.gca().transAxes, ha='center',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, 'latency_comparison.png'))
        plt.close()
    
    def plot_training_history(self, ex_situ_history=None, in_situ_history=None):
        """
        Plot training history.
        
        Args:
            ex_situ_history (dict): Ex-situ training history.
            in_situ_history (dict): In-situ training history.
        """
        if ex_situ_history is None and in_situ_history is None:
            print("No training history provided.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        
        if ex_situ_history is not None:
            plt.plot(ex_situ_history['train_loss'], label='Ex-situ Train Loss')
            plt.plot(ex_situ_history['val_loss'], label='Ex-situ Val Loss')
        
        if in_situ_history is not None:
            # If both histories are provided, offset in-situ epochs
            offset = len(ex_situ_history['train_loss']) if ex_situ_history is not None else 0
            epochs = list(range(offset, offset + len(in_situ_history['train_loss'])))
            
            plt.plot(epochs, in_situ_history['train_loss'], label='In-situ Train Loss')
            plt.plot(epochs, in_situ_history['val_loss'], label='In-situ Val Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 1, 2)
        
        if ex_situ_history is not None:
            plt.plot(ex_situ_history['train_acc'], label='Ex-situ Train Accuracy')
            plt.plot(ex_situ_history['val_acc'], label='Ex-situ Val Accuracy')
        
        if in_situ_history is not None:
            # If both histories are provided, offset in-situ epochs
            offset = len(ex_situ_history['train_acc']) if ex_situ_history is not None else 0
            epochs = list(range(offset, offset + len(in_situ_history['train_acc'])))
            
            plt.plot(epochs, in_situ_history['train_acc'], label='In-situ Train Accuracy')
            plt.plot(epochs, in_situ_history['val_acc'], label='In-situ Val Accuracy')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, 'training_history.png'))
        plt.close()

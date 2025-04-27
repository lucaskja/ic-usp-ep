"""
Unified evaluator for all MobileNetV2 variants.

This module provides a unified interface for evaluating all model variants:
- Base MobileNetV2
- MobileNetV2 with Mish activation
- MobileNetV2 with Mish and Triplet Attention
- MobileNetV2 with Mish, Triplet Attention, and CNSN
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import logging
from configs.model_configs import EVAL_CONFIG


class ModelEvaluator:
    """
    Evaluator for MobileNetV2 variants.
    """
    
    def __init__(self, model, data_loader, device=None):
        """
        Initialize the evaluator.
        
        Args:
            model (nn.Module): Model to evaluate
            data_loader (DataLoader): Data loader for evaluation
            device (torch.device, optional): Device to use for evaluation
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Get class names if available
        self.class_names = None
        if hasattr(data_loader.dataset, 'classes'):
            self.class_names = data_loader.dataset.classes
        elif hasattr(data_loader.dataset, 'dataset') and hasattr(data_loader.dataset.dataset, 'classes'):
            self.class_names = data_loader.dataset.dataset.classes
    
    def evaluate(self):
        """
        Evaluate the model on the data loader.
        
        Returns:
            dict: Evaluation results
        """
        all_preds = []
        all_targets = []
        all_probs = []
        running_loss = 0.0
        num_samples = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Get predictions
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Accumulate results
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update running statistics
                running_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        
        # Generate classification report
        if self.class_names is not None:
            report = classification_report(all_targets, all_preds, target_names=self.class_names, output_dict=True)
        else:
            report = classification_report(all_targets, all_preds, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'loss': running_loss / num_samples,
            'confusion_matrix': cm,
            'report': report,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        logging.info(f"Evaluation results: Accuracy={accuracy:.2f}%, Precision={precision*100:.2f}%, Recall={recall*100:.2f}%, F1={f1*100:.2f}%")
        
        return results
    
    def plot_confusion_matrix(self, results, save_path=None, figsize=(10, 8)):
        """
        Plot confusion matrix.
        
        Args:
            results (dict): Evaluation results from evaluate()
            save_path (str, optional): Path to save the plot
            figsize (tuple): Figure size
        """
        cm = results['confusion_matrix']
        
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else "auto",
            yticklabels=self.class_names if self.class_names else "auto"
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics(self, results, save_path=None, figsize=(12, 8)):
        """
        Plot per-class metrics.
        
        Args:
            results (dict): Evaluation results from evaluate()
            save_path (str, optional): Path to save the plot
            figsize (tuple): Figure size
        """
        report = results['report']
        
        # Extract per-class metrics
        classes = []
        precision = []
        recall = []
        f1 = []
        
        for cls, metrics in report.items():
            if isinstance(metrics, dict) and cls not in ['accuracy', 'macro avg', 'weighted avg']:
                classes.append(cls)
                precision.append(metrics['precision'] * 100)
                recall.append(metrics['recall'] * 100)
                f1.append(metrics['f1-score'] * 100)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1-score')
        
        plt.xlabel('Class')
        plt.ylabel('Score (%)')
        plt.title('Per-class Metrics')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_detailed_report(self, results, model_type, model_size_mb, output_dir):
        """
        Save detailed evaluation report to file.
        
        Args:
            results (dict): Evaluation results from evaluate()
            model_type (str): Type of model
            model_size_mb (float): Model size in MB
            output_dir (str): Output directory
        """
        report = results['report']
        report_path = os.path.join(output_dir, 'classification_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"Model: MobileNetV2 ({model_type})\n")
            f.write(f"Model Size: {model_size_mb:.2f} MB\n")
            f.write(f"Accuracy: {results['accuracy']:.2f}%\n\n")
            f.write("Classification Report:\n")
            
            # Write per-class metrics
            for cls, metrics in report.items():
                if isinstance(metrics, dict):
                    f.write(f"{cls}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"  {metric_name}: {value:.4f}\n")
                        else:
                            f.write(f"  {metric_name}: {value}\n")
                else:
                    f.write(f"{cls}: {metrics}\n")
        
        logging.info(f"Detailed report saved to {report_path}")

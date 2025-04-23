"""
Evaluator for MobileNetV2 model.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class MobileNetV2Evaluator:
    """
    Evaluator for MobileNetV2 model.
    """
    def __init__(self, model, val_loader, device):
        """
        Initialize evaluator.
        
        Args:
            model (nn.Module): Model to evaluate
            val_loader (DataLoader): Validation data loader
            device (torch.device): Device to use
        """
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Get class names from dataset
        if hasattr(val_loader.dataset, 'classes'):
            self.class_names = val_loader.dataset.classes
        elif hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'classes'):
            self.class_names = val_loader.dataset.dataset.classes
        else:
            self.class_names = [str(i) for i in range(model.model.classifier[1].out_features)]
    
    def evaluate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            dict: Evaluation results
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Update metrics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets).item()
                total_samples += inputs.size(0)
                
                # Store predictions and targets for later analysis
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100.0 * running_corrects / total_samples
        loss = running_loss / total_samples
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate classification report
        report = classification_report(
            all_targets, 
            all_preds, 
            target_names=self.class_names,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'confusion_matrix': cm,
            'report': report,
            'predictions': all_preds,
            'targets': all_targets,
            'class_names': self.class_names
        }
    
    def plot_confusion_matrix(self, results, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            results (dict): Evaluation results
            save_path (str): Path to save the plot
        """
        cm = results['confusion_matrix']
        class_names = results['class_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_metrics(self, results, save_path=None):
        """
        Plot metrics from classification report.
        
        Args:
            results (dict): Evaluation results
            save_path (str): Path to save the plot
        """
        report = results['report']
        class_names = results['class_names']
        
        # Extract metrics for each class
        precision = [report[cls]['precision'] for cls in class_names]
        recall = [report[cls]['recall'] for cls in class_names]
        f1_score = [report[cls]['f1-score'] for cls in class_names]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1_score, width, label='F1-score')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Classification Metrics')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

"""
Evaluator for base MobileNetV2 model.
"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MobileNetV2Evaluator:
    """
    Evaluator class for MobileNetV2 model.
    """
    def __init__(self, model, data_loader, device):
        """
        Initialize evaluator.
        
        Args:
            model (nn.Module): Model to evaluate
            data_loader (DataLoader): Data loader for evaluation
            device (torch.device): Device to evaluate on
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def evaluate(self):
        """
        Evaluate the model.
        
        Returns:
            dict: Evaluation results
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.data_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        class_names = self.data_loader.dataset.dataset.classes if hasattr(self.data_loader.dataset, 'dataset') else None
        report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate accuracy
        accuracy = (all_preds == all_targets).mean() * 100
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def plot_confusion_matrix(self, results, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            results (dict): Evaluation results
            save_path (str, optional): Path to save the plot
        """
        cm = results['confusion_matrix']
        class_names = self.data_loader.dataset.dataset.classes if hasattr(self.data_loader.dataset, 'dataset') else None
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_metrics(self, results, save_path=None):
        """
        Plot evaluation metrics.
        
        Args:
            results (dict): Evaluation results
            save_path (str, optional): Path to save the plot
        """
        report = results['report']
        
        # Extract metrics
        classes = []
        precision = []
        recall = []
        f1_score = []
        
        for cls, metrics in report.items():
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                classes.append(cls)
                precision.append(metrics['precision'])
                recall.append(metrics['recall'])
                f1_score.append(metrics['f1-score'])
        
        # Plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1_score, width, label='F1-score')
        
        ax.set_ylabel('Score')
        ax.set_title('Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

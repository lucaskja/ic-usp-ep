"""
Evaluation script for MobileNetV2 with Mish and Triplet Attention.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from models.mobilenetv2_mish_triplet import MobileNetV2MishTripletModel
import sys
sys.path.append('..')  # Add parent directory to path
from utils.data_utils import load_dataset
from utils.model_utils import get_model_size, print_model_summary
from utils.training_utils import setup_logging
from base_mobilenetv2.evaluation.evaluator import MobileNetV2Evaluator
from base_mobilenetv2.configs.default_config import DATA_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV2 with Mish and Triplet Attention')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='experiments/stage2_triplet',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda", "cpu")')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with smaller dataset')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up logging
    setup_logging('stage2_triplet_eval', 'logs')
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    _, val_loader, num_classes = load_dataset(
        args.data_dir,
        img_size=DATA_CONFIG['img_size'],
        batch_size=args.batch_size or DATA_CONFIG['batch_size'],
        val_split=DATA_CONFIG['val_split'],
        num_workers=DATA_CONFIG['num_workers'],
        debug=args.debug
    )
    print(f"Dataset loaded with {num_classes} classes")
    
    # Create model
    model = MobileNetV2MishTripletModel(num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Print model summary
    print_model_summary(model)
    model_size_mb = get_model_size(model)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    # Move model to device
    model = model.to(device)
    
    # Create evaluator
    evaluator = MobileNetV2Evaluator(model, val_loader, device)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluator.evaluate()
    
    # Print results
    print(f"Accuracy: {results['accuracy']:.2f}%")
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(results, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save metrics plot
    metrics_path = os.path.join(args.output_dir, 'metrics.png')
    evaluator.plot_metrics(results, save_path=metrics_path)
    print(f"Metrics plot saved to {metrics_path}")
    
    # Save detailed report
    report = results['report']
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: MobileNetV2 with Mish and Triplet Attention\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n")
        f.write(f"Accuracy: {results['accuracy']:.2f}%\n\n")
        f.write("Classification Report:\n")
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"{cls}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
            else:
                f.write(f"{cls}: {metrics:.4f}\n")
    
    print(f"Detailed report saved to {report_path}")


if __name__ == '__main__':
    main()

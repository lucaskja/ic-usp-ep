"""
Unified evaluation script for all MobileNetV2 variants.

This script consolidates the evaluation process for all model variants:
- Base MobileNetV2
- MobileNetV2 with Mish activation
- MobileNetV2 with Mish and Triplet Attention
- MobileNetV2 with Mish, Triplet Attention, and CNSN

It supports both standard and enhanced data preprocessing techniques.
"""
import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model factory
from utils.model_factory import create_model, load_model_from_checkpoint

# Import utilities
from utils.data_utils import load_dataset
from utils.enhanced_data_utils import load_enhanced_dataset
from utils.model_utils import get_model_size, print_model_summary
from utils.training_utils import setup_logging
from utils.evaluator import ModelEvaluator
from configs.model_configs import DATA_CONFIG, EVAL_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV2 variants for leaf disease classification')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Model selection
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'mish', 'triplet', 'cnsn'],
                        help='Model type to evaluate')
    
    # Data parameters
    parser.add_argument('--enhanced_preprocessing', action='store_true',
                        help='Use enhanced data preprocessing')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size (default: from config)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers (default: from config)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: experiments/{model_type}_eval)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    # Misc parameters
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with reduced dataset size')
    
    return parser.parse_args()


def get_data_config(args):
    """
    Get data configuration with command line overrides.
    
    Args:
        args (Namespace): Command line arguments
        
    Returns:
        dict: Data configuration
    """
    data_config = DATA_CONFIG.copy()
    
    # Override with command line arguments if provided
    if args.img_size is not None:
        data_config['img_size'] = args.img_size
    if args.batch_size is not None:
        data_config['batch_size'] = args.batch_size
    if args.num_workers is not None:
        data_config['num_workers'] = args.num_workers
        
    return data_config


def get_output_dir(args):
    """
    Get output directory with default based on model type.
    
    Args:
        args (Namespace): Command line arguments
        
    Returns:
        str: Output directory path
    """
    # Set default output directory if not provided
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join('experiments', f"{args.model_type}_eval")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up logging
    log_name = f"{args.model_type}_eval_{'enhanced' if args.enhanced_preprocessing else 'standard'}"
    setup_logging(log_name, args.log_dir)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get output directory
    output_dir = get_output_dir(args)
    
    # Get data configuration
    data_config = get_data_config(args)
    
    # Load dataset
    if args.enhanced_preprocessing:
        logging.info("Using enhanced data preprocessing")
        train_loader, val_loader, test_loader, num_classes = load_enhanced_dataset(
            args.data_dir,
            img_size=data_config['img_size'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            debug=args.debug
        )
        # Use test loader for evaluation
        eval_loader = test_loader
    else:
        logging.info("Using standard data preprocessing")
        train_loader, val_loader, test_loader, num_classes = load_dataset(
            args.data_dir,
            img_size=data_config['img_size'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            debug=args.debug
        )
        # Use test loader for evaluation
        eval_loader = test_loader
    
    logging.info(f"Dataset loaded with {num_classes} classes")
    
    # Load model from checkpoint
    try:
        model = load_model_from_checkpoint(args.model_type, num_classes, args.checkpoint, device)
        logging.info(f"Loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return
    
    # Print model summary
    print_model_summary(model)
    model_size_mb = get_model_size(model)
    logging.info(f"Model Size: {model_size_mb:.2f} MB")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, eval_loader, device)
    
    # Evaluate model
    logging.info("Evaluating model...")
    results = evaluator.evaluate()
    
    # Print results
    logging.info(f"Accuracy: {results['accuracy']:.2f}%")
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(results, save_path=cm_path)
    logging.info(f"Confusion matrix saved to {cm_path}")
    
    # Save metrics plot
    metrics_path = os.path.join(output_dir, 'metrics.png')
    evaluator.plot_metrics(results, save_path=metrics_path)
    logging.info(f"Metrics plot saved to {metrics_path}")
    
    # Save detailed report
    evaluator.save_detailed_report(results, args.model_type, model_size_mb, output_dir)
    
    # Print summary
    logging.info(f"Evaluation completed for {args.model_type} model")
    logging.info(f"Accuracy: {results['accuracy']:.5f}%")
    logging.info(f"Precision: {results['precision']:.5f}%")
    logging.info(f"Recall: {results['recall']:.5f}%")
    logging.info(f"F1: {results['f1']:.5f}%")
    logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

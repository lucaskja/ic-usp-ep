"""
Main script to run training and evaluation for different MobileNetV2 variants.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from mobilenetv2_improvements.utils.data_utils import load_dataset
from mobilenetv2_improvements.base_mobilenetv2.models.mobilenetv2 import MobileNetV2Model
from mobilenetv2_improvements.stage1_mish.models.mobilenetv2_mish import MobileNetV2MishModel
from mobilenetv2_improvements.stage2_triplet.models.mobilenetv2_triplet import MobileNetV2TripletModel
from mobilenetv2_improvements.stage3_cnsn.models.mobilenetv2_cnsn import MobileNetV2CNSNModel
from mobilenetv2_improvements.base_mobilenetv2.training.trainer import MobileNetV2Trainer
from mobilenetv2_improvements.base_mobilenetv2.evaluation.evaluator import MobileNetV2Evaluator
from mobilenetv2_improvements.base_mobilenetv2.configs.default_config import TRAIN_CONFIG, DATA_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MobileNetV2 Improvements')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'mish', 'triplet', 'cnsn'],
                        help='Model type to train/evaluate')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation or resuming training')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()


def get_model(model_type, num_classes, pretrained=True):
    """Get model based on model type."""
    if model_type == 'base':
        return MobileNetV2Model(num_classes, pretrained)
    elif model_type == 'mish':
        return MobileNetV2MishModel(num_classes, pretrained)
    elif model_type == 'triplet':
        return MobileNetV2TripletModel(num_classes, pretrained, use_mish=True)
    elif model_type == 'cnsn':
        return MobileNetV2CNSNModel(num_classes, pretrained, use_mish=True, use_triplet=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load dataset
    train_loader, val_loader, num_classes = load_dataset(
        args.data_dir,
        img_size=DATA_CONFIG['img_size'],
        batch_size=args.batch_size or DATA_CONFIG['batch_size'],
        val_split=DATA_CONFIG['val_split'],
        num_workers=DATA_CONFIG['num_workers']
    )
    print(f"Dataset loaded with {num_classes} classes")
    
    # Create model
    model = get_model(args.model_type, num_classes, args.pretrained)
    print(f"Created {args.model_type} model")
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Training or evaluation
    if args.mode == 'train':
        # Update training config with command line arguments
        train_config = TRAIN_CONFIG.copy()
        if args.epochs:
            train_config['epochs'] = args.epochs
        if args.lr:
            train_config['lr'] = args.lr
        
        # Create trainer
        trainer = MobileNetV2Trainer(
            model,
            train_loader,
            val_loader,
            train_config,
            device
        )
        
        # Train model
        print(f"Training {args.model_type} model for {train_config['epochs']} epochs")
        results = trainer.train(checkpoint_dir)
        print(f"Training completed. Best validation accuracy: {max(results['val_acc']):.2f}%")
        
    elif args.mode == 'evaluate':
        # Create evaluator
        evaluator = MobileNetV2Evaluator(
            model,
            val_loader,
            device
        )
        
        # Evaluate model
        print(f"Evaluating {args.model_type} model")
        results = evaluator.evaluate()
        print(f"Evaluation completed. Accuracy: {results['accuracy']:.2f}%")
        
        # Plot results
        plot_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
        evaluator.plot_confusion_matrix(results, save_path=plot_path)
        print(f"Confusion matrix saved to {plot_path}")
        
        metrics_path = os.path.join(checkpoint_dir, 'metrics.png')
        evaluator.plot_metrics(results, save_path=metrics_path)
        print(f"Metrics plot saved to {metrics_path}")


if __name__ == '__main__':
    main()

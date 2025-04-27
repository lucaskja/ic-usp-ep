"""
Utility functions for loading trained models from checkpoints.

This module provides functions to easily load models from saved checkpoints
for inference or continued training.
"""
import os
import torch
import logging
from utils.model_factory import load_model_from_checkpoint
from utils.model_utils import get_model_size, count_parameters


def load_model(model_type, checkpoint_path, num_classes, device=None):
    """
    Load a model from a checkpoint file.
    
    Args:
        model_type (str): Type of model ('base', 'mish', 'triplet', or 'cnsn')
        checkpoint_path (str): Path to the checkpoint file
        num_classes (int): Number of output classes
        device (torch.device, optional): Device to load the model to
        
    Returns:
        nn.Module: The loaded model
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        ValueError: If the model type is invalid
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    try:
        model = load_model_from_checkpoint(
            model_type=model_type,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            device=device
        )
        
        # Set to evaluation mode by default
        model.eval()
        
        # Get model info
        model_size_mb = get_model_size(model)
        num_params = count_parameters(model)
        
        logging.info(f"Model loaded successfully from {checkpoint_path}")
        logging.info(f"Model type: {model_type}")
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Model size: {model_size_mb:.2f} MB")
        logging.info(f"Number of parameters: {num_params:,}")
        logging.info(f"Device: {device}")
        
        return model
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
    

def load_best_model(model_type, num_classes, checkpoint_dir=None, device=None):
    """
    Load the best model from the default checkpoint directory.
    
    Args:
        model_type (str): Type of model ('base', 'mish', 'triplet', or 'cnsn')
        num_classes (int): Number of output classes
        checkpoint_dir (str, optional): Custom checkpoint directory
        device (torch.device, optional): Device to load the model to
        
    Returns:
        nn.Module: The loaded model
        
    Raises:
        FileNotFoundError: If the best.pth file doesn't exist
    """
    # Set default checkpoint directory if not provided
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join('checkpoints', model_type)
    
    # Path to best model
    best_model_path = os.path.join(checkpoint_dir, 'best.pth')
    
    # Load the model
    return load_model(model_type, best_model_path, num_classes, device)


def load_model_for_inference(model_type, checkpoint_path, num_classes, device=None):
    """
    Load a model specifically for inference (with torch.no_grad() context).
    
    Args:
        model_type (str): Type of model ('base', 'mish', 'triplet', or 'cnsn')
        checkpoint_path (str): Path to the checkpoint file
        num_classes (int): Number of output classes
        device (torch.device, optional): Device to load the model to
        
    Returns:
        function: A function that takes input tensors and returns predictions
    """
    # Load the model
    model = load_model(model_type, checkpoint_path, num_classes, device)
    
    # Create inference function
    def inference_fn(inputs):
        """
        Run inference on input tensors.
        
        Args:
            inputs (torch.Tensor): Input tensor or batch of tensors
            
        Returns:
            tuple: (predictions, probabilities)
                - predictions: Class indices (argmax)
                - probabilities: Softmax probabilities
        """
        # Ensure inputs are on the correct device
        if inputs.device != device:
            inputs = inputs.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities
    
    return inference_fn


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Load a trained model from checkpoint')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['base', 'mish', 'triplet', 'cnsn'],
                        help='Type of model to load')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the checkpoint file')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes in the model')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to load the model on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s [%(levelname)s] - %(message)s')
    
    # Load model
    model = load_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=torch.device(args.device) if args.device else None
    )
    
    print("Model loaded successfully and ready for use!")

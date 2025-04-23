"""
Utility functions for model operations such as size calculation, 
parameter counting, and other model-related metrics.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Union, List


def get_model_size(model: nn.Module) -> float:
    """
    Calculate the size of a model in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Size of the model in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_in_mb = (param_size + buffer_size) / 1024**2
    return size_in_mb


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict: Dictionary containing model information
    """
    return {
        'size_mb': get_model_size(model),
        'parameters': count_parameters(model),
        'layers': len(list(model.modules())),
    }


def print_model_summary(model: nn.Module) -> None:
    """
    Print a summary of the model architecture and size.
    
    Args:
        model: PyTorch model
    """
    info = get_model_info(model)
    print(f"Model Summary:")
    print(f"  - Size: {info['size_mb']:.2f} MB")
    print(f"  - Parameters: {info['parameters']:,}")
    print(f"  - Layers: {info['layers']}")
    print(f"  - Device: {next(model.parameters()).device}")

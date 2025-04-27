"""
Model factory for creating different MobileNetV2 variants.

This module provides a unified interface for creating all model variants:
- Base MobileNetV2
- MobileNetV2 with Mish activation
- MobileNetV2 with Mish and Triplet Attention
- MobileNetV2 with Mish, Triplet Attention, and CNSN
"""
import logging
import torch
import torch.nn as nn

from base_mobilenetv2.models.mobilenetv2 import create_mobilenetv2
from stage1_mish.models.mobilenetv2_mish import create_mobilenetv2_mish
from stage2_triplet.models.mobilenetv2_triplet_fixed import create_mobilenetv2_triplet
from stage3_cnsn.models.mobilenetv2_cnsn_fixed import create_mobilenetv2_cnsn
from configs.model_configs import get_model_config


def create_model(model_type, num_classes, pretrained=True, **kwargs):
    """
    Create model based on type.
    
    Args:
        model_type (str): Type of model to create ('base', 'mish', 'triplet', or 'cnsn')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        **kwargs: Additional model-specific parameters
        
    Returns:
        nn.Module: The created model
    """
    # Get model configuration
    try:
        model_config = get_model_config(model_type)
        model_name = model_config['name']
        model_params = model_config['params'].copy()
        
        # Override with any provided kwargs
        model_params.update(kwargs)
    except ValueError as e:
        logging.error(f"Error getting model configuration: {e}")
        model_name = f"MobileNetV2 ({model_type})"
        model_params = kwargs
    
    # Create the model based on type
    if model_type == 'base':
        model = create_mobilenetv2(num_classes, pretrained, **model_params)
        logging.info(f"Created {model_name}")
    elif model_type == 'mish':
        model = create_mobilenetv2_mish(num_classes, pretrained, **model_params)
        logging.info(f"Created {model_name}")
    elif model_type == 'triplet':
        model = create_mobilenetv2_triplet(num_classes, pretrained, **model_params)
        logging.info(f"Created {model_name}")
    elif model_type == 'cnsn':
        model = create_mobilenetv2_cnsn(num_classes, pretrained, **model_params)
        logging.info(f"Created {model_name}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def load_model_from_checkpoint(model_type, num_classes, checkpoint_path, device=None):
    """
    Load a model from a checkpoint.
    
    Args:
        model_type (str): Type of model ('base', 'mish', 'triplet', or 'cnsn')
        num_classes (int): Number of output classes
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device, optional): Device to load the model to
        
    Returns:
        nn.Module: The loaded model
    """
    # Create the model
    model = create_model(model_type, num_classes, pretrained=False)
    
    # Set device if provided
    if device is not None:
        model = model.to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device if device else torch.device('cpu'))
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise
    
    return model

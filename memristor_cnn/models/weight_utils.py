"""
Weight utility functions for memristor-based neural networks.

This module provides utility functions for weight quantization, mapping,
and error compensation in memristor-based neural networks.
"""

import torch
import numpy as np


def quantize_15_level(weights):
    """
    Quantize weights to 15 discrete levels.
    
    Args:
        weights (torch.Tensor): Input weight tensor.
        
    Returns:
        torch.Tensor: Quantized weight tensor with 15 discrete levels.
    """
    # Get the maximum absolute value
    max_val = torch.max(torch.abs(weights))
    if max_val == 0:
        return weights
        
    # Scale to 7 positive levels + 7 negative levels + zero
    scale = 7 / max_val
    
    # Quantize
    quantized = torch.round(weights * scale) / scale
    
    return quantized


def map_to_differential_pairs(weights):
    """
    Map weights to differential conductance pairs.
    
    Args:
        weights (torch.Tensor): Input weight tensor.
        
    Returns:
        tuple: Positive and negative conductance tensors.
    """
    # Split into positive and negative components
    positive_conductance = torch.clamp(weights, min=0)
    negative_conductance = torch.clamp(-weights, min=0)
    
    return positive_conductance, negative_conductance


def compensate_device_variations(conductance_values, variation_map=None, variation_std=0.05):
    """
    Compensate for device-to-device variations.
    
    Args:
        conductance_values (torch.Tensor): Conductance values.
        variation_map (torch.Tensor, optional): Pre-defined variation map.
        variation_std (float, optional): Standard deviation of variations.
        
    Returns:
        torch.Tensor: Compensated conductance values.
    """
    if variation_map is None:
        # Generate random variation map with normal distribution
        variation_map = torch.randn_like(conductance_values) * variation_std
    
    # Apply compensation
    compensated = conductance_values * (1 + variation_map)
    
    return compensated


def apply_closed_loop_programming(target_conductance, actual_conductance, learning_rate=0.1, max_iterations=10):
    """
    Apply closed-loop programming to match target conductance values.
    
    Args:
        target_conductance (torch.Tensor): Target conductance values.
        actual_conductance (torch.Tensor): Actual conductance values.
        learning_rate (float): Learning rate for adjustments.
        max_iterations (int): Maximum number of programming iterations.
        
    Returns:
        torch.Tensor: Updated conductance values.
    """
    current_conductance = actual_conductance.clone()
    
    for _ in range(max_iterations):
        # Calculate error
        error = target_conductance - current_conductance
        
        # Stop if error is small enough
        if torch.max(torch.abs(error)) < 1e-4:
            break
            
        # Update conductance
        current_conductance += learning_rate * error
    
    return current_conductance


def verify_after_write(programmed_conductance, target_conductance, tolerance=0.05):
    """
    Verify programmed conductance values after writing.
    
    Args:
        programmed_conductance (torch.Tensor): Programmed conductance values.
        target_conductance (torch.Tensor): Target conductance values.
        tolerance (float): Acceptable error tolerance.
        
    Returns:
        bool: True if verification passes, False otherwise.
    """
    # Calculate relative error
    error = torch.abs(programmed_conductance - target_conductance)
    relative_error = error / (target_conductance + 1e-8)  # Avoid division by zero
    
    # Check if all errors are within tolerance
    return torch.all(relative_error < tolerance).item()

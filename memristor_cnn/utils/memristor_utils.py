"""
Utility functions for memristor-based neural networks.
"""

import torch
import numpy as np
import time


def simulate_memristor_programming(weights, closed_loop=False, verify=True, max_attempts=3):
    """
    Simulate memristor programming with optional closed-loop and verification.
    
    Args:
        weights (torch.Tensor): Weight tensor to program.
        closed_loop (bool): Whether to use closed-loop programming.
        verify (bool): Whether to verify after programming.
        max_attempts (int): Maximum number of programming attempts.
        
    Returns:
        tuple: Programmed weights and programming statistics.
    """
    # Import weight utilities
    from ..models.weight_utils import (
        quantize_15_level, 
        map_to_differential_pairs, 
        compensate_device_variations,
        apply_closed_loop_programming,
        verify_after_write
    )
    
    device = weights.device
    original_shape = weights.shape
    
    # Flatten weights for processing
    flat_weights = weights.reshape(-1)
    
    # Apply 15-level quantization
    quantized_weights = quantize_15_level(flat_weights)
    
    # Map to differential conductance pairs
    positive_conductance, negative_conductance = map_to_differential_pairs(quantized_weights)
    
    # Simulate programming time (50ns per cell)
    num_cells = flat_weights.numel()
    programming_time_ns = num_cells * 50  # 50ns per cell
    
    # Add variation to simulate real device behavior
    variation_std = 0.05  # 5% variation
    
    # Simulate programming with device variations
    programmed_positive = compensate_device_variations(
        positive_conductance, variation_std=variation_std
    )
    programmed_negative = compensate_device_variations(
        negative_conductance, variation_std=variation_std
    )
    
    # Apply closed-loop programming if requested
    if closed_loop:
        attempts = 0
        verified = False
        
        while attempts < max_attempts and not verified:
            # Apply closed-loop programming
            programmed_positive = apply_closed_loop_programming(
                positive_conductance, programmed_positive
            )
            programmed_negative = apply_closed_loop_programming(
                negative_conductance, programmed_negative
            )
            
            # Additional time for closed-loop programming
            programming_time_ns += num_cells * 30  # 30ns per cell for verification
            
            # Verify if requested
            if verify:
                verified_positive = verify_after_write(
                    programmed_positive, positive_conductance
                )
                verified_negative = verify_after_write(
                    programmed_negative, negative_conductance
                )
                verified = verified_positive and verified_negative
            else:
                verified = True
                
            attempts += 1
    
    # Reconstruct weights from differential pairs
    programmed_weights = programmed_positive - programmed_negative
    
    # Calculate programming accuracy
    mse = torch.mean((quantized_weights - programmed_weights) ** 2).item()
    max_error = torch.max(torch.abs(quantized_weights - programmed_weights)).item()
    programming_accuracy = 100 * (1 - mse / (torch.mean(quantized_weights ** 2).item() + 1e-8))
    
    # Reshape back to original shape
    programmed_weights = programmed_weights.reshape(original_shape)
    
    # Return programmed weights and statistics
    stats = {
        'programming_accuracy': programming_accuracy,
        'max_error': max_error,
        'programming_time_ns': programming_time_ns,
        'attempts': 1 if not closed_loop else attempts
    }
    
    return programmed_weights, stats


def threshold_based_update(weights, gradients, learning_rate=0.001, threshold=0.1):
    """
    Apply threshold-based weight update for in-situ training.
    
    Args:
        weights (torch.Tensor): Current weight values.
        gradients (torch.Tensor): Gradient values.
        learning_rate (float): Learning rate for updates.
        threshold (float): Threshold for weight updates.
        
    Returns:
        torch.Tensor: Updated weights.
    """
    # Apply threshold
    mask = torch.abs(gradients) > threshold
    
    # Update only weights that exceed the threshold
    updated_weights = weights.clone()
    updated_weights[mask] -= learning_rate * gradients[mask]
    
    return updated_weights


def calculate_memristor_energy(input_size, output_size, batch_size=1):
    """
    Calculate energy consumption for memristor-based computation.
    
    Args:
        input_size (int): Number of input neurons.
        output_size (int): Number of output neurons.
        batch_size (int): Batch size.
        
    Returns:
        float: Energy consumption in nJ.
    """
    # Energy parameters
    read_energy_per_cell_pJ = 0.1  # 0.1 pJ per cell read
    
    # Calculate total energy
    total_cells = input_size * output_size
    total_energy_pJ = total_cells * read_energy_per_cell_pJ * batch_size
    
    return total_energy_pJ / 1000  # Convert to nJ

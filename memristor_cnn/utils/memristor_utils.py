"""
Utility functions for memristor-based operations.
"""

import torch
import numpy as np
import time


def simulate_memristor_programming(weights, conductance_levels=15, read_voltage=0.2, 
                                  programming_pulse_width=50, closed_loop=True):
    """
    Simulate memristor programming with realistic constraints.
    
    Args:
        weights (torch.Tensor): Weight tensor to program.
        conductance_levels (int): Number of discrete conductance levels.
        read_voltage (float): Read voltage in volts.
        programming_pulse_width (float): Programming pulse width in nanoseconds.
        closed_loop (bool): Whether to use closed-loop programming.
        
    Returns:
        tuple: Programmed weights and programming statistics.
    """
    # Start timing
    start_time = time.time()
    
    # Get original weight shape
    original_shape = weights.shape
    flat_weights = weights.reshape(-1)
    
    # Normalize weights to [0, 1] range for conductance mapping
    if flat_weights.max() != flat_weights.min():
        normalized_weights = (flat_weights - flat_weights.min()) / (flat_weights.max() - flat_weights.min())
    else:
        normalized_weights = torch.zeros_like(flat_weights)
    
    # Quantize to discrete conductance levels
    quantized_indices = torch.round(normalized_weights * (conductance_levels - 1))
    
    # Simulate programming variations (device-to-device and cycle-to-cycle)
    device_variation = 0.05  # 5% device-to-device variation
    cycle_variation = 0.02   # 2% cycle-to-cycle variation
    
    # Add variations to quantized indices
    if closed_loop:
        # Closed-loop programming reduces variations through feedback
        device_noise = torch.randn_like(quantized_indices) * device_variation * 0.3
        cycle_noise = torch.randn_like(quantized_indices) * cycle_variation * 0.3
    else:
        # Open-loop programming has full variations
        device_noise = torch.randn_like(quantized_indices) * device_variation
        cycle_noise = torch.randn_like(quantized_indices) * cycle_variation
    
    noisy_indices = quantized_indices + device_noise + cycle_noise
    
    # Clamp to valid range
    noisy_indices = torch.clamp(noisy_indices, 0, conductance_levels - 1)
    
    # Convert back to weight scale
    if flat_weights.max() != flat_weights.min():
        weight_scale = flat_weights.max() - flat_weights.min()
        weight_min = flat_weights.min()
        programmed_weights = (noisy_indices / (conductance_levels - 1)) * weight_scale + weight_min
    else:
        programmed_weights = torch.zeros_like(noisy_indices)
    
    # Reshape back to original shape
    programmed_weights = programmed_weights.reshape(original_shape)
    
    # Calculate programming time (simplified model)
    # Assume each level takes one pulse to program
    avg_levels_programmed = torch.mean(quantized_indices) + 1  # +1 because level 0 also needs programming
    total_pulses = avg_levels_programmed * flat_weights.numel()
    programming_time_ns = total_pulses * programming_pulse_width
    
    # Calculate programming accuracy
    if closed_loop:
        verify_cycles = 3  # Number of verify cycles in closed-loop programming
        programming_time_ns *= verify_cycles
        
        # Closed-loop improves accuracy
        accuracy = 1.0 - torch.mean(torch.abs(programmed_weights - weights) / 
                                   (torch.max(weights) - torch.min(weights) + 1e-8)).item()
    else:
        accuracy = 1.0 - torch.mean(torch.abs(programmed_weights - weights) / 
                                   (torch.max(weights) - torch.min(weights) + 1e-8)).item()
    
    # End timing
    elapsed_time = time.time() - start_time
    
    # Return programmed weights and statistics
    stats = {
        'programming_time_ns': float(programming_time_ns),
        'programming_accuracy': float(accuracy * 100),  # as percentage
        'simulation_time_s': float(elapsed_time),
        'conductance_levels_used': int(conductance_levels)
    }
    
    return programmed_weights, stats


def calculate_memristor_energy(input_size, output_size, batch_size=1, 
                              read_voltage=0.2, avg_conductance=1e-6):
    """
    Calculate energy consumption for memristor-based matrix multiplication.
    
    Args:
        input_size (int): Input dimension.
        output_size (int): Output dimension.
        batch_size (int): Batch size.
        read_voltage (float): Read voltage in volts.
        avg_conductance (float): Average conductance in Siemens.
        
    Returns:
        float: Energy consumption in nanojoules.
    """
    # Energy = V^2 * G * t
    # Assuming operation time of 10ns per MAC operation
    operation_time_ns = 10
    
    # Calculate total number of MAC operations
    mac_operations = input_size * output_size * batch_size
    
    # Calculate energy per operation (V^2 * G * t)
    energy_per_op = read_voltage**2 * avg_conductance * operation_time_ns * 1e9  # in nJ
    
    # Total energy
    total_energy_nj = energy_per_op * mac_operations
    
    return total_energy_nj


def calculate_memristor_latency(input_size, output_size, batch_size=1, 
                               parallel_arrays=1, read_time_ns=10):
    """
    Calculate latency for memristor-based matrix multiplication.
    
    Args:
        input_size (int): Input dimension.
        output_size (int): Output dimension.
        batch_size (int): Batch size.
        parallel_arrays (int): Number of parallel arrays.
        read_time_ns (float): Read time in nanoseconds.
        
    Returns:
        float: Latency in nanoseconds.
    """
    # Calculate operations per array
    ops_per_array = (input_size * output_size * batch_size) / parallel_arrays
    
    # Calculate latency
    latency_ns = ops_per_array * read_time_ns
    
    return latency_ns


def compare_energy_efficiency(memristor_energy, gpu_energy):
    """
    Compare energy efficiency between memristor and GPU.
    
    Args:
        memristor_energy (float): Memristor energy consumption in nanojoules.
        gpu_energy (float): GPU energy consumption in nanojoules.
        
    Returns:
        float: Energy efficiency ratio (GPU/Memristor).
    """
    return gpu_energy / memristor_energy


def compare_latency(memristor_latency, gpu_latency):
    """
    Compare latency between memristor and GPU.
    
    Args:
        memristor_latency (float): Memristor latency in nanoseconds.
        gpu_latency (float): GPU latency in nanoseconds.
        
    Returns:
        float: Latency reduction ratio (GPU/Memristor).
    """
    return gpu_latency / memristor_latency

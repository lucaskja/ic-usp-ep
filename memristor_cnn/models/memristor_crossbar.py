"""
Memristor Crossbar Array implementation for in-memory computing.
"""

import numpy as np
import torch
import torch.nn as nn


class MemristorCrossbar(nn.Module):
    """
    Memristor Crossbar Array for in-memory computing.
    
    This class implements a memristor crossbar array with differential conductance pairs
    for weight representation. The crossbar is designed to perform matrix-vector multiplication
    operations in the analog domain.
    
    Attributes:
        rows (int): Number of rows in the crossbar array.
        cols (int): Number of columns in the crossbar array.
        conductance_levels (int): Number of discrete conductance levels.
        read_voltage (float): Voltage used for read operations (in V).
        programming_pulse_width (float): Width of programming pulses (in ns).
        conductance_positive (torch.Tensor): Positive conductance values.
        conductance_negative (torch.Tensor): Negative conductance values.
        device (torch.device): Device to store the tensors.
    """
    
    def __init__(self, rows=128, cols=16, conductance_levels=15, 
                 read_voltage=0.2, programming_pulse_width=50,
                 device=None):
        """
        Initialize the memristor crossbar array.
        
        Args:
            rows (int): Number of rows in the crossbar array.
            cols (int): Number of columns in the crossbar array.
            conductance_levels (int): Number of discrete conductance levels.
            read_voltage (float): Voltage used for read operations (in V).
            programming_pulse_width (float): Width of programming pulses (in ns).
            device (torch.device): Device to store the tensors.
        """
        super(MemristorCrossbar, self).__init__()
        
        self.rows = rows
        self.cols = cols
        self.conductance_levels = conductance_levels
        self.read_voltage = read_voltage
        self.programming_pulse_width = programming_pulse_width
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize conductance values
        self.conductance_positive = torch.zeros((rows, cols), device=self.device)
        self.conductance_negative = torch.zeros((rows, cols), device=self.device)
        
        # Register buffers to ensure they're saved in the state dict
        self.register_buffer('_conductance_positive', self.conductance_positive)
        self.register_buffer('_conductance_negative', self.conductance_negative)
        
    def program_weights(self, weights):
        """
        Program the memristor crossbar with the given weights using differential pairs.
        
        Args:
            weights (torch.Tensor): Weight tensor to program into the crossbar.
                                   Shape should be compatible with the crossbar dimensions.
        
        Returns:
            tuple: Quantized positive and negative conductance values.
        """
        # Ensure weights have the right shape and are on the correct device
        weights = weights.to(self.device).reshape(self.rows, self.cols)
        
        # Split weights into positive and negative components
        weights_positive = torch.clamp(weights, min=0)
        weights_negative = torch.clamp(-weights, min=0)
        
        # Quantize to discrete conductance levels
        # Scale to [0, conductance_levels-1] range
        max_weight = max(weights_positive.max().item(), weights_negative.max().item())
        if max_weight > 0:
            scale_factor = (self.conductance_levels - 1) / max_weight
            conductance_positive = torch.round(weights_positive * scale_factor)
            conductance_negative = torch.round(weights_negative * scale_factor)
        else:
            conductance_positive = torch.zeros_like(weights_positive)
            conductance_negative = torch.zeros_like(weights_negative)
        
        # Store the quantized conductance values
        self._conductance_positive = conductance_positive
        self._conductance_negative = conductance_negative
        
        return conductance_positive, conductance_negative
    
    def forward(self, input_voltages):
        """
        Perform matrix-vector multiplication using the memristor crossbar.
        
        Args:
            input_voltages (torch.Tensor): Input voltages to apply to the crossbar.
                                          Shape should be (batch_size, rows).
        
        Returns:
            torch.Tensor: Output currents from the crossbar.
        """
        # Ensure input has the right shape
        batch_size = input_voltages.shape[0]
        input_voltages = input_voltages.reshape(batch_size, self.rows)
        
        # Scale input voltages by read voltage
        scaled_inputs = input_voltages * self.read_voltage
        
        # Perform matrix-vector multiplication for both conductance pairs
        output_positive = torch.matmul(scaled_inputs, self._conductance_positive)
        output_negative = torch.matmul(scaled_inputs, self._conductance_negative)
        
        # Differential output (positive - negative)
        output_currents = output_positive - output_negative
        
        return output_currents
    
    def get_programmed_weights(self):
        """
        Get the effective weights programmed in the crossbar.
        
        Returns:
            torch.Tensor: Effective weights (positive - negative conductance).
        """
        return self._conductance_positive - self._conductance_negative
    
    def get_conductance_pair(self):
        """
        Get the conductance pairs.
        
        Returns:
            tuple: Positive and negative conductance values.
        """
        return self._conductance_positive, self._conductance_negative

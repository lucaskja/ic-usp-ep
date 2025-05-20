"""
Memristor Processing Element (PE) implementation.
"""

import torch
import torch.nn as nn
from .memristor_crossbar import MemristorCrossbar


class MemristorPE(nn.Module):
    """
    Memristor Processing Element (PE) for hardware-accelerated neural network operations.
    
    This class implements a processing element that contains multiple memristor crossbar arrays
    and performs parallel operations for convolutional or fully-connected layers.
    
    Attributes:
        name (str): Name identifier for the PE.
        num_arrays (int): Number of crossbar arrays in this PE.
        array_rows (int): Number of rows in each crossbar array.
        array_cols (int): Number of columns in each crossbar array.
        crossbars (nn.ModuleList): List of memristor crossbar arrays.
        adc_resolution (int): Resolution of the analog-to-digital converters.
        dac_resolution (int): Resolution of the digital-to-analog converters.
    """
    
    def __init__(self, name, num_arrays=1, array_rows=128, array_cols=16, 
                 conductance_levels=15, read_voltage=0.2, programming_pulse_width=50,
                 adc_resolution=8, dac_resolution=8, device=None):
        """
        Initialize the memristor processing element.
        
        Args:
            name (str): Name identifier for the PE.
            num_arrays (int): Number of crossbar arrays in this PE.
            array_rows (int): Number of rows in each crossbar array.
            array_cols (int): Number of columns in each crossbar array.
            conductance_levels (int): Number of discrete conductance levels.
            read_voltage (float): Voltage used for read operations (in V).
            programming_pulse_width (float): Width of programming pulses (in ns).
            adc_resolution (int): Resolution of the analog-to-digital converters.
            dac_resolution (int): Resolution of the digital-to-analog converters.
            device (torch.device): Device to store the tensors.
        """
        super(MemristorPE, self).__init__()
        
        self.name = name
        self.num_arrays = num_arrays
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.adc_resolution = adc_resolution
        self.dac_resolution = dac_resolution
        
        # Create multiple crossbar arrays
        self.crossbars = nn.ModuleList([
            MemristorCrossbar(
                rows=array_rows,
                cols=array_cols,
                conductance_levels=conductance_levels,
                read_voltage=read_voltage,
                programming_pulse_width=programming_pulse_width,
                device=device
            ) for _ in range(num_arrays)
        ])
        
    def program_weights(self, weights_list):
        """
        Program weights into the memristor crossbar arrays.
        
        Args:
            weights_list (list or torch.Tensor): List of weight tensors or a single tensor
                                to program into the crossbars.
        
        Returns:
            list: List of tuples containing quantized conductance pairs for each array.
        """
        # Check for empty list
        if isinstance(weights_list, list) and len(weights_list) == 0:
            raise IndexError("Empty weights list provided")
            
        # Handle case where a single weight tensor is provided
        if not isinstance(weights_list, list):
            weights_list = [weights_list]
            
        # If fewer tensors than arrays, pad with zeros
        if len(weights_list) < self.num_arrays:
            device = weights_list[0].device
            for _ in range(self.num_arrays - len(weights_list)):
                weights_list.append(torch.zeros(self.array_rows, self.array_cols, device=device))
        
        # If more tensors than arrays, use only the first num_arrays tensors
        if len(weights_list) > self.num_arrays:
            weights_list = weights_list[:self.num_arrays]
        
        conductance_pairs = []
        for i, weights in enumerate(weights_list):
            conductance_pair = self.crossbars[i].program_weights(weights)
            conductance_pairs.append(conductance_pair)
            
        return conductance_pairs
    
    def forward(self, input_batch_list):
        """
        Process inputs through the memristor PE.
        
        Args:
            input_batch_list (list or torch.Tensor): Input tensor or list of input tensors for each crossbar array.
                                    If a single tensor is provided, it will be used for all arrays.
        
        Returns:
            list: List of output tensors from each crossbar array.
        """
        # Handle case where a single input tensor is provided
        if not isinstance(input_batch_list, list):
            input_batch_list = [input_batch_list] * self.num_arrays
            
        # If fewer inputs than arrays, pad with zeros
        if len(input_batch_list) < self.num_arrays:
            batch_size = input_batch_list[0].size(0)
            for _ in range(self.num_arrays - len(input_batch_list)):
                input_batch_list.append(torch.zeros(batch_size, self.array_rows, 
                                                  device=input_batch_list[0].device))
        
        # If more inputs than arrays, use only the first num_arrays inputs
        if len(input_batch_list) > self.num_arrays:
            input_batch_list = input_batch_list[:self.num_arrays]
        
        outputs = []
        for i, input_batch in enumerate(input_batch_list):
            # Apply DAC quantization (simulate limited resolution)
            if self.dac_resolution < 32:  # Skip if high resolution
                input_batch = self._simulate_dac(input_batch)
                
            # Process through crossbar
            output = self.crossbars[i](input_batch)
            
            # Apply ADC quantization (simulate limited resolution)
            if self.adc_resolution < 32:  # Skip if high resolution
                output = self._simulate_adc(output)
                
            outputs.append(output)
            
        return outputs
    
    def _simulate_dac(self, input_tensor):
        """
        Simulate digital-to-analog conversion with limited resolution.
        
        Args:
            input_tensor (torch.Tensor): Input tensor to quantize.
            
        Returns:
            torch.Tensor: Quantized tensor simulating DAC output.
        """
        # Scale to [0, 2^resolution - 1]
        max_val = torch.max(torch.abs(input_tensor))
        if max_val > 0:
            scale = (2**self.dac_resolution - 1) / max_val
            quantized = torch.round(input_tensor * scale) / scale
            return quantized
        return input_tensor
    
    def _simulate_adc(self, output_tensor):
        """
        Simulate analog-to-digital conversion with limited resolution.
        
        Args:
            output_tensor (torch.Tensor): Output tensor to quantize.
            
        Returns:
            torch.Tensor: Quantized tensor simulating ADC output.
        """
        # Scale to [0, 2^resolution - 1]
        max_val = torch.max(torch.abs(output_tensor))
        if max_val > 0:
            scale = (2**self.adc_resolution - 1) / max_val
            quantized = torch.round(output_tensor * scale) / scale
            return quantized
        return output_tensor
    
    def get_programmed_weights(self):
        """
        Get the effective weights programmed in all crossbar arrays.
        
        Returns:
            list: List of effective weight tensors from each crossbar.
        """
        return [crossbar.get_programmed_weights() for crossbar in self.crossbars]

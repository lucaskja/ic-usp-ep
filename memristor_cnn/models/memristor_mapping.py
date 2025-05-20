"""
Memristor Mapping module for mapping neural network layers to memristor arrays.
"""

import torch
import torch.nn as nn
import numpy as np
from .memristor_pe import MemristorPE


class MemristorMapping:
    """
    Maps neural network layers to memristor processing elements.
    
    This class handles the mapping of convolutional and fully-connected layers
    to memristor crossbar arrays, including weight distribution and data flow.
    
    Attributes:
        processing_elements (dict): Dictionary of memristor processing elements.
        layer_to_pe_mapping (dict): Mapping from layer names to PE names.
        weight_mapping (dict): Detailed mapping of weights to crossbar arrays.
    """
    
    def __init__(self):
        """Initialize the memristor mapping."""
        self.processing_elements = {}
        self.layer_to_pe_mapping = {}
        self.weight_mapping = {}
        
    def create_processing_element(self, pe_name, num_arrays=1, array_rows=128, array_cols=16, 
                                 conductance_levels=15, read_voltage=0.2, programming_pulse_width=50,
                                 adc_resolution=8, dac_resolution=8, device=None):
        """
        Create a new memristor processing element.
        
        Args:
            pe_name (str): Name identifier for the PE.
            num_arrays (int): Number of crossbar arrays in this PE.
            array_rows (int): Number of rows in each crossbar array.
            array_cols (int): Number of columns in each crossbar array.
            conductance_levels (int): Number of discrete conductance levels.
            read_voltage (float): Voltage used for read operations (in V).
            programming_pulse_width (float): Width of programming pulses (in ns).
            adc_resolution (int): Resolution of the analog-to-digital converters.
            dac_resolution (int): Resolution of the digital-to-analog converters.
            device (torch.device): Device to store the tensors.
            
        Returns:
            MemristorPE: The created processing element.
        """
        pe = MemristorPE(
            name=pe_name,
            num_arrays=num_arrays,
            array_rows=array_rows,
            array_cols=array_cols,
            conductance_levels=conductance_levels,
            read_voltage=read_voltage,
            programming_pulse_width=programming_pulse_width,
            adc_resolution=adc_resolution,
            dac_resolution=dac_resolution,
            device=device
        )
        
        self.processing_elements[pe_name] = pe
        return pe
    
    def map_conv_layer(self, layer_name, conv_layer, pe_names):
        """
        Map a convolutional layer to one or more processing elements.
        
        Args:
            layer_name (str): Name identifier for the layer.
            conv_layer (nn.Conv2d): Convolutional layer to map.
            pe_names (list): List of PE names to map this layer to.
            
        Returns:
            dict: Mapping details for this layer.
        """
        if not all(pe_name in self.processing_elements for pe_name in pe_names):
            missing = [pe for pe in pe_names if pe not in self.processing_elements]
            raise ValueError(f"Processing elements not found: {missing}")
        
        # Get layer parameters
        out_channels, in_channels, kernel_h, kernel_w = conv_layer.weight.shape
        
        # Calculate total weight elements
        total_weights = out_channels * in_channels * kernel_h * kernel_w
        
        # Calculate weights per PE
        weights_per_pe = total_weights // len(pe_names)
        if total_weights % len(pe_names) != 0:
            weights_per_pe += 1
            
        # Create mapping
        mapping = {
            'layer_type': 'conv2d',
            'out_channels': out_channels,
            'in_channels': in_channels,
            'kernel_size': (kernel_h, kernel_w),
            'pe_mapping': {}
        }
        
        # Distribute weights across PEs
        weights = conv_layer.weight.detach().clone()
        bias = conv_layer.bias.detach().clone() if conv_layer.bias is not None else None
        
        flat_weights = weights.reshape(-1)
        
        for i, pe_name in enumerate(pe_names):
            pe = self.processing_elements[pe_name]
            start_idx = i * weights_per_pe
            end_idx = min((i + 1) * weights_per_pe, total_weights)
            
            if start_idx >= total_weights:
                break
                
            # Calculate how many arrays we need in this PE
            pe_weights = flat_weights[start_idx:end_idx]
            arrays_needed = (pe_weights.numel() + (pe.array_rows * pe.array_cols) - 1) // (pe.array_rows * pe.array_cols)
            
            if arrays_needed > pe.num_arrays:
                raise ValueError(f"PE {pe_name} has {pe.num_arrays} arrays but needs {arrays_needed}")
                
            # Split weights for each array in the PE
            pe_weight_chunks = []
            for j in range(arrays_needed):
                chunk_start = j * (pe.array_rows * pe.array_cols)
                chunk_end = min((j + 1) * (pe.array_rows * pe.array_cols), pe_weights.numel())
                
                if chunk_start >= pe_weights.numel():
                    break
                    
                chunk = pe_weights[chunk_start:chunk_end]
                
                # Pad if necessary
                if chunk.numel() < pe.array_rows * pe.array_cols:
                    padded = torch.zeros(pe.array_rows * pe.array_cols, device=chunk.device)
                    padded[:chunk.numel()] = chunk
                    chunk = padded
                    
                # Reshape to match crossbar dimensions
                chunk = chunk.reshape(pe.array_rows, pe.array_cols)
                pe_weight_chunks.append(chunk)
                
            # Program weights into PE
            pe.program_weights(pe_weight_chunks)
            
            # Store mapping details
            mapping['pe_mapping'][pe_name] = {
                'weight_range': (start_idx, end_idx),
                'arrays_used': arrays_needed
            }
            
        # Store the mapping
        self.layer_to_pe_mapping[layer_name] = pe_names
        self.weight_mapping[layer_name] = mapping
        
        return mapping
    
    def map_fc_layer(self, layer_name, fc_layer, pe_names):
        """
        Map a fully-connected layer to one or more processing elements.
        
        Args:
            layer_name (str): Name identifier for the layer.
            fc_layer (nn.Linear): Fully-connected layer to map.
            pe_names (list): List of PE names to map this layer to.
            
        Returns:
            dict: Mapping details for this layer.
        """
        if not all(pe_name in self.processing_elements for pe_name in pe_names):
            missing = [pe for pe in pe_names if pe not in self.processing_elements]
            raise ValueError(f"Processing elements not found: {missing}")
        
        # Get layer parameters
        out_features, in_features = fc_layer.weight.shape
        
        # Calculate total weight elements
        total_weights = out_features * in_features
        
        # Calculate weights per PE
        weights_per_pe = total_weights // len(pe_names)
        if total_weights % len(pe_names) != 0:
            weights_per_pe += 1
            
        # Create mapping
        mapping = {
            'layer_type': 'linear',
            'out_features': out_features,
            'in_features': in_features,
            'pe_mapping': {}
        }
        
        # Distribute weights across PEs
        weights = fc_layer.weight.detach().clone()
        bias = fc_layer.bias.detach().clone() if fc_layer.bias is not None else None
        
        flat_weights = weights.reshape(-1)
        
        for i, pe_name in enumerate(pe_names):
            pe = self.processing_elements[pe_name]
            start_idx = i * weights_per_pe
            end_idx = min((i + 1) * weights_per_pe, total_weights)
            
            if start_idx >= total_weights:
                break
                
            # Calculate how many arrays we need in this PE
            pe_weights = flat_weights[start_idx:end_idx]
            arrays_needed = (pe_weights.numel() + (pe.array_rows * pe.array_cols) - 1) // (pe.array_rows * pe.array_cols)
            
            if arrays_needed > pe.num_arrays:
                raise ValueError(f"PE {pe_name} has {pe.num_arrays} arrays but needs {arrays_needed}")
                
            # Split weights for each array in the PE
            pe_weight_chunks = []
            for j in range(arrays_needed):
                chunk_start = j * (pe.array_rows * pe.array_cols)
                chunk_end = min((j + 1) * (pe.array_rows * pe.array_cols), pe_weights.numel())
                
                if chunk_start >= pe_weights.numel():
                    break
                    
                chunk = pe_weights[chunk_start:chunk_end]
                
                # Pad if necessary
                if chunk.numel() < pe.array_rows * pe.array_cols:
                    padded = torch.zeros(pe.array_rows * pe.array_cols, device=chunk.device)
                    padded[:chunk.numel()] = chunk
                    chunk = padded
                    
                # Reshape to match crossbar dimensions
                chunk = chunk.reshape(pe.array_rows, pe.array_cols)
                pe_weight_chunks.append(chunk)
                
            # Program weights into PE
            pe.program_weights(pe_weight_chunks)
            
            # Store mapping details
            mapping['pe_mapping'][pe_name] = {
                'weight_range': (start_idx, end_idx),
                'arrays_used': arrays_needed
            }
            
        # Store the mapping
        self.layer_to_pe_mapping[layer_name] = pe_names
        self.weight_mapping[layer_name] = mapping
        
        return mapping
    
    def get_pe_for_layer(self, layer_name):
        """
        Get the processing elements assigned to a layer.
        
        Args:
            layer_name (str): Name of the layer.
            
        Returns:
            list: List of MemristorPE objects assigned to this layer.
        """
        if layer_name not in self.layer_to_pe_mapping:
            raise ValueError(f"Layer {layer_name} not mapped to any PE")
            
        pe_names = self.layer_to_pe_mapping[layer_name]
        return [self.processing_elements[pe_name] for pe_name in pe_names]
    
    def get_mapping_details(self, layer_name=None):
        """
        Get mapping details for a specific layer or all layers.
        
        Args:
            layer_name (str, optional): Name of the layer. If None, returns all mappings.
            
        Returns:
            dict: Mapping details.
        """
        if layer_name is not None:
            if layer_name not in self.weight_mapping:
                raise ValueError(f"Layer {layer_name} not mapped")
            return self.weight_mapping[layer_name]
        
        return self.weight_mapping

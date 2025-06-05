"""
MemTorch-based CNN model for leaf disease classification.
"""

import torch
import torch.nn as nn
import memtorch
from memtorch.mn import Module
from memtorch.map import Parameter
import os
import time

class MemTorchCNN(nn.Module):
    """
    MemTorch-based CNN for leaf disease classification.
    
    This model uses memristor crossbar arrays for computation, providing
    energy-efficient inference and training capabilities.
    """
    
    def __init__(self, num_classes=10, width_mult=0.75):
        """
        Initialize the MemTorch CNN model.
        
        Args:
            num_classes (int): Number of output classes.
            width_mult (float): Width multiplier for the network.
        """
        super(MemTorchCNN, self).__init__()
        
        # Store configuration first
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.is_memristive = False
        
        # Define input channels
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult)
        
        # First conv layer (keep as digital)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6()
        )
        
        # Inverted residual blocks (will be converted to memristive)
        self.inverted_residual_blocks, input_channel = self._build_inverted_residual_blocks(input_channel)
        
        # Last conv layer (will be converted to memristive)
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6()
        )
        
        # Global average pooling (keep as digital)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier (will be converted to memristive)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, num_classes)
        )
    
    def _build_inverted_residual_blocks(self, input_channel):
        """
        Build the inverted residual blocks.
        
        Args:
            input_channel (int): Number of input channels.
            
        Returns:
            tuple: (nn.Sequential of blocks, updated input channel count)
        """
        # MobileNetV2 configuration: [t, c, n, s]
        # t: expansion factor, c: output channels, n: repeat times, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        blocks = []
        current_channel = input_channel
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * self.width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                
                # Create inverted residual block
                block = self._create_inverted_residual_block(
                    current_channel, output_channel, stride, t
                )
                blocks.append(block)
                
                current_channel = output_channel
                
        return nn.Sequential(*blocks), current_channel
    
    def _create_inverted_residual_block(self, inp, oup, stride, expand_ratio):
        """
        Create an inverted residual block.
        
        Args:
            inp (int): Input channels.
            oup (int): Output channels.
            stride (int): Stride for depthwise convolution.
            expand_ratio (int): Expansion ratio for the block.
            
        Returns:
            nn.Sequential: Inverted residual block.
        """
        hidden_dim = int(inp * expand_ratio)
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                # Pointwise convolution to expand channels
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6()
            ])
        
        # Depthwise convolution
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6()
        ])
        
        # Projection phase
        layers.extend([
            # Pointwise convolution to project back
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        
        # Skip connection if input and output dimensions match
        use_skip_connection = (stride == 1 and inp == oup)
        
        if use_skip_connection:
            return InvertedResidualWithSkip(nn.Sequential(*layers))
        else:
            return nn.Sequential(*layers)
    
    def convert_to_memristive(self, memristor_model=None, tile_shape=(128, 128), 
                             adc_resolution=8, dac_resolution=8, max_input_voltage=0.3,
                             device=None):
        """
        Convert the model to use memristive layers.
        
        Args:
            memristor_model: Memristor device model to use.
            tile_shape (tuple): Shape of memristor crossbar tiles.
            adc_resolution (int): ADC resolution in bits.
            dac_resolution (int): DAC resolution in bits.
            max_input_voltage (float): Maximum input voltage.
            device: Device to use for computation.
        """
        if self.is_memristive:
            print("Model is already memristive.")
            return
        
        # Set device
        if device is None:
            device = next(self.parameters()).device
        
        # Define memristor model if not provided
        if memristor_model is None:
            memristor_model = memtorch.bh.memristor.LinearIonDrift(
                r_on=100,      # On resistance (ohms)
                r_off=16000,   # Off resistance (ohms)
                time_series_resolution=1e-10,  # Time resolution (s)
                window_function=memtorch.bh.memristor.window.Biolek()  # Window function
            )
        
        print(f"Converting model to memristive using {memristor_model.__class__.__name__}...")
        print(f"Tile shape: {tile_shape}, ADC resolution: {adc_resolution} bits, DAC resolution: {dac_resolution} bits")
        
        # Convert convolutional layers in inverted residual blocks
        for i, block in enumerate(self.inverted_residual_blocks):
            if isinstance(block, InvertedResidualWithSkip):
                # Convert layers in the block's conv sequence
                for j, layer in enumerate(block.conv):
                    if isinstance(layer, nn.Conv2d):
                        print(f"Converting inverted residual block {i}, conv layer {j}")
                        block.conv[j] = memtorch.mn.Conv2d(
                            module=layer,
                            memristor=memristor_model,
                            mapping_technique=memtorch.map.Parameter.CrossPoint,
                            tile_shape=tile_shape,
                            max_input_voltage=max_input_voltage,
                            ADC_resolution=adc_resolution,
                            DAC_resolution=dac_resolution
                        ).to(device)
            else:
                # Convert layers in the sequential block
                for j, layer in enumerate(block):
                    if isinstance(layer, nn.Conv2d):
                        print(f"Converting inverted residual block {i}, conv layer {j}")
                        block[j] = memtorch.mn.Conv2d(
                            module=layer,
                            memristor=memristor_model,
                            mapping_technique=memtorch.map.Parameter.CrossPoint,
                            tile_shape=tile_shape,
                            max_input_voltage=max_input_voltage,
                            ADC_resolution=adc_resolution,
                            DAC_resolution=dac_resolution
                        ).to(device)
        
        # Convert last conv layer
        for i, layer in enumerate(self.last_conv):
            if isinstance(layer, nn.Conv2d):
                print(f"Converting last conv layer {i}")
                self.last_conv[i] = memtorch.mn.Conv2d(
                    module=layer,
                    memristor=memristor_model,
                    mapping_technique=memtorch.map.Parameter.CrossPoint,
                    tile_shape=tile_shape,
                    max_input_voltage=max_input_voltage,
                    ADC_resolution=adc_resolution,
                    DAC_resolution=dac_resolution
                ).to(device)
        
        # Convert classifier
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                print(f"Converting classifier layer {i}")
                self.classifier[i] = memtorch.mn.Linear(
                    module=layer,
                    memristor=memristor_model,
                    mapping_technique=memtorch.map.Parameter.CrossPoint,
                    tile_shape=tile_shape,
                    max_input_voltage=max_input_voltage,
                    ADC_resolution=adc_resolution,
                    DAC_resolution=dac_resolution
                ).to(device)
        
        self.is_memristive = True
        print("Model conversion complete.")
    
    def apply_non_idealities(self, non_idealities=None, params=None):
        """
        Apply non-ideal characteristics to memristive layers.
        
        Args:
            non_idealities (list): List of non-idealities to apply.
            params (dict): Parameters for non-idealities.
        """
        if not self.is_memristive:
            raise ValueError("Model must be converted to memristive first.")
        
        if non_idealities is None:
            non_idealities = [
                memtorch.bh.nonideality.NonIdeality.DeviceFaults,
                memtorch.bh.nonideality.NonIdeality.Drift
            ]
        
        if params is None:
            params = {
                'conductance_variance': 0.1,  # 10% variance
                'drift_coefficient': 0.01     # Drift coefficient
            }
        
        print("Applying non-idealities to memristive layers...")
        
        # Apply to all memristive layers
        for name, module in self.named_modules():
            if isinstance(module, (memtorch.mn.Conv2d, memtorch.mn.Linear)):
                print(f"Applying non-idealities to {name}")
                module.apply_non_idealities(
                    non_idealities=non_idealities,
                    params=params
                )
        
        print("Non-idealities applied.")
    
    def apply_weight_quantization(self, bits=4):
        """
        Apply weight quantization to memristive layers.
        
        Args:
            bits (int): Number of bits for quantization.
        """
        if not self.is_memristive:
            raise ValueError("Model must be converted to memristive first.")
        
        print(f"Applying {bits}-bit weight quantization to memristive layers...")
        
        # Apply to all memristive layers
        for name, module in self.named_modules():
            if isinstance(module, (memtorch.mn.Conv2d, memtorch.mn.Linear)):
                print(f"Quantizing weights in {name}")
                module.apply_weight_constraints(
                    constraint=memtorch.bh.Constraint.WeightQuantization,
                    params={'bits': bits}
                )
        
        print("Weight quantization applied.")
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output logits.
        """
        # First conv layer
        x = self.first_conv(x)
        
        # Inverted residual blocks
        x = self.inverted_residual_blocks(x)
        
        # Last conv layer
        x = self.last_conv(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class InvertedResidualWithSkip(nn.Module):
    """
    Inverted Residual block with skip connection.
    
    Attributes:
        conv (nn.Sequential): Convolutional layers.
    """
    
    def __init__(self, conv):
        """
        Initialize the block.
        
        Args:
            conv (nn.Sequential): Convolutional layers.
        """
        super(InvertedResidualWithSkip, self).__init__()
        self.conv = conv
        
    def forward(self, x):
        """
        Forward pass with skip connection.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return x + self.conv(x)

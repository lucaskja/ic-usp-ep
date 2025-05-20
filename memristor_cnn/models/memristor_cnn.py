"""
Memristor-based Convolutional Neural Network (mCNN) architecture
optimized for leaf disease detection using TTN-MobileNetV2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# Import components from existing project
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stage1_mish.models.mish import Mish
from stage2_triplet.models.triplet_attention import TripletAttention
from stage3_cnsn.models.cnsn import CNSN

from .memristor_mapping import MemristorMapping


class MemristorCNN(nn.Module):
    """
    Memristor-based Convolutional Neural Network (mCNN) architecture
    optimized for leaf disease detection using TTN-MobileNetV2.
    
    This model integrates MobileNetV2 with Triplet Attention, CNSN normalization,
    and Mish activation, mapped to memristor crossbar arrays for efficient computation.
    
    Attributes:
        num_classes (int): Number of output classes.
        width_mult (float): Width multiplier for the network.
        memristor_mapping (MemristorMapping): Mapping of layers to memristor arrays.
        features (nn.Sequential): Feature extraction layers.
        classifier (nn.Sequential): Classification layers.
        hybrid_training_mode (str): Current training mode ('ex-situ' or 'in-situ').
    """
    
    def __init__(self, num_classes=10, width_mult=0.75):
        """
        Initialize the Memristor CNN model.
        
        Args:
            num_classes (int): Number of output classes.
            width_mult (float): Width multiplier for the network.
        """
        super(MemristorCNN, self).__init__()
        
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.memristor_mapping = MemristorMapping()
        self.hybrid_training_mode = 'ex-situ'  # Start with ex-situ training
        
        # Build the network
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_network(self):
        """Build the TTN-MobileNetV2 network architecture."""
        # Input conv layer
        input_channel = int(32 * self.width_mult)
        
        # First conv layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            Mish()  # Replace ReLU6 with Mish
        )
        
        # Inverted residual blocks
        self.inverted_residual_blocks = self._build_inverted_residual_blocks(input_channel)
        
        # Last conv layer
        last_channel = int(1280 * self.width_mult)
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            Mish()  # Replace ReLU6 with Mish
        )
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier (fully connected layer)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, self.num_classes)
        )
        
    def _build_inverted_residual_blocks(self, input_channel):
        """
        Build the inverted residual blocks with Triplet Attention and CNSN.
        
        Args:
            input_channel (int): Number of input channels.
            
        Returns:
            nn.Sequential: Sequence of inverted residual blocks.
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
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * self.width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                
                # Create inverted residual block
                block = self._create_inverted_residual_block(
                    input_channel, output_channel, stride, t
                )
                blocks.append(block)
                
                input_channel = output_channel
                
        return nn.Sequential(*blocks)
    
    def _create_inverted_residual_block(self, inp, oup, stride, expand_ratio):
        """
        Create an inverted residual block with Triplet Attention and CNSN.
        
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
                Mish()  # Replace ReLU6 with Mish
            ])
        
        # Depthwise convolution
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Mish()  # Replace ReLU6 with Mish
        ])
        
        # Triplet Attention after depthwise convolution
        layers.append(TripletAttention())
        
        # CNSN after Triplet Attention
        layers.append(CNSN(hidden_dim))
        
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
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def setup_memristor_mapping(self, device=None):
        """
        Set up the memristor mapping for the model.
        
        Args:
            device (torch.device, optional): Device to use for tensors.
        """
        # Create processing elements
        self.memristor_mapping.create_processing_element("PE1", num_arrays=4, device=device)
        self.memristor_mapping.create_processing_element("PE3", num_arrays=4, device=device)
        self.memristor_mapping.create_processing_element("PE5", num_arrays=8, device=device)
        self.memristor_mapping.create_processing_element("PE7", num_arrays=8, device=device)
        
        # Map convolutional layers
        # First conv layer
        self.memristor_mapping.map_conv_layer(
            "first_conv.0", self.first_conv[0], ["PE1"]
        )
        
        # Map fully connected layer
        self.memristor_mapping.map_fc_layer(
            "classifier.0", self.classifier[0], ["PE5", "PE7"]
        )
    
    def set_hybrid_training_mode(self, mode):
        """
        Set the hybrid training mode.
        
        Args:
            mode (str): Training mode ('ex-situ' or 'in-situ').
        """
        if mode not in ['ex-situ', 'in-situ']:
            raise ValueError("Mode must be 'ex-situ' or 'in-situ'")
        
        self.hybrid_training_mode = mode
        
        # In ex-situ mode, all layers are trainable
        if mode == 'ex-situ':
            for param in self.parameters():
                param.requires_grad = True
        
        # In in-situ mode, only FC layer is trainable
        elif mode == 'in-situ':
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier parameters
            for param in self.classifier.parameters():
                param.requires_grad = True
    
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
        
        # Get the current number of channels
        current_channels = x.size(1)
        
        # Last conv layer - ensure input and output channels match
        if not hasattr(self, '_last_conv_fixed'):
            # Dynamically adjust the last conv layer to match input channels
            last_channel = int(1280 * self.width_mult)
            self.last_conv = nn.Sequential(
                nn.Conv2d(current_channels, last_channel, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(last_channel),
                Mish()
            )
            self._last_conv_fixed = True
        
        x = self.last_conv(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def transfer_weights_to_memristor(self):
        """
        Transfer the trained weights to memristor arrays using closed-loop programming.
        This is used during the transition from ex-situ to in-situ training.
        """
        # This would involve actual hardware programming in a real implementation
        # Here we simulate the process
        print("Transferring weights to memristor arrays...")
        
        # The actual weight transfer would happen here, using the memristor_mapping
        # For simulation purposes, we just print the process
        
        print("Weight transfer complete.")
    
    def threshold_based_update(self, inputs, targets, learning_rate=0.001, threshold=0.1):
        """
        Perform threshold-based weight update for in-situ training.
        This is used during in-situ training phase for the FC layer.
        
        Args:
            inputs (torch.Tensor): Input features to the FC layer.
            targets (torch.Tensor): Target labels.
            learning_rate (float): Learning rate for the update.
            threshold (float): Threshold for weight updates.
            
        Returns:
            float: Loss value.
        """
        if self.hybrid_training_mode != 'in-situ':
            raise ValueError("Threshold-based update can only be used in in-situ mode")
        
        # Forward pass
        outputs = self.classifier(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Calculate gradients
        loss.backward()
        
        # Apply threshold-based update to FC layer weights
        with torch.no_grad():
            for param in self.classifier.parameters():
                if param.grad is not None:
                    # Apply threshold
                    mask = torch.abs(param.grad) > threshold
                    # Update only weights that exceed the threshold
                    param[mask] -= learning_rate * param.grad[mask]
        
        return loss.item()


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

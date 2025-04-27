"""
MobileNetV2 model with Mish activation and Triplet Attention.
"""
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from stage1_mish.models.mish import replace_relu_with_mish
from stage2_triplet.models.triplet_attention import TripletAttention


class InvertedResidualWithTripletAttention(nn.Module):
    """
    Inverted Residual block with Triplet Attention.
    Wraps the original block and adds Triplet Attention after it.
    """
    def __init__(self, inverted_residual_block, kernel_size=7):
        """
        Initialize wrapper for inverted residual block.
        
        Args:
            inverted_residual_block: Original inverted residual block
            kernel_size (int): Kernel size for Triplet Attention
        """
        super(InvertedResidualWithTripletAttention, self).__init__()
        self.block = inverted_residual_block
        self.attention = TripletAttention(kernel_size=kernel_size)
        self.use_res_connect = self.block.use_res_connect
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.use_res_connect:
            out = self.block.conv(x)
            out = self.attention(out)
            return x + out
        else:
            out = self.block(x)
            out = self.attention(out)
            return out


def add_triplet_attention_to_mobilenetv2(model, kernel_size=7):
    """
    Add Triplet Attention to MobileNetV2 model.
    
    Args:
        model (nn.Module): MobileNetV2 model
        kernel_size (int): Kernel size for Triplet Attention
        
    Returns:
        nn.Module: MobileNetV2 model with Triplet Attention
    """
    # Add Triplet Attention after each inverted residual block
    for i, layer in enumerate(model.features):
        if hasattr(layer, 'conv'):  # Check if it's an inverted residual block
            model.features[i] = InvertedResidualWithTripletAttention(layer, kernel_size=kernel_size)
    
    return model


def create_mobilenetv2_triplet(num_classes, pretrained=True, triplet_attention_kernel_size=7):
    """
    Create a MobileNetV2 model with Mish activation and Triplet Attention.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        triplet_attention_kernel_size (int): Kernel size for Triplet Attention
        
    Returns:
        nn.Module: MobileNetV2 model with Mish and Triplet Attention
    """
    # Ensure num_classes is correct for leaf disease dataset (3 classes)
    if num_classes != 3 and 'leaf_disease' in os.getcwd():
        print(f"Warning: Expected 3 classes for leaf disease dataset, but got {num_classes}. Using 3 classes.")
        num_classes = 3
        
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)
    
    # Replace ReLU6 with Mish
    model = replace_relu_with_mish(model)
    
    # Add Triplet Attention
    model = add_triplet_attention_to_mobilenetv2(model, kernel_size=triplet_attention_kernel_size)
    
    # Modify the classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


class MobileNetV2TripletModel(nn.Module):
    """
    Wrapper class for MobileNetV2 model with Mish activation and Triplet Attention.
    """
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize MobileNetV2 model with Mish activation and Triplet Attention.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(MobileNetV2TripletModel, self).__init__()
        # For leaf disease dataset, ensure we have 3 classes
        if 'leaf_disease' in os.getcwd() and num_classes != 3:
            print(f"Warning: Expected 3 classes for leaf disease dataset, but got {num_classes}. Using 3 classes.")
            num_classes = 3
            
        self.model = create_mobilenetv2_triplet(num_classes, pretrained)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

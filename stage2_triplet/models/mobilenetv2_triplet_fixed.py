"""
MobileNetV2 model with Mish activation and Triplet Attention.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from stage1_mish.models.mish import replace_relu_with_mish
from stage2_triplet.models.triplet_attention_fixed import TripletAttention


class InvertedResidualWithTripletAttention(nn.Module):
    """
    Inverted Residual block with Triplet Attention.
    Adds Triplet Attention after the depthwise convolution.
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
        
        # Extract the layers from the original block
        if hasattr(self.block, 'conv'):
            # For blocks with residual connection
            layers = list(self.block.conv)
            
            # Find the depthwise convolution layer
            dw_conv_idx = None
            for i, layer in enumerate(layers):
                if isinstance(layer, nn.Conv2d) and layer.groups > 1:
                    dw_conv_idx = i
                    break
            
            if dw_conv_idx is not None:
                # Split the layers before and after depthwise conv
                self.layers_before_dw = nn.Sequential(*layers[:dw_conv_idx+1])
                self.layers_after_dw = nn.Sequential(*layers[dw_conv_idx+1:])
            else:
                # Fallback if depthwise conv not found
                self.layers_before_dw = self.block.conv
                self.layers_after_dw = nn.Identity()
        else:
            # Fallback for other types of blocks
            self.layers_before_dw = self.block
            self.layers_after_dw = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.use_res_connect:
            # Apply layers before depthwise conv
            out = self.layers_before_dw(x)
            
            # Apply Triplet Attention
            out = self.attention(out)
            
            # Apply layers after depthwise conv
            out = self.layers_after_dw(out)
            
            # Add residual connection
            return x + out
        else:
            # Apply layers before depthwise conv
            out = self.layers_before_dw(x)
            
            # Apply Triplet Attention
            out = self.attention(out)
            
            # Apply layers after depthwise conv
            out = self.layers_after_dw(out)
            
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


def create_mobilenetv2_triplet(num_classes, pretrained=True, triplet_attention_kernel_size=7, width_mult=0.75):
    """
    Create a MobileNetV2 model with Mish activation and Triplet Attention.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        triplet_attention_kernel_size (int): Kernel size for Triplet Attention
        width_mult (float): Width multiplier for the network (default: 0.75)
        
    Returns:
        nn.Module: MobileNetV2 model with Mish and Triplet Attention
    """
    if pretrained and width_mult == 1.0:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None, width_mult=width_mult)
        if pretrained and width_mult != 1.0:
            print(f"Warning: Pretrained weights are only available for width_mult=1.0. Using random initialization for width_mult={width_mult}.")
    
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
    def __init__(self, num_classes, pretrained=True, width_mult=0.75):
        """
        Initialize MobileNetV2 model with Mish activation and Triplet Attention.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            width_mult (float): Width multiplier for the network (default: 0.75)
        """
        super(MobileNetV2TripletModel, self).__init__()
        self.model = create_mobilenetv2_triplet(num_classes, pretrained, width_mult=width_mult)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

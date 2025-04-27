"""
MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from stage1_mish.models.mish import replace_relu_with_mish
from stage2_triplet.models.triplet_attention_fixed import TripletAttention
from stage3_cnsn.models.cnsn_fixed import CNSN


class InvertedResidualWithTripletAttentionAndCNSN(nn.Module):
    """
    Inverted Residual block with Triplet Attention and CNSN.
    Adds Triplet Attention and CNSN after the depthwise convolution.
    """
    def __init__(self, inverted_residual_block, num_channels, kernel_size=7, p=0.5):
        """
        Initialize wrapper for inverted residual block.
        
        Args:
            inverted_residual_block: Original inverted residual block
            num_channels (int): Number of output channels
            kernel_size (int): Kernel size for Triplet Attention
            p (float): Probability of applying CrossNorm during training
        """
        super(InvertedResidualWithTripletAttentionAndCNSN, self).__init__()
        self.block = inverted_residual_block
        self.attention = TripletAttention(kernel_size=kernel_size)
        self.cnsn = CNSN(num_channels, p=p)
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
            
            # Apply CNSN
            out = self.cnsn(out)
            
            # Apply layers after depthwise conv
            out = self.layers_after_dw(out)
            
            # Add residual connection
            return x + out
        else:
            # Apply layers before depthwise conv
            out = self.layers_before_dw(x)
            
            # Apply Triplet Attention
            out = self.attention(out)
            
            # Apply CNSN
            out = self.cnsn(out)
            
            # Apply layers after depthwise conv
            out = self.layers_after_dw(out)
            
            return out


def get_output_channels(layer):
    """
    Get the number of output channels from a layer.
    
    Args:
        layer (nn.Module): Layer to analyze
        
    Returns:
        int: Number of output channels
    """
    # For inverted residual blocks with residual connection
    if hasattr(layer, 'conv'):
        # Navigate through the conv sequential to find the last conv layer
        for m in reversed(list(layer.conv.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
    
    # For other blocks, try to find the last layer with out_channels
    for m in reversed(list(layer.modules())):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    
    # Fallback: use a default value
    return 32


def add_triplet_attention_and_cnsn_to_mobilenetv2(model, kernel_size=7, p=0.5):
    """
    Add Triplet Attention and CNSN to MobileNetV2 model.
    
    Args:
        model (nn.Module): MobileNetV2 model
        kernel_size (int): Kernel size for Triplet Attention
        p (float): Probability of applying CrossNorm during training
        
    Returns:
        nn.Module: MobileNetV2 model with Triplet Attention and CNSN
    """
    # Add Triplet Attention and CNSN after each inverted residual block
    for i, layer in enumerate(model.features):
        if hasattr(layer, 'conv'):  # Check if it's an inverted residual block
            # Get number of output channels
            num_channels = get_output_channels(layer)
            model.features[i] = InvertedResidualWithTripletAttentionAndCNSN(
                layer, 
                num_channels, 
                kernel_size=kernel_size,
                p=p
            )
    
    return model


def create_mobilenetv2_cnsn(num_classes, pretrained=True, kernel_size=7, p=0.5):
    """
    Create a MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        kernel_size (int): Kernel size for Triplet Attention
        p (float): Probability of applying CrossNorm during training
        
    Returns:
        nn.Module: MobileNetV2 model with Mish, Triplet Attention, and CNSN
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)
    
    # Replace ReLU6 with Mish
    model = replace_relu_with_mish(model)
    
    # Add Triplet Attention and CNSN
    model = add_triplet_attention_and_cnsn_to_mobilenetv2(
        model, 
        kernel_size=kernel_size,
        p=p
    )
    
    # Modify the classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


class MobileNetV2CNSNModel(nn.Module):
    """
    Wrapper class for MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
    """
    def __init__(self, num_classes, pretrained=True, p=0.5):
        """
        Initialize MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            p (float): Probability of applying CrossNorm during training
        """
        super(MobileNetV2CNSNModel, self).__init__()
        self.model = create_mobilenetv2_cnsn(num_classes, pretrained, p=p)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

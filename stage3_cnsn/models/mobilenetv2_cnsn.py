"""
MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
"""
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from stage1_mish.models.mish import replace_relu_with_mish
from stage2_triplet.models.triplet_attention import TripletAttention
from stage3_cnsn.models.cnsn import CNSN


class InvertedResidualWithTripletAttentionAndCNSN(nn.Module):
    """
    Inverted Residual block with Triplet Attention and CNSN.
    """
    def __init__(self, inverted_residual_block, num_channels, crossnorm_mode='2-instance', triplet_attention_kernel_size=7):
        """
        Initialize wrapper for inverted residual block.
        
        Args:
            inverted_residual_block: Original inverted residual block
            num_channels (int): Number of output channels
            crossnorm_mode (str): CrossNorm mode ('1-instance', '2-instance', or 'crop')
            triplet_attention_kernel_size (int): Kernel size for Triplet Attention
        """
        super(InvertedResidualWithTripletAttentionAndCNSN, self).__init__()
        self.block = inverted_residual_block
        self.attention = TripletAttention(kernel_size=triplet_attention_kernel_size)
        self.cnsn = CNSN(num_channels, crossnorm_mode=crossnorm_mode, p=0.5)
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
            out = self.cnsn(out)
            return x + out
        else:
            out = self.block(x)
            out = self.attention(out)
            out = self.cnsn(out)
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
            if hasattr(m, 'out_channels'):
                return m.out_channels
    
    # For other blocks, try to find the last layer with out_channels
    for m in reversed(list(layer.modules())):
        if hasattr(m, 'out_channels'):
            return m.out_channels
    
    # Fallback: try to infer from the output shape
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        try:
            output = layer(dummy_input)
            return output.shape[1]  # Channels dimension
        except:
            # If all else fails, use a default value
            return 32


def add_triplet_attention_and_cnsn_to_mobilenetv2(model, crossnorm_mode='2-instance', triplet_attention_kernel_size=7):
    """
    Add Triplet Attention and CNSN to MobileNetV2 model.
    
    Args:
        model (nn.Module): MobileNetV2 model
        crossnorm_mode (str): CrossNorm mode ('1-instance', '2-instance', or 'crop')
        triplet_attention_kernel_size (int): Kernel size for Triplet Attention
        
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
                crossnorm_mode=crossnorm_mode,
                triplet_attention_kernel_size=triplet_attention_kernel_size
            )
    
    return model


def create_mobilenetv2_cnsn(num_classes, pretrained=True, cn_mode='2-instance', triplet_attention_kernel_size=7):
    """
    Create a MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        cn_mode (str): CrossNorm mode ('1-instance', '2-instance', or 'crop')
        triplet_attention_kernel_size (int): Kernel size for Triplet Attention
        
    Returns:
        nn.Module: MobileNetV2 model with Mish, Triplet Attention, and CNSN
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
    
    # Add Triplet Attention and CNSN
    model = add_triplet_attention_and_cnsn_to_mobilenetv2(
        model, 
        crossnorm_mode=cn_mode,
        triplet_attention_kernel_size=triplet_attention_kernel_size
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
    def __init__(self, num_classes, pretrained=True, cn_mode='2-instance'):
        """
        Initialize MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            cn_mode (str): CrossNorm mode ('1-instance', '2-instance', or 'crop')
        """
        super(MobileNetV2CNSNModel, self).__init__()
        self.model = create_mobilenetv2_cnsn(num_classes, pretrained, cn_mode)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

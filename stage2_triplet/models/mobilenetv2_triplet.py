"""
MobileNetV2 model with Mish activation and Triplet Attention.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from mobilenetv2_improvements.stage1_mish.models.mish import replace_relu_with_mish
from mobilenetv2_improvements.stage2_triplet.models.triplet_attention import TripletAttention


class InvertedResidualWithTriplet(nn.Module):
    """
    Inverted Residual block with Triplet Attention.
    """
    def __init__(self, inverted_residual_block, use_triplet=True, kernel_size=7):
        """
        Initialize Inverted Residual block with Triplet Attention.
        
        Args:
            inverted_residual_block (nn.Module): Original inverted residual block
            use_triplet (bool): Whether to use triplet attention
            kernel_size (int): Kernel size for triplet attention
        """
        super(InvertedResidualWithTriplet, self).__init__()
        self.inverted_block = inverted_residual_block
        self.use_triplet = use_triplet
        if use_triplet:
            self.triplet_attention = TripletAttention(kernel_size=kernel_size)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.inverted_block(x)
        if self.use_triplet:
            x = self.triplet_attention(x)
        return x


def add_triplet_attention_to_mobilenetv2(model, kernel_size=7):
    """
    Add Triplet Attention to MobileNetV2 model.
    
    Args:
        model (nn.Module): MobileNetV2 model
        kernel_size (int): Kernel size for triplet attention
        
    Returns:
        nn.Module: MobileNetV2 model with Triplet Attention
    """
    # Add Triplet Attention after each inverted residual block
    for i, layer in enumerate(model.features):
        if hasattr(layer, 'conv'):  # Check if it's an inverted residual block
            model.features[i] = InvertedResidualWithTriplet(layer, kernel_size=kernel_size)
    
    return model


def create_mobilenetv2_triplet(num_classes, pretrained=True, use_mish=True, kernel_size=7):
    """
    Create a MobileNetV2 model with Mish activation and Triplet Attention.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        use_mish (bool): Whether to use Mish activation
        kernel_size (int): Kernel size for triplet attention
        
    Returns:
        nn.Module: MobileNetV2 model with Mish activation and Triplet Attention
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)
    
    # Replace ReLU6 with Mish if specified
    if use_mish:
        model = replace_relu_with_mish(model)
    
    # Add Triplet Attention
    model = add_triplet_attention_to_mobilenetv2(model, kernel_size=kernel_size)
    
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
    def __init__(self, num_classes, pretrained=True, use_mish=True, kernel_size=7):
        """
        Initialize MobileNetV2 model with Mish activation and Triplet Attention.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            use_mish (bool): Whether to use Mish activation
            kernel_size (int): Kernel size for triplet attention
        """
        super(MobileNetV2TripletModel, self).__init__()
        self.model = create_mobilenetv2_triplet(
            num_classes, 
            pretrained=pretrained, 
            use_mish=use_mish, 
            kernel_size=kernel_size
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

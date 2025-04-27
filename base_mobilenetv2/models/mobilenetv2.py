"""
Base MobileNetV2 model for leaf disease classification.
"""
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def create_mobilenetv2(num_classes, pretrained=True, width_mult=0.75):
    """
    Create a MobileNetV2 model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        width_mult (float): Width multiplier for the network (default: 0.75)
        
    Returns:
        nn.Module: MobileNetV2 model
    """
    # Ensure num_classes is correct for leaf disease dataset (3 classes)
    if num_classes != 3 and 'leaf_disease' in os.getcwd():
        print(f"Warning: Expected 3 classes for leaf disease dataset, but got {num_classes}. Using 3 classes.")
        num_classes = 3
        
    if pretrained and width_mult == 1.0:
        # Pretrained weights are only available for width_mult=1.0
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        # For custom width_mult, we can't use pretrained weights
        model = mobilenet_v2(weights=None, width_mult=width_mult)
        if pretrained and width_mult != 1.0:
            print(f"Warning: Pretrained weights are only available for width_mult=1.0. Using random initialization for width_mult={width_mult}.")
    
    # Modify the classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


class MobileNetV2Model(nn.Module):
    """
    Wrapper class for MobileNetV2 model.
    """
    def __init__(self, num_classes, pretrained=True, width_mult=0.75):
        """
        Initialize MobileNetV2 model.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            width_mult (float): Width multiplier for the network (default: 0.75)
        """
        super(MobileNetV2Model, self).__init__()
        self.model = create_mobilenetv2(num_classes, pretrained, width_mult)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

"""
Base MobileNetV2 model implementation.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def create_mobilenetv2(num_classes, pretrained=True):
    """
    Create a MobileNetV2 model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: MobileNetV2 model
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)
    
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
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize MobileNetV2 model.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(MobileNetV2Model, self).__init__()
        self.model = create_mobilenetv2(num_classes, pretrained)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

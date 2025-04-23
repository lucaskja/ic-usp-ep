"""
MobileNetV2 model with Mish activation function.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from stage1_mish.models.mish import replace_relu_with_mish


def create_mobilenetv2_mish(num_classes, pretrained=True):
    """
    Create a MobileNetV2 model with Mish activation.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: MobileNetV2 model with Mish activation
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)
    
    # Replace ReLU6 with Mish
    model = replace_relu_with_mish(model)
    
    # Modify the classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


class MobileNetV2MishModel(nn.Module):
    """
    Wrapper class for MobileNetV2 model with Mish activation.
    """
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize MobileNetV2 model with Mish activation.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(MobileNetV2MishModel, self).__init__()
        self.model = create_mobilenetv2_mish(num_classes, pretrained)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

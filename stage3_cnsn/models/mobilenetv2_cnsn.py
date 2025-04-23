"""
MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from mobilenetv2_improvements.stage1_mish.models.mish import replace_relu_with_mish
from mobilenetv2_improvements.stage2_triplet.models.triplet_attention import TripletAttention
from mobilenetv2_improvements.stage3_cnsn.models.cnsn import CNSN


class InvertedResidualWithTripletAndCNSN(nn.Module):
    """
    Inverted Residual block with Triplet Attention and CNSN.
    """
    def __init__(self, inverted_residual_block, use_triplet=True, use_cnsn=True, 
                 triplet_kernel_size=7, cn_mode='1-instance'):
        """
        Initialize Inverted Residual block with Triplet Attention and CNSN.
        
        Args:
            inverted_residual_block (nn.Module): Original inverted residual block
            use_triplet (bool): Whether to use triplet attention
            use_cnsn (bool): Whether to use CNSN
            triplet_kernel_size (int): Kernel size for triplet attention
            cn_mode (str): Mode for CrossNorm operation
        """
        super(InvertedResidualWithTripletAndCNSN, self).__init__()
        self.inverted_block = inverted_residual_block
        self.use_triplet = use_triplet
        self.use_cnsn = use_cnsn
        
        # Get number of output channels from inverted residual block
        # This is a bit hacky but works for MobileNetV2's implementation
        if hasattr(inverted_residual_block, 'conv'):
            if hasattr(inverted_residual_block.conv, 'layers'):
                # Find the last conv layer to get output channels
                for layer in reversed(inverted_residual_block.conv.layers):
                    if isinstance(layer, nn.Conv2d):
                        out_channels = layer.out_channels
                        break
            else:
                # Fallback to a default value
                out_channels = 32
        else:
            # Fallback to a default value
            out_channels = 32
        
        if use_triplet:
            self.triplet_attention = TripletAttention(kernel_size=triplet_kernel_size)
        
        if use_cnsn:
            self.cnsn = CNSN(num_channels=out_channels, cn_mode=cn_mode)
        
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
        
        if self.use_cnsn:
            x = self.cnsn(x)
        
        return x


def add_triplet_and_cnsn_to_mobilenetv2(model, use_triplet=True, use_cnsn=True, 
                                        triplet_kernel_size=7, cn_mode='1-instance'):
    """
    Add Triplet Attention and CNSN to MobileNetV2 model.
    
    Args:
        model (nn.Module): MobileNetV2 model
        use_triplet (bool): Whether to use triplet attention
        use_cnsn (bool): Whether to use CNSN
        triplet_kernel_size (int): Kernel size for triplet attention
        cn_mode (str): Mode for CrossNorm operation
        
    Returns:
        nn.Module: MobileNetV2 model with Triplet Attention and CNSN
    """
    # Add Triplet Attention and CNSN after each inverted residual block
    for i, layer in enumerate(model.features):
        if hasattr(layer, 'conv'):  # Check if it's an inverted residual block
            model.features[i] = InvertedResidualWithTripletAndCNSN(
                layer, 
                use_triplet=use_triplet, 
                use_cnsn=use_cnsn,
                triplet_kernel_size=triplet_kernel_size,
                cn_mode=cn_mode
            )
    
    return model


def create_mobilenetv2_cnsn(num_classes, pretrained=True, use_mish=True, 
                           use_triplet=True, use_cnsn=True, 
                           triplet_kernel_size=7, cn_mode='1-instance'):
    """
    Create a MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        use_mish (bool): Whether to use Mish activation
        use_triplet (bool): Whether to use Triplet Attention
        use_cnsn (bool): Whether to use CNSN
        triplet_kernel_size (int): Kernel size for triplet attention
        cn_mode (str): Mode for CrossNorm operation
        
    Returns:
        nn.Module: MobileNetV2 model with Mish activation, Triplet Attention, and CNSN
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)
    
    # Replace ReLU6 with Mish if specified
    if use_mish:
        model = replace_relu_with_mish(model)
    
    # Add Triplet Attention and CNSN
    model = add_triplet_and_cnsn_to_mobilenetv2(
        model, 
        use_triplet=use_triplet, 
        use_cnsn=use_cnsn,
        triplet_kernel_size=triplet_kernel_size,
        cn_mode=cn_mode
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
    def __init__(self, num_classes, pretrained=True, use_mish=True, 
                use_triplet=True, use_cnsn=True, 
                triplet_kernel_size=7, cn_mode='1-instance'):
        """
        Initialize MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            use_mish (bool): Whether to use Mish activation
            use_triplet (bool): Whether to use Triplet Attention
            use_cnsn (bool): Whether to use CNSN
            triplet_kernel_size (int): Kernel size for triplet attention
            cn_mode (str): Mode for CrossNorm operation
        """
        super(MobileNetV2CNSNModel, self).__init__()
        self.model = create_mobilenetv2_cnsn(
            num_classes, 
            pretrained=pretrained, 
            use_mish=use_mish,
            use_triplet=use_triplet,
            use_cnsn=use_cnsn,
            triplet_kernel_size=triplet_kernel_size,
            cn_mode=cn_mode
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

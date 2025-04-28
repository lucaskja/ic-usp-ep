# SelfNorm
class SelfNorm(nn.Module):
    """
    SelfNorm module for bridging train-test distribution gap.
    
    Learns channel statistics recalibration using FC layers.
    Active during both training and inference.
    """
    def __init__(self, channels):
        """
        Initialize SelfNorm module.
        
        Args:
            channels (int): Number of input channels
        """
        super(SelfNorm, self).__init__()
        
        # FC layers for mean weight generation
        self.fc_mean = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # FC layers for std weight generation
        self.fc_std = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        B, C, H, W = x.size()
        
        # Calculate mean and std across spatial dimensions
        mean = x.mean(dim=[2, 3])  # [B, C]
        std = torch.sqrt(x.var(dim=[2, 3]) + 1e-5)  # [B, C]
        
        # Process each channel independently
        mean_weights = []
        std_weights = []
        
        for c in range(C):
            # Get statistics for this channel
            channel_stats = torch.stack([mean[:, c], std[:, c]], dim=1)  # [B, 2]
            
            # Generate weights for this channel
            mean_weight = self.fc_mean(channel_stats)  # [B, 1]
            std_weight = self.fc_std(channel_stats)  # [B, 1]
            
            mean_weights.append(mean_weight)
            std_weights.append(std_weight)
        
        # Stack weights for all channels
        mean_weights = torch.stack(mean_weights, dim=1)  # [B, C, 1]
        std_weights = torch.stack(std_weights, dim=1)  # [B, C, 1]
        
        # Reshape for broadcasting
        mean = mean.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        std = std.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        mean_weights = mean_weights.unsqueeze(-1)  # [B, C, 1, 1]
        std_weights = std_weights.unsqueeze(-1)  # [B, C, 1, 1]
        
        # Apply recalibration
        x_norm = (x - mean) / std
        x_selfnorm = x_norm * std_weights + mean * mean_weights
        
        return x_selfnorm

# CNSN
class CNSN(nn.Module):
    """
    CNSN (CrossNorm and SelfNorm) module.
    
    Combines CrossNorm for training distribution expansion and
    SelfNorm for bridging train-test distribution gap.
    """
    def __init__(self, channels, p=0.5):
        """
        Initialize CNSN module.
        
        Args:
            channels (int): Number of input channels
            p (float): Probability of applying CrossNorm during training
        """
        super(CNSN, self).__init__()
        self.crossnorm = CrossNorm(p=p)
        self.selfnorm = SelfNorm(channels)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Apply CrossNorm (only during training with probability p)
        x = self.crossnorm(x)
        
        # Apply SelfNorm (both training and testing)
        x = self.selfnorm(x)
        
        return x

#######################
# MobileNetV2 Models
#######################

# Base MobileNetV2
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
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    
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

# MobileNetV2 with Mish
def create_mobilenetv2_mish(num_classes, pretrained=True, width_mult=0.75):
    """
    Create a MobileNetV2 model with Mish activation.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        width_mult (float): Width multiplier for the network (default: 0.75)
        
    Returns:
        nn.Module: MobileNetV2 model with Mish activation
    """
    model = create_mobilenetv2(num_classes, pretrained, width_mult)
    model = replace_relu_with_mish(model)
    return model

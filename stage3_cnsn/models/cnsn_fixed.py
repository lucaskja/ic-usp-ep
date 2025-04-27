"""
Implementation of CNSN (CrossNorm and SelfNorm) modules.

Paper: CrossNorm and SelfNorm for Generalization under Distribution Shifts
https://arxiv.org/abs/2102.02811
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNorm(nn.Module):
    """
    CrossNorm module for enlarging training distribution.
    
    Exchanges channel-wise mean and variance between feature maps.
    Only active during training with probability p.
    """
    def __init__(self, p=0.5):
        """
        Initialize CrossNorm module.
        
        Args:
            p (float): Probability of applying CrossNorm during training
        """
        super(CrossNorm, self).__init__()
        self.p = p
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        # Skip during inference or with probability 1-p during training
        if not self.training or torch.rand(1).item() > self.p:
            return x
        
        B, C, H, W = x.size()
        
        # Need at least 2 samples in batch for swapping
        if B < 2:
            return x
            
        # Calculate mean and std across spatial dimensions
        mean = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        std = torch.sqrt(x.var(dim=[2, 3], keepdim=True) + 1e-5)  # [B, C, 1, 1]
        
        # Create random pairs for swapping
        indices = torch.randperm(B)
        
        # Swap statistics between samples
        mean_swap = mean[indices]
        std_swap = std[indices]
        
        # Normalize and denormalize with swapped statistics
        x_norm = (x - mean) / std
        x_crossnorm = x_norm * std_swap + mean_swap
        
        return x_crossnorm


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
        x_selfnorm = x_norm * (std * std_weights) + mean * mean_weights
        
        return x_selfnorm


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

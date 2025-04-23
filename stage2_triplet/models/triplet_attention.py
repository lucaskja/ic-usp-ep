"""
Implementation of Triplet Attention module.

Paper: Rotate to Attend: Convolutional Triplet Attention Module
https://arxiv.org/abs/2010.03045
"""
import torch
import torch.nn as nn


class ZPool(nn.Module):
    """
    Z-pool module that concatenates max and average pooling.
    """
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, 2, H, W)
        """
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return torch.cat([avg_pool, max_pool], dim=1)


class AttentionGate(nn.Module):
    """
    Attention gate module.
    """
    def __init__(self, kernel_size=7):
        """
        Initialize attention gate.
        
        Args:
            kernel_size (int): Kernel size for the convolutional layer
        """
        super(AttentionGate, self).__init__()
        self.zpool = ZPool()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying attention
        """
        z = self.zpool(x)
        z = self.conv(z)
        z = self.bn(z)
        z = torch.sigmoid(z)
        return x * z


class TripletAttention(nn.Module):
    """
    Triplet Attention Module.
    
    This module has three branches:
    1. Channel-Height branch: Captures channel-height interactions
    2. Channel-Width branch: Captures channel-width interactions
    3. Spatial branch: Captures spatial interactions
    """
    def __init__(self, kernel_size=7):
        """
        Initialize Triplet Attention module.
        
        Args:
            kernel_size (int): Kernel size for the convolutional layers
        """
        super(TripletAttention, self).__init__()
        self.ch_gate = AttentionGate(kernel_size)
        self.cw_gate = AttentionGate(kernel_size)
        self.hw_gate = AttentionGate(kernel_size)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        # Original input
        x_perm1 = x
        
        # Channel-Height branch
        x_perm2 = x.permute(0, 2, 1, 3)  # (B, H, C, W)
        x_perm2 = self.ch_gate(x_perm2)
        x_perm2 = x_perm2.permute(0, 2, 1, 3)  # (B, C, H, W)
        
        # Channel-Width branch
        x_perm3 = x.permute(0, 3, 2, 1)  # (B, W, H, C)
        x_perm3 = self.cw_gate(x_perm3)
        x_perm3 = x_perm3.permute(0, 3, 2, 1)  # (B, C, H, W)
        
        # Spatial branch
        x_perm4 = self.hw_gate(x)
        
        # Combine all branches (average)
        x_out = (x_perm2 + x_perm3 + x_perm4) / 3
        
        return x_out

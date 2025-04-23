"""
Implementation of Triplet Attention module.

Paper: Rotate to Attend: Convolutional Triplet Attention Module
https://arxiv.org/abs/2010.03045
"""
import torch
import torch.nn as nn


class ZPool(nn.Module):
    """
    Z-pool module for Triplet Attention.
    Combines max pooling and average pooling along a dimension.
    """
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Concatenated max and average pooled features
        """
        # Apply max pooling and average pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        
        # Concatenate along channel dimension
        return torch.cat([avg_pool, max_pool], dim=1)


class AttentionGate(nn.Module):
    """
    Attention gate for Triplet Attention.
    """
    def __init__(self, kernel_size=7):
        """
        Initialize attention gate.
        
        Args:
            kernel_size (int): Size of convolutional kernel
        """
        super(AttentionGate, self).__init__()
        self.zpool = ZPool()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Attention map
        """
        z_pooled = self.zpool(x)
        conv_out = self.conv(z_pooled)
        attention_map = self.sigmoid(conv_out)
        return attention_map


class TripletAttention(nn.Module):
    """
    Triplet Attention Module.
    
    Captures cross-dimension interactions through a three-branch structure:
    1. Channel-Height interaction
    2. Channel-Width interaction
    3. Height-Width (spatial) interaction
    """
    def __init__(self, kernel_size=7):
        """
        Initialize Triplet Attention module.
        
        Args:
            kernel_size (int): Size of convolutional kernel in attention gates
        """
        super(TripletAttention, self).__init__()
        
        # Branch 1: Channel-Height interaction
        self.ch_gate = AttentionGate(kernel_size)
        
        # Branch 2: Channel-Width interaction
        self.cw_gate = AttentionGate(kernel_size)
        
        # Branch 3: Height-Width (spatial) interaction
        self.hw_gate = AttentionGate(kernel_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        # Original input dimensions
        B, C, H, W = x.size()
        
        # Branch 1: Channel-Height interaction
        # Rotate tensor to [B, C, W, H]
        x_perm1 = x.permute(0, 1, 3, 2)
        # Apply attention
        ch_attention = self.ch_gate(x_perm1)
        # Rotate back and apply attention
        ch_attention = ch_attention.permute(0, 1, 3, 2)
        x_ch = x * ch_attention
        
        # Branch 2: Channel-Width interaction
        # Rotate tensor to [B, H, C, W]
        x_perm2 = x.permute(0, 2, 1, 3)
        # Apply attention
        cw_attention = self.cw_gate(x_perm2)
        # Rotate back and apply attention
        cw_attention = cw_attention.permute(0, 2, 1, 3)
        x_cw = x * cw_attention
        
        # Branch 3: Height-Width (spatial) interaction
        hw_attention = self.hw_gate(x)
        x_hw = x * hw_attention
        
        # Combine all branches (average)
        x_out = (x_ch + x_cw + x_hw) / 3
        
        return x_out

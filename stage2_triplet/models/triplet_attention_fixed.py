"""
Implementation of Triplet Attention module.

Paper: Rotate to Attend: Convolutional Triplet Attention Module
https://arxiv.org/abs/2010.03045
"""
import torch
import torch.nn as nn


class Z_Pool(nn.Module):
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
        # Return shape: [B, 2, H, W]
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), 
                         torch.mean(x, 1).unsqueeze(1)), dim=1)


class TripletAttention(nn.Module):
    """
    Triplet Attention Module.
    
    Captures cross-dimension interactions through a three-branch structure:
    1. Branch 1: Spatial H -> Channel (with tensor rotation)
    2. Branch 2: Spatial W -> Channel (with tensor rotation)
    3. Branch 3: Channel -> Spatial
    """
    def __init__(self, kernel_size=7):
        """
        Initialize Triplet Attention module.
        
        Args:
            kernel_size (int): Size of convolutional kernel in attention gates
        """
        super(TripletAttention, self).__init__()
        
        # Branch 1: Spatial H -> Channel interaction
        self.branch1 = nn.Sequential(
            Z_Pool(),
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Branch 2: Spatial W -> Channel interaction
        self.branch2 = nn.Sequential(
            Z_Pool(),
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Branch 3: Channel -> Spatial interaction
        self.branch3 = nn.Sequential(
            Z_Pool(),
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(1),
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
        # Original input dimensions
        B, C, H, W = x.size()
        
        # Branch 1: Spatial H -> Channel interaction
        # Rotate tensor to [B, H, C, W]
        x_perm1 = x.permute(0, 2, 1, 3)
        # Apply attention
        attn1 = self.branch1(x_perm1)
        # Rotate back to [B, C, H, W]
        attn1 = attn1.permute(0, 2, 1, 3)
        
        # Branch 2: Spatial W -> Channel interaction
        # Rotate tensor to [B, W, H, C]
        x_perm2 = x.permute(0, 3, 2, 1)
        # Apply attention
        attn2 = self.branch2(x_perm2)
        # Rotate back to [B, C, H, W]
        attn2 = attn2.permute(0, 3, 2, 1)
        
        # Branch 3: Channel -> Spatial interaction
        attn3 = self.branch3(x)
        
        # Combine all branches (average)
        x_out = (x * attn1 + x * attn2 + x * attn3) / 3
        
        return x_out

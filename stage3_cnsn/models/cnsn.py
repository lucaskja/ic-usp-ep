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
    CrossNorm module.
    
    Exchanges channel-wise mean and variance between feature maps.
    """
    def __init__(self, mode='1-instance'):
        """
        Initialize CrossNorm module.
        
        Args:
            mode (str): Mode for CrossNorm operation
                '1-instance': Exchange between channels within same instance
                '2-instance': Exchange between corresponding channels of different instances
                'crop': Apply to specific spatial regions
        """
        super(CrossNorm, self).__init__()
        self.mode = mode
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        if not self.training:
            return x
        
        B, C, H, W = x.shape
        
        if self.mode == '1-instance':
            # Compute mean and std for each channel
            mean = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
            std = torch.sqrt(x.var(dim=[2, 3], keepdim=True) + 1e-5)  # (B, C, 1, 1)
            
            # Shuffle mean and std across channels
            perm_idx = torch.randperm(C)
            mean_perm = mean[:, perm_idx]
            std_perm = std[:, perm_idx]
            
            # Normalize and denormalize with shuffled stats
            x_norm = (x - mean) / std
            x_norm = x_norm * std_perm + mean_perm
            
        elif self.mode == '2-instance':
            if B <= 1:
                return x
                
            # Compute mean and std for each channel
            mean = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
            std = torch.sqrt(x.var(dim=[2, 3], keepdim=True) + 1e-5)  # (B, C, 1, 1)
            
            # Shuffle mean and std across instances
            perm_idx = torch.randperm(B)
            mean_perm = mean[perm_idx]
            std_perm = std[perm_idx]
            
            # Normalize and denormalize with shuffled stats
            x_norm = (x - mean) / std
            x_norm = x_norm * std_perm + mean_perm
            
        elif self.mode == 'crop':
            # Randomly select crop region
            crop_size = min(H, W) // 2
            top = torch.randint(0, H - crop_size + 1, (1,)).item()
            left = torch.randint(0, W - crop_size + 1, (1,)).item()
            
            # Extract crop
            crop = x[:, :, top:top+crop_size, left:left+crop_size]
            
            # Compute mean and std for crop
            crop_mean = crop.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
            crop_std = torch.sqrt(crop.var(dim=[2, 3], keepdim=True) + 1e-5)  # (B, C, 1, 1)
            
            # Shuffle mean and std across channels
            perm_idx = torch.randperm(C)
            crop_mean_perm = crop_mean[:, perm_idx]
            crop_std_perm = crop_std[:, perm_idx]
            
            # Apply to original tensor
            x_norm = x.clone()
            x_norm[:, :, top:top+crop_size, left:left+crop_size] = (
                (crop - crop_mean) / crop_std * crop_std_perm + crop_mean_perm
            )
            
        else:
            raise ValueError(f"Unknown CrossNorm mode: {self.mode}")
            
        return x_norm


class SelfNorm(nn.Module):
    """
    SelfNorm module.
    
    Recalibrates statistics using attention mechanism.
    """
    def __init__(self, num_channels):
        """
        Initialize SelfNorm module.
        
        Args:
            num_channels (int): Number of input channels
        """
        super(SelfNorm, self).__init__()
        
        # Two FC networks for attention functions f and g
        self.f_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
        self.g_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Compute mean and std for each channel
        mean = x.mean(dim=[2, 3])  # (B, C)
        std = torch.sqrt(x.var(dim=[2, 3]) + 1e-5)  # (B, C)
        
        # Stack mean and std for FC input
        stats = torch.stack([mean, std], dim=2)  # (B, C, 2)
        
        # Apply attention functions
        f_out = self.f_fc(stats).squeeze(-1)  # (B, C)
        g_out = self.g_fc(stats).squeeze(-1)  # (B, C)
        
        # Apply softmax to get attention weights
        f_attn = F.softmax(f_out, dim=1)  # (B, C)
        g_attn = F.softmax(g_out, dim=1)  # (B, C)
        
        # Compute weighted mean and std
        mean_attn = torch.sum(mean * f_attn, dim=1, keepdim=True)  # (B, 1)
        std_attn = torch.sum(std * g_attn, dim=1, keepdim=True)  # (B, 1)
        
        # Reshape for broadcasting
        mean_attn = mean_attn.view(B, 1, 1, 1)
        std_attn = std_attn.view(B, 1, 1, 1)
        
        # Normalize and denormalize with attended stats
        x_norm = (x - mean.view(B, C, 1, 1)) / std.view(B, C, 1, 1)
        x_norm = x_norm * std_attn + mean_attn
        
        return x_norm


class CNSN(nn.Module):
    """
    CNSN (CrossNorm and SelfNorm) module.
    
    Combines CrossNorm and SelfNorm modules.
    """
    def __init__(self, num_channels, cn_mode='1-instance'):
        """
        Initialize CNSN module.
        
        Args:
            num_channels (int): Number of input channels
            cn_mode (str): Mode for CrossNorm operation
        """
        super(CNSN, self).__init__()
        self.cross_norm = CrossNorm(mode=cn_mode)
        self.self_norm = SelfNorm(num_channels)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        # Apply CrossNorm (only during training)
        x_cn = self.cross_norm(x)
        
        # Apply SelfNorm (during both training and testing)
        x_sn = self.self_norm(x_cn)
        
        return x_sn

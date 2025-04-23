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
    """
    def __init__(self, mode='2-instance', p=0.5, crop_size=None):
        """
        Initialize CrossNorm module.
        
        Args:
            mode (str): Mode of operation:
                - '1-instance': Exchange between channels within same instance
                - '2-instance': Exchange between corresponding channels of different instances
                - 'crop': Apply to specific spatial regions
            p (float): Probability of applying CrossNorm during training
            crop_size (tuple): Size of crop region for 'crop' mode (H, W)
        """
        super(CrossNorm, self).__init__()
        assert mode in ['1-instance', '2-instance', 'crop'], f"Invalid mode: {mode}"
        self.mode = mode
        self.p = p
        self.crop_size = crop_size
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        # Only apply during training with probability p
        if not self.training or torch.rand(1).item() > self.p:
            return x
        
        B, C, H, W = x.size()
        
        if self.mode == '1-instance':
            # Randomly permute channels within each instance
            perm_indices = torch.randperm(C)
            mean = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
            std = torch.sqrt(x.var(dim=[2, 3], keepdim=True) + 1e-5)  # [B, C, 1, 1]
            
            # Get permuted statistics
            mean_perm = mean[:, perm_indices]
            std_perm = std[:, perm_indices]
            
            # Normalize and denormalize with permuted statistics
            x_norm = (x - mean) / std
            x_crossnorm = x_norm * std_perm + mean_perm
            
            return x_crossnorm
            
        elif self.mode == '2-instance':
            if B <= 1:
                return x  # Need at least 2 instances
                
            # Shuffle batch indices
            perm_indices = torch.randperm(B)
            
            mean = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
            std = torch.sqrt(x.var(dim=[2, 3], keepdim=True) + 1e-5)  # [B, C, 1, 1]
            
            # Get statistics from shuffled batch
            mean_perm = mean[perm_indices]
            std_perm = std[perm_indices]
            
            # Normalize and denormalize with shuffled statistics
            x_norm = (x - mean) / std
            x_crossnorm = x_norm * std_perm + mean_perm
            
            return x_crossnorm
            
        elif self.mode == 'crop':
            if self.crop_size is None:
                crop_h, crop_w = H // 2, W // 2
            else:
                crop_h, crop_w = self.crop_size
                
            # Ensure crop size is valid
            if H <= crop_h or W <= crop_w:
                return x  # Skip if crop size is too large
                
            # Randomly select crop region
            h_start = torch.randint(0, H - crop_h + 1, (1,)).item()
            w_start = torch.randint(0, W - crop_w + 1, (1,)).item()
            
            # Extract crop region
            crop = x[:, :, h_start:h_start+crop_h, w_start:w_start+crop_w]
            
            # Compute statistics for crop region
            crop_mean = crop.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
            crop_std = torch.sqrt(crop.var(dim=[2, 3], keepdim=True) + 1e-5)  # [B, C, 1, 1]
            
            # Shuffle batch indices for crop statistics
            if B > 1:  # Need at least 2 instances for shuffling
                perm_indices = torch.randperm(B)
                crop_mean_perm = crop_mean[perm_indices]
                crop_std_perm = crop_std[perm_indices]
                
                # Apply CrossNorm only to crop region
                x_result = x.clone()
                crop_norm = (crop - crop_mean) / crop_std
                crop_crossnorm = crop_norm * crop_std_perm + crop_mean_perm
                
                # Replace crop region with CrossNorm result
                x_result[:, :, h_start:h_start+crop_h, w_start:w_start+crop_w] = crop_crossnorm
                
                return x_result
            else:
                return x  # Skip if batch size is 1


class SelfNorm(nn.Module):
    """
    SelfNorm module for bridging train-test distribution gap.
    
    Recalibrates statistics using attention mechanism.
    """
    def __init__(self, num_features, eps=1e-5):
        """
        Initialize SelfNorm module.
        
        Args:
            num_features (int): Number of input channels
            eps (float): Small constant for numerical stability
        """
        super(SelfNorm, self).__init__()
        self.eps = eps
        
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
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        B, C, H, W = x.size()
        
        # Compute mean and standard deviation
        mean = x.mean(dim=[2, 3])  # [B, C]
        std = torch.sqrt(x.var(dim=[2, 3]) + self.eps)  # [B, C]
        
        # Concatenate mean and std for each channel
        mean_std = torch.stack([mean, std], dim=2)  # [B, C, 2]
        
        # Compute attention weights for mean and std
        f_weights = self.f_fc(mean_std).squeeze(2)  # [B, C]
        g_weights = self.g_fc(mean_std).squeeze(2)  # [B, C]
        
        # Apply softmax to get normalized weights
        f_weights = F.softmax(f_weights, dim=1)  # [B, C]
        g_weights = F.softmax(g_weights, dim=1)  # [B, C]
        
        # Compute weighted mean and std
        mean_weighted = torch.sum(f_weights.unsqueeze(2).unsqueeze(3) * mean.unsqueeze(2).unsqueeze(3), dim=1, keepdim=True)  # [B, 1, 1, 1]
        std_weighted = torch.sum(g_weights.unsqueeze(2).unsqueeze(3) * std.unsqueeze(2).unsqueeze(3), dim=1, keepdim=True)  # [B, 1, 1, 1]
        
        # Normalize and denormalize with weighted statistics
        x_norm = (x - mean.unsqueeze(2).unsqueeze(3)) / std.unsqueeze(2).unsqueeze(3)
        x_selfnorm = x_norm * std_weighted + mean_weighted
        
        return x_selfnorm


class CNSN(nn.Module):
    """
    CNSN (CrossNorm and SelfNorm) module.
    
    Combines CrossNorm for training distribution expansion and
    SelfNorm for bridging train-test distribution gap.
    """
    def __init__(self, num_features, crossnorm_mode='2-instance', p=0.5, crop_size=None):
        """
        Initialize CNSN module.
        
        Args:
            num_features (int): Number of input channels
            crossnorm_mode (str): Mode for CrossNorm
            p (float): Probability of applying CrossNorm during training
            crop_size (tuple): Size of crop region for 'crop' mode
        """
        super(CNSN, self).__init__()
        self.crossnorm = CrossNorm(mode=crossnorm_mode, p=p, crop_size=crop_size)
        self.selfnorm = SelfNorm(num_features)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Apply CrossNorm (only during training)
        x = self.crossnorm(x)
        
        # Apply SelfNorm (both training and testing)
        x = self.selfnorm(x)
        
        return x

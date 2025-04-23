# Detailed Model Descriptions

This document provides in-depth explanations of each model implementation in our project, including their architectural details, improvements, and expected benefits.

## Base MobileNetV2

### Architecture Overview

MobileNetV2 is a lightweight convolutional neural network architecture designed for mobile and embedded vision applications. It builds upon the ideas introduced in MobileNetV1 with several key improvements.

### Key Components

1. **Inverted Residual Structure**:
   - Unlike traditional residual connections that go from wide → narrow → wide
   - MobileNetV2 uses narrow → wide → narrow (inverted) structure
   - Expands the input before the depthwise convolution, then projects back to a smaller dimension

2. **Depthwise Separable Convolutions**:
   - Factorizes standard convolution into:
     - Depthwise convolution: applies a single filter per input channel
     - Pointwise convolution: 1×1 convolution to combine the outputs
   - Significantly reduces computation and model size

3. **Linear Bottlenecks**:
   - Uses linear activation (no ReLU) in the bottleneck layers
   - Preserves information in low-dimensional spaces

4. **ReLU6 Activation**:
   - ReLU capped at 6: min(max(0, x), 6)
   - Provides robustness for low-precision computation

### Implementation Details

Our base MobileNetV2 implementation uses the pre-trained model from PyTorch's torchvision library, with the classifier head modified for our leaf disease classification task:

```python
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def create_base_mobilenetv2(num_classes):
    # Load pre-trained MobileNetV2
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Modify the classifier for our task
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model
```

## Stage 1: MobileNetV2 with Mish Activation

### Mish Activation Function

Mish is a self-regularized non-monotonic activation function defined as:

```
f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
```

### Key Advantages Over ReLU6

1. **Smoothness**: Mish is smooth and non-monotonic, allowing better information flow
2. **Unbounded Above**: Unlike ReLU6, Mish has no upper bound, preserving important features
3. **Bounded Below**: Slightly bounded below, providing regularization benefits
4. **Better Gradient Flow**: Reduces the vanishing gradient problem
5. **Preserves Slight Negative Values**: Can help preserve important negative information

### Implementation Details

We replace all ReLU6 activations in MobileNetV2 with Mish:

```python
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def convert_to_mish(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU6):
            setattr(model, name, Mish())
        else:
            convert_to_mish(module)
    return model

def create_mish_mobilenetv2(num_classes):
    # Load base model
    model = create_base_mobilenetv2(num_classes)
    
    # Convert ReLU6 to Mish
    model = convert_to_mish(model)
    
    return model
```

## Stage 2: MobileNetV2 with Mish and Triplet Attention

### Triplet Attention Mechanism

Triplet Attention is a lightweight attention mechanism that captures cross-dimension interactions through a three-branch structure.

### Key Components

1. **Three Parallel Branches**:
   - Each branch focuses on different dimensional interactions
   - Branch 1: Channel-Height interaction
   - Branch 2: Channel-Width interaction
   - Branch 3: Standard spatial attention

2. **Z-Pool Operation**:
   - Combines max-pooling and average-pooling along the channel dimension
   - Captures both prominent and overall feature distributions

3. **No Channel Reduction**:
   - Unlike many attention mechanisms, Triplet Attention doesn't reduce channel dimensions
   - Preserves full information flow

### Implementation Details

```python
class ZPool(nn.Module):
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return torch.cat([avg_pool, max_pool], dim=1)

class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TripletAttention, self).__init__()
        
        # Common components
        self.kernel_size = kernel_size
        
        # Branch 1: Channel-Height interaction
        self.ch_branch = nn.Sequential(
            ZPool(),
            nn.Conv2d(2, 1, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Branch 2: Channel-Width interaction
        self.cw_branch = nn.Sequential(
            ZPool(),
            nn.Conv2d(2, 1, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Branch 3: Spatial attention
        self.hw_branch = nn.Sequential(
            ZPool(),
            nn.Conv2d(2, 1, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Original tensor shape: [B, C, H, W]
        
        # Branch 1: Channel-Height interaction
        x_perm1 = x.permute(0, 2, 1, 3)  # [B, H, C, W]
        ch_att = self.ch_branch(x_perm1)
        ch_att = ch_att.permute(0, 2, 1, 3)  # [B, 1, C, W]
        x_ch = x * ch_att
        
        # Branch 2: Channel-Width interaction
        x_perm2 = x.permute(0, 3, 2, 1)  # [B, W, H, C]
        cw_att = self.cw_branch(x_perm2)
        cw_att = cw_att.permute(0, 3, 2, 1)  # [B, C, H, 1]
        x_cw = x * cw_att
        
        # Branch 3: Spatial attention
        hw_att = self.hw_branch(x)
        x_hw = x * hw_att
        
        # Combine branches
        x_out = (x_ch + x_cw + x_hw) / 3
        
        return x_out
```

### Integration with MobileNetV2

Triplet Attention is added after each inverted residual block in the MobileNetV2 architecture.

## Stage 3: MobileNetV2 with Mish, Triplet Attention, and CNSN

### CNSN (CrossNorm and SelfNorm)

CNSN combines two complementary normalization techniques:

1. **CrossNorm (CN)**:
   - Enlarges the training distribution by exchanging channel-wise statistics between feature maps
   - Only active during training
   - Creates diverse feature representations

2. **SelfNorm (SN)**:
   - Bridges the train-test distribution gap
   - Uses an attention mechanism to recalibrate statistics
   - Active during both training and testing

### CrossNorm Implementation

```python
class CrossNorm(nn.Module):
    def __init__(self, mode='2-instance'):
        super(CrossNorm, self).__init__()
        self.mode = mode  # '1-instance', '2-instance', or 'crop'
    
    def forward(self, x):
        if not self.training:
            return x
        
        B, C, H, W = x.size()
        
        if self.mode == '1-instance':
            # Exchange statistics within the same instance
            perm_idx = torch.randperm(C)
            mean = x.mean(dim=[2, 3], keepdim=True)
            std = x.std(dim=[2, 3], keepdim=True) + 1e-5
            
            mean_perm = mean[:, perm_idx, :, :]
            std_perm = std[:, perm_idx, :, :]
            
            # Normalize and denormalize with exchanged statistics
            x_norm = (x - mean) / std
            x_out = x_norm * std_perm + mean_perm
            
        elif self.mode == '2-instance':
            # Exchange statistics between different instances
            if B > 1:
                perm_idx = torch.randperm(B)
                mean = x.mean(dim=[2, 3], keepdim=True)
                std = x.std(dim=[2, 3], keepdim=True) + 1e-5
                
                mean_perm = mean[perm_idx]
                std_perm = std[perm_idx]
                
                # Normalize and denormalize with exchanged statistics
                x_norm = (x - mean) / std
                x_out = x_norm * std_perm + mean_perm
            else:
                x_out = x
                
        elif self.mode == 'crop':
            # Apply to specific spatial regions
            crop_size = min(H, W) // 2
            h_start = torch.randint(0, H - crop_size + 1, (1,)).item()
            w_start = torch.randint(0, W - crop_size + 1, (1,)).item()
            
            crop = x[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
            
            mean_crop = crop.mean(dim=[2, 3], keepdim=True)
            std_crop = crop.std(dim=[2, 3], keepdim=True) + 1e-5
            
            perm_idx = torch.randperm(B) if B > 1 else torch.arange(B)
            mean_perm = mean_crop[perm_idx]
            std_perm = std_crop[perm_idx]
            
            # Apply only to the crop region
            x_out = x.clone()
            crop_norm = (crop - mean_crop) / std_crop
            crop_out = crop_norm * std_perm + mean_perm
            x_out[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size] = crop_out
            
        return x_out
```

### SelfNorm Implementation

```python
class SelfNorm(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SelfNorm, self).__init__()
        self.attention_f = nn.Sequential(
            nn.Linear(2, reduction),
            nn.ReLU(),
            nn.Linear(reduction, 1),
            nn.Sigmoid()
        )
        
        self.attention_g = nn.Sequential(
            nn.Linear(2, reduction),
            nn.ReLU(),
            nn.Linear(reduction, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Calculate channel-wise statistics
        mean = x.mean(dim=[2, 3])  # [B, C]
        std = x.std(dim=[2, 3]) + 1e-5  # [B, C]
        
        # Process each channel independently
        mean_weights = []
        std_weights = []
        
        for i in range(C):
            # Get statistics for this channel
            mean_i = mean[:, i].unsqueeze(1)  # [B, 1]
            std_i = std[:, i].unsqueeze(1)  # [B, 1]
            stats = torch.cat([mean_i, std_i], dim=1)  # [B, 2]
            
            # Calculate attention weights
            f_weight = self.attention_f(stats)  # [B, 1]
            g_weight = self.attention_g(stats)  # [B, 1]
            
            mean_weights.append(f_weight)
            std_weights.append(g_weight)
        
        # Stack weights for all channels
        mean_weights = torch.stack(mean_weights, dim=1)  # [B, C, 1]
        std_weights = torch.stack(std_weights, dim=1)  # [B, C, 1]
        
        # Apply weighted normalization
        mean = mean.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        std = std.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        mean_weights = mean_weights.unsqueeze(3)  # [B, C, 1, 1]
        std_weights = std_weights.unsqueeze(3)  # [B, C, 1, 1]
        
        x_norm = (x - mean) / std
        x_out = x_norm * std_weights + mean * mean_weights
        
        return x_out
```

### CNSN Module

```python
class CNSN(nn.Module):
    def __init__(self, channels, cn_mode='2-instance'):
        super(CNSN, self).__init__()
        self.cn = CrossNorm(mode=cn_mode)
        self.sn = SelfNorm(channels)
    
    def forward(self, x):
        x = self.cn(x)
        x = self.sn(x)
        return x
```

### Integration with MobileNetV2

CNSN modules are added after each inverted residual block, following the Triplet Attention module.

## Expected Performance Improvements

| Model | Expected Improvement |
|-------|----------------------|
| Base MobileNetV2 | Baseline |
| MobileNetV2 + Mish | +1-2% accuracy, better convergence |
| MobileNetV2 + Mish + Triplet Attention | +2-3% accuracy, better feature representation |
| MobileNetV2 + Mish + Triplet Attention + CNSN | +3-5% accuracy, better generalization |

Each improvement builds upon the previous one, with the final model expected to provide the best performance in terms of accuracy and generalization to unseen data.

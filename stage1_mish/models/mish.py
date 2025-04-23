"""
Implementation of Mish activation function.

Paper: Mish: A Self Regularized Non-Monotonic Neural Activation Function
https://arxiv.org/abs/1908.08681
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """
    Mish activation function.
    
    Formula: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying Mish activation
        """
        return x * torch.tanh(F.softplus(x))


def replace_relu_with_mish(model):
    """
    Replace all ReLU and ReLU6 activations in a model with Mish.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        nn.Module: Model with ReLU/ReLU6 replaced by Mish
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            setattr(model, name, Mish())
        else:
            replace_relu_with_mish(module)
    
    return model

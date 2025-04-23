"""
Tests for Mish activation function.
"""
import torch
import pytest
import numpy as np
from stage1_mish.models.mish import Mish, replace_relu_with_mish


def test_mish_forward():
    """Test forward pass of Mish activation function."""
    mish = Mish()
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Expected values calculated using the formula: x * tanh(softplus(x))
    expected = x * torch.tanh(torch.log(1 + torch.exp(x)))
    
    output = mish(x)
    
    assert torch.allclose(output, expected)


def test_mish_zero():
    """Test Mish activation at x=0."""
    mish = Mish()
    x = torch.tensor([0.0])
    
    # At x=0, Mish should be 0
    output = mish(x)
    
    assert torch.allclose(output, torch.tensor([0.0]))


def test_mish_positive():
    """Test Mish activation for positive values."""
    mish = Mish()
    x = torch.tensor([1.0, 2.0, 3.0])
    
    # For positive x, Mish should be positive and close to x for large x
    output = mish(x)
    
    assert torch.all(output > 0)
    assert torch.all(output < x)  # Mish(x) < x for positive x


def test_mish_negative():
    """Test Mish activation for negative values."""
    mish = Mish()
    x = torch.tensor([-3.0, -2.0, -1.0])
    
    # For negative x, Mish can be negative but with smaller magnitude
    output = mish(x)
    
    assert torch.all(output > x)  # Mish(x) > x for negative x


def test_mish_gradient():
    """Test that Mish gradient flows properly."""
    mish = Mish()
    x = torch.tensor([1.0], requires_grad=True)
    
    output = mish(x)
    output.backward()
    
    # Gradient should exist and not be zero
    assert x.grad is not None
    assert x.grad.item() != 0


class SimpleModel(torch.nn.Module):
    """Simple model for testing replace_relu_with_mish function."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.ReLU6(),
            torch.nn.BatchNorm2d(3)
        )
    
    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = self.relu2(self.bn(x))
        x = self.seq(x)
        return x


def test_replace_relu_with_mish():
    """Test replace_relu_with_mish function."""
    model = SimpleModel()
    
    # Count ReLU and ReLU6 modules before replacement
    relu_count_before = sum(1 for m in model.modules() if isinstance(m, (torch.nn.ReLU, torch.nn.ReLU6)))
    
    # Replace ReLU with Mish
    model = replace_relu_with_mish(model)
    
    # Count Mish modules after replacement
    mish_count_after = sum(1 for m in model.modules() if isinstance(m, Mish))
    
    # Count ReLU and ReLU6 modules after replacement
    relu_count_after = sum(1 for m in model.modules() if isinstance(m, (torch.nn.ReLU, torch.nn.ReLU6)))
    
    # Check that all ReLU modules were replaced
    assert relu_count_before > 0
    assert relu_count_after == 0
    assert mish_count_after == relu_count_before

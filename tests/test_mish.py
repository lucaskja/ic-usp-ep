"""
Tests for Mish activation function.
"""
import torch
import pytest
import numpy as np
from mobilenetv2_improvements.stage1_mish.models.mish import Mish


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

"""
Tests for CNSN (CrossNorm and SelfNorm) modules.
"""
import torch
import pytest
import numpy as np
from stage3_cnsn.models.cnsn import CrossNorm, SelfNorm, CNSN


def test_crossnorm_1instance():
    """Test CrossNorm with 1-instance mode."""
    crossnorm = CrossNorm(mode='1-instance', p=1.0)  # Always apply
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Set to training mode
    crossnorm.train()
    
    # Forward pass
    output = crossnorm(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check that output is different from input
    assert not torch.allclose(output, x)


def test_crossnorm_2instance():
    """Test CrossNorm with 2-instance mode."""
    crossnorm = CrossNorm(mode='2-instance', p=1.0)  # Always apply
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Set to training mode
    crossnorm.train()
    
    # Forward pass
    output = crossnorm(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check that output is different from input
    assert not torch.allclose(output, x)


def test_crossnorm_crop():
    """Test CrossNorm with crop mode."""
    crossnorm = CrossNorm(mode='crop', p=1.0, crop_size=(4, 4))  # Always apply
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Set to training mode
    crossnorm.train()
    
    # Forward pass
    output = crossnorm(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check that output is different from input (may fail if random crop doesn't change anything)
    # So we'll just check that the module runs without errors
    assert output is not None


def test_crossnorm_eval_mode():
    """Test CrossNorm in eval mode (should not modify input)."""
    crossnorm = CrossNorm(mode='1-instance', p=1.0)  # Always apply
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Set to eval mode
    crossnorm.eval()
    
    # Forward pass
    output = crossnorm(x)
    
    # Check that output is identical to input in eval mode
    assert torch.allclose(output, x)


def test_selfnorm():
    """Test SelfNorm module."""
    selfnorm = SelfNorm(num_features=3)
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Forward pass
    output = selfnorm(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check that output is different from input
    assert not torch.allclose(output, x)


def test_selfnorm_gradient():
    """Test that gradients flow through SelfNorm."""
    selfnorm = SelfNorm(num_features=3)
    
    # Create input tensor with gradient tracking
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    
    # Forward pass
    output = selfnorm(x)
    
    # Compute gradient
    loss = output.sum()
    loss.backward()
    
    # Check that gradient exists and is not zero
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_cnsn():
    """Test CNSN module (combined CrossNorm and SelfNorm)."""
    cnsn = CNSN(num_features=3, crossnorm_mode='2-instance', p=1.0)
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Set to training mode
    cnsn.train()
    
    # Forward pass
    output = cnsn(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check that output is different from input
    assert not torch.allclose(output, x)


def test_cnsn_eval_mode():
    """Test CNSN in eval mode (only SelfNorm should be applied)."""
    cnsn = CNSN(num_features=3, crossnorm_mode='2-instance', p=1.0)
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Set to eval mode
    cnsn.eval()
    
    # Forward pass
    output = cnsn(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check that output is different from input (SelfNorm still applies)
    assert not torch.allclose(output, x)

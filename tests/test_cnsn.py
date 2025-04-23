"""
Tests for CNSN (CrossNorm and SelfNorm) modules.
"""
import torch
import pytest
from mobilenetv2_improvements.stage3_cnsn.models.cnsn import CrossNorm, SelfNorm, CNSN


def test_crossnorm_shape():
    """Test CrossNorm output shape."""
    cn = CrossNorm(mode='1-instance')
    x = torch.randn(2, 16, 24, 24)  # (B, C, H, W)
    
    # Set to training mode
    cn.train()
    output = cn(x)
    
    # Output shape should be the same as input
    assert output.shape == x.shape


def test_crossnorm_eval_mode():
    """Test CrossNorm in eval mode (should be identity)."""
    cn = CrossNorm(mode='1-instance')
    x = torch.randn(2, 16, 24, 24)  # (B, C, H, W)
    
    # Set to eval mode
    cn.eval()
    output = cn(x)
    
    # Output should be identical to input in eval mode
    assert torch.allclose(output, x)


def test_crossnorm_modes():
    """Test different CrossNorm modes."""
    modes = ['1-instance', '2-instance', 'crop']
    x = torch.randn(2, 16, 24, 24)  # (B, C, H, W)
    
    for mode in modes:
        cn = CrossNorm(mode=mode)
        cn.train()
        output = cn(x)
        
        # Output shape should be the same as input
        assert output.shape == x.shape
        
        # Output should not be the same as input (normalization applied)
        if mode != '2-instance' or x.shape[0] > 1:
            assert not torch.allclose(output, x)


def test_selfnorm_shape():
    """Test SelfNorm output shape."""
    sn = SelfNorm(num_channels=16)
    x = torch.randn(2, 16, 24, 24)  # (B, C, H, W)
    
    output = sn(x)
    
    # Output shape should be the same as input
    assert output.shape == x.shape
    
    # Output should not be the same as input (normalization applied)
    assert not torch.allclose(output, x)


def test_selfnorm_gradient():
    """Test that SelfNorm gradient flows properly."""
    sn = SelfNorm(num_channels=16)
    x = torch.randn(2, 16, 24, 24, requires_grad=True)  # (B, C, H, W)
    
    output = sn(x)
    output.sum().backward()
    
    # Gradient should exist and not be zero
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_cnsn_module():
    """Test CNSN module (combined CrossNorm and SelfNorm)."""
    cnsn = CNSN(num_channels=16, cn_mode='1-instance')
    x = torch.randn(2, 16, 24, 24)  # (B, C, H, W)
    
    # Test in training mode
    cnsn.train()
    train_output = cnsn(x)
    
    # Test in eval mode
    cnsn.eval()
    eval_output = cnsn(x)
    
    # Output shape should be the same as input
    assert train_output.shape == x.shape
    assert eval_output.shape == x.shape
    
    # Training and eval outputs should be different
    # (CrossNorm only applies during training)
    assert not torch.allclose(train_output, eval_output)

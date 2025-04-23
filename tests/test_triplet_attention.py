"""
Tests for Triplet Attention module.
"""
import torch
import pytest
from mobilenetv2_improvements.stage2_triplet.models.triplet_attention import (
    ZPool, AttentionGate, TripletAttention
)


def test_zpool_shape():
    """Test ZPool output shape."""
    zpool = ZPool()
    x = torch.randn(2, 16, 24, 24)  # (B, C, H, W)
    
    output = zpool(x)
    
    # Output should have 2 channels (avg_pool and max_pool)
    assert output.shape == (2, 2, 24, 24)


def test_attention_gate():
    """Test AttentionGate module."""
    gate = AttentionGate(kernel_size=3)
    x = torch.randn(2, 16, 24, 24)  # (B, C, H, W)
    
    output = gate(x)
    
    # Output shape should be the same as input
    assert output.shape == x.shape
    
    # Output should not be the same as input (attention applied)
    assert not torch.allclose(output, x)


def test_triplet_attention():
    """Test TripletAttention module."""
    triplet = TripletAttention(kernel_size=3)
    x = torch.randn(2, 16, 24, 24)  # (B, C, H, W)
    
    output = triplet(x)
    
    # Output shape should be the same as input
    assert output.shape == x.shape
    
    # Output should not be the same as input (attention applied)
    assert not torch.allclose(output, x)


def test_triplet_attention_gradient():
    """Test that TripletAttention gradient flows properly."""
    triplet = TripletAttention(kernel_size=3)
    x = torch.randn(2, 16, 24, 24, requires_grad=True)  # (B, C, H, W)
    
    output = triplet(x)
    output.sum().backward()
    
    # Gradient should exist and not be zero
    assert x.grad is not None
    assert torch.any(x.grad != 0)

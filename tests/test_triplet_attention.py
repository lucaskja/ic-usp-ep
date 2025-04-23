"""
Tests for Triplet Attention module.
"""
import torch
import pytest
import numpy as np
from stage2_triplet.models.triplet_attention import ZPool, AttentionGate, TripletAttention
from stage2_triplet.models.mobilenetv2_triplet import (
    InvertedResidualWithTripletAttention,
    add_triplet_attention_to_mobilenetv2,
    MobileNetV2TripletModel
)


def test_zpool():
    """Test ZPool module."""
    zpool = ZPool()
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Forward pass
    output = zpool(x)
    
    # Check output shape
    assert output.shape == (2, 2, 16, 16)
    
    # Check that output contains max and avg pooled features
    max_pool = torch.max(x, dim=1, keepdim=True)[0]
    avg_pool = torch.mean(x, dim=1, keepdim=True)
    expected = torch.cat([avg_pool, max_pool], dim=1)
    
    assert torch.allclose(output, expected)


def test_attention_gate():
    """Test AttentionGate module."""
    gate = AttentionGate(kernel_size=3)
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Forward pass
    output = gate(x)
    
    # Check output shape
    assert output.shape == (2, 1, 16, 16)
    
    # Check that output values are between 0 and 1 (sigmoid output)
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_triplet_attention():
    """Test TripletAttention module."""
    attention = TripletAttention(kernel_size=3)
    
    # Create input tensor [batch, channels, height, width]
    x = torch.randn(2, 3, 16, 16)
    
    # Forward pass
    output = attention(x)
    
    # Check output shape matches input shape
    assert output.shape == x.shape
    
    # Check that output is different from input (attention applied)
    assert not torch.allclose(output, x)


def test_triplet_attention_gradient():
    """Test that gradients flow through TripletAttention."""
    attention = TripletAttention(kernel_size=3)
    
    # Create input tensor with gradient tracking
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    
    # Forward pass
    output = attention(x)
    
    # Compute gradient
    loss = output.sum()
    loss.backward()
    
    # Check that gradient exists and is not zero
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_inverted_residual_with_triplet_attention():
    """Test InvertedResidualWithTripletAttention module."""
    # Create a mock inverted residual block
    class MockInvertedResidual(torch.nn.Module):
        def __init__(self, use_res_connect=True):
            super(MockInvertedResidual, self).__init__()
            self.use_res_connect = use_res_connect
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        
        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)
    
    # Test with residual connection
    mock_block = MockInvertedResidual(use_res_connect=True)
    block_with_attention = InvertedResidualWithTripletAttention(mock_block)
    
    x = torch.randn(2, 3, 16, 16)
    output = block_with_attention(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Test without residual connection
    mock_block = MockInvertedResidual(use_res_connect=False)
    block_with_attention = InvertedResidualWithTripletAttention(mock_block)
    
    output = block_with_attention(x)
    
    # Check output shape
    assert output.shape == x.shape


def test_mobilenetv2_triplet_model():
    """Test MobileNetV2TripletModel."""
    # Create a small model for testing
    model = MobileNetV2TripletModel(num_classes=10, pretrained=False)
    
    # Check that model can be created
    assert model is not None
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    # Set model to eval mode to avoid batch norm issues
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (2, 10)

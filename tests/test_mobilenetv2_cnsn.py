"""
Tests for MobileNetV2 with CNSN.
"""
import torch
import pytest
from stage3_cnsn.models.mobilenetv2_cnsn import (
    get_output_channels,
    InvertedResidualWithTripletAttentionAndCNSN,
    MobileNetV2CNSNModel
)


class MockInvertedResidual(torch.nn.Module):
    """Mock inverted residual block for testing."""
    def __init__(self, in_channels=3, out_channels=16, use_res_connect=True):
        super(MockInvertedResidual, self).__init__()
        self.use_res_connect = use_res_connect
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU6()
        )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def test_get_output_channels():
    """Test get_output_channels function."""
    # Test with mock inverted residual block
    mock_block = MockInvertedResidual(in_channels=3, out_channels=16)
    channels = get_output_channels(mock_block)
    assert channels == 16


def test_inverted_residual_with_triplet_attention_and_cnsn():
    """Test InvertedResidualWithTripletAttentionAndCNSN module."""
    # Test with residual connection - use same in/out channels for residual connection
    mock_block = MockInvertedResidual(in_channels=16, out_channels=16, use_res_connect=True)
    block_with_attention_cnsn = InvertedResidualWithTripletAttentionAndCNSN(mock_block, 16)
    
    x = torch.randn(2, 16, 16, 16)
    output = block_with_attention_cnsn(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Test without residual connection
    mock_block = MockInvertedResidual(in_channels=3, out_channels=16, use_res_connect=False)
    block_with_attention_cnsn = InvertedResidualWithTripletAttentionAndCNSN(mock_block, 16)
    
    x = torch.randn(2, 3, 16, 16)
    output = block_with_attention_cnsn(x)
    
    # Check output shape
    assert output.shape == (2, 16, 16, 16)  # Output channels changed to 16


def test_mobilenetv2_cnsn_model_creation():
    """Test MobileNetV2CNSNModel creation."""
    # Skip pretrained test due to certificate issues in CI environment
    model = MobileNetV2CNSNModel(num_classes=10, pretrained=False)
    assert model is not None


def test_mobilenetv2_cnsn_model_forward():
    """Test MobileNetV2CNSNModel forward pass."""
    model = MobileNetV2CNSNModel(num_classes=10, pretrained=False)
    
    # Set model to eval mode to avoid batch norm issues
    model.eval()
    
    # Test with a small input to save memory
    x = torch.randn(2, 3, 32, 32)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (2, 10)

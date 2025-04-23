"""
Tests for base MobileNetV2 model.
"""
import torch
import pytest
from base_mobilenetv2.models.mobilenetv2 import MobileNetV2Model, create_mobilenetv2


def test_mobilenetv2_creation():
    """Test MobileNetV2 model creation."""
    # Skip pretrained test due to certificate issues in CI environment
    # model = create_mobilenetv2(num_classes=10, pretrained=True)
    # assert model is not None
    
    # Test without pretrained weights
    model = create_mobilenetv2(num_classes=10, pretrained=False)
    assert model is not None


def test_mobilenetv2_output_shape():
    """Test MobileNetV2 output shape."""
    num_classes = 10
    batch_size = 2
    
    model = MobileNetV2Model(num_classes, pretrained=False)
    x = torch.randn(batch_size, 3, 224, 224)
    
    output = model(x)
    
    assert output.shape == (batch_size, num_classes)


def test_mobilenetv2_forward_pass():
    """Test MobileNetV2 forward pass."""
    num_classes = 10
    
    model = MobileNetV2Model(num_classes, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    
    # Set model to eval mode to avoid batch norm issues
    model.eval()
    
    # Forward pass should not raise an error
    try:
        output = model(x)
        assert True
    except Exception as e:
        assert False, f"Forward pass raised an exception: {e}"


def test_mobilenetv2_classifier():
    """Test MobileNetV2 classifier customization."""
    num_classes = 5
    
    model = MobileNetV2Model(num_classes, pretrained=False)
    
    # Check that the classifier has been modified
    assert isinstance(model.model.classifier, torch.nn.Sequential)
    assert isinstance(model.model.classifier[1], torch.nn.Linear)
    assert model.model.classifier[1].out_features == num_classes

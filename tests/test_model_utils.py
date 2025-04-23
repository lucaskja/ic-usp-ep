"""
Tests for model utility functions.
"""
import sys
import os
import unittest
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import get_model_size, count_parameters


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x


class TestModelUtils(unittest.TestCase):
    """Test cases for model utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
    
    def test_count_parameters(self):
        """Test parameter counting function."""
        params = count_parameters(self.model)
        self.assertGreater(params, 0)
        
        # Calculate parameters manually
        manual_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(params, manual_count)
    
    def test_get_model_size(self):
        """Test model size calculation function."""
        size_mb = get_model_size(self.model)
        self.assertGreater(size_mb, 0)
        
        # Rough calculation to verify
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        expected_size = (param_size + buffer_size) / 1024**2
        
        # Allow for small differences due to floating point precision
        self.assertAlmostEqual(size_mb, expected_size, places=5)


if __name__ == '__main__':
    unittest.main()

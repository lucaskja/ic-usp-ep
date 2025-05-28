"""
Tests for MemTorch-based CNN model.
"""

import unittest
import torch
import memtorch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memtorch_cnn.models import MemTorchCNN


class TestMemTorchCNN(unittest.TestCase):
    """Test cases for MemTorch-based CNN model."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 10
        self.batch_size = 2
        self.input_shape = (self.batch_size, 3, 224, 224)
        
        # Create model
        self.model = MemTorchCNN(num_classes=self.num_classes)
        self.model = self.model.to(self.device)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.model, MemTorchCNN)
        self.assertEqual(self.model.num_classes, self.num_classes)
        self.assertFalse(self.model.is_memristive)
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Create random input
        x = torch.randn(self.input_shape).to(self.device)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_memristive_conversion(self):
        """Test conversion to memristive model."""
        # Skip test if memtorch is not available
        try:
            import memtorch
        except ImportError:
            self.skipTest("memtorch not available")
        
        # Convert model to memristive
        self.model.convert_to_memristive(device=self.device)
        
        # Check if model is memristive
        self.assertTrue(self.model.is_memristive)
        
        # Check if layers are converted
        memristive_layers = 0
        for module in self.model.modules():
            if isinstance(module, (memtorch.mn.Conv2d, memtorch.mn.Linear)):
                memristive_layers += 1
        
        # Should have at least one memristive layer
        self.assertGreater(memristive_layers, 0)
    
    def test_memristive_forward_pass(self):
        """Test forward pass with memristive model."""
        # Skip test if memtorch is not available
        try:
            import memtorch
        except ImportError:
            self.skipTest("memtorch not available")
        
        # Convert model to memristive
        self.model.convert_to_memristive(device=self.device)
        
        # Create random input
        x = torch.randn(self.input_shape).to(self.device)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_weight_quantization(self):
        """Test weight quantization."""
        # Skip test if memtorch is not available
        try:
            import memtorch
        except ImportError:
            self.skipTest("memtorch not available")
        
        # Convert model to memristive
        self.model.convert_to_memristive(device=self.device)
        
        # Apply weight quantization
        self.model.apply_weight_quantization(bits=4)
        
        # Check if weights are quantized
        for name, module in self.model.named_modules():
            if isinstance(module, (memtorch.mn.Conv2d, memtorch.mn.Linear)):
                # Get unique weight values
                unique_values = torch.unique(module.weight)
                
                # 4-bit quantization should have at most 2^4 = 16 unique values
                self.assertLessEqual(len(unique_values), 16)
    
    def test_non_idealities(self):
        """Test application of non-idealities."""
        # Skip test if memtorch is not available
        try:
            import memtorch
        except ImportError:
            self.skipTest("memtorch not available")
        
        # Convert model to memristive
        self.model.convert_to_memristive(device=self.device)
        
        # Apply non-idealities
        self.model.apply_non_idealities()
        
        # No assertion needed, just checking if it runs without errors


if __name__ == '__main__':
    unittest.main()

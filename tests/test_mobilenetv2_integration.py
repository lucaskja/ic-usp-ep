"""
Tests for MobileNetV2 integration with Triplet Attention and CNSN.
"""
import torch
import unittest
from stage2_triplet.models.mobilenetv2_triplet_fixed import create_mobilenetv2_triplet
from stage3_cnsn.models.mobilenetv2_cnsn_fixed import create_mobilenetv2_cnsn


class TestMobileNetV2Integration(unittest.TestCase):
    """
    Test cases for MobileNetV2 integration with Triplet Attention and CNSN.
    """
    def test_mobilenetv2_triplet(self):
        """
        Test MobileNetV2 with Triplet Attention.
        """
        # Create model
        model = create_mobilenetv2_triplet(num_classes=10, pretrained=False)
        
        # Set to eval mode to speed up testing
        model.eval()
        
        # Create a random input tensor
        x = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10))
        
    def test_mobilenetv2_cnsn(self):
        """
        Test MobileNetV2 with Triplet Attention and CNSN.
        """
        # Create model
        model = create_mobilenetv2_cnsn(num_classes=10, pretrained=False)
        
        # Set to eval mode to speed up testing
        model.eval()
        
        # Create a random input tensor
        x = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10))
        
    def test_training_mode(self):
        """
        Test models in training mode.
        """
        # Create models
        triplet_model = create_mobilenetv2_triplet(num_classes=10, pretrained=False)
        cnsn_model = create_mobilenetv2_cnsn(num_classes=10, pretrained=False)
        
        # Set to training mode
        triplet_model.train()
        cnsn_model.train()
        
        # Create a random input tensor
        x = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            triplet_output = triplet_model(x)
            cnsn_output = cnsn_model(x)
        
        # Check output shapes
        self.assertEqual(triplet_output.shape, (2, 10))
        self.assertEqual(cnsn_output.shape, (2, 10))


if __name__ == '__main__':
    unittest.main()

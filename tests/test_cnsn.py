"""
Tests for CNSN module.
"""
import torch
import unittest
from stage3_cnsn.models.cnsn_fixed import CrossNorm, SelfNorm, CNSN


class TestCNSN(unittest.TestCase):
    """
    Test cases for CNSN module.
    """
    def test_crossnorm(self):
        """
        Test CrossNorm module.
        """
        # Create a random input tensor
        x = torch.randn(4, 16, 24, 24)
        
        # Create CrossNorm module
        crossnorm = CrossNorm(p=1.0)  # Always apply during training
        
        # Set to training mode
        crossnorm.train()
        
        # Forward pass
        output = crossnorm(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Set to evaluation mode
        crossnorm.eval()
        
        # Forward pass in eval mode
        output_eval = crossnorm(x)
        
        # In eval mode, CrossNorm should be identity
        self.assertTrue(torch.allclose(output_eval, x))
        
    def test_selfnorm(self):
        """
        Test SelfNorm module.
        """
        # Create a random input tensor
        x = torch.randn(4, 16, 24, 24)
        
        # Create SelfNorm module
        selfnorm = SelfNorm(channels=16)
        
        # Forward pass
        output = selfnorm(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
    def test_cnsn(self):
        """
        Test CNSN module.
        """
        # Create a random input tensor
        x = torch.randn(4, 16, 24, 24)
        
        # Create CNSN module
        cnsn = CNSN(channels=16, p=1.0)  # Always apply CrossNorm during training
        
        # Set to training mode
        cnsn.train()
        
        # Forward pass
        output = cnsn(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Set to evaluation mode
        cnsn.eval()
        
        # Forward pass in eval mode
        output_eval = cnsn(x)
        
        # Check output shape
        self.assertEqual(output_eval.shape, x.shape)
        
    def test_cnsn_different_sizes(self):
        """
        Test CNSN module with different input sizes.
        """
        # Test with different input sizes
        for c in [3, 16, 64]:
            for h, w in [(32, 32), (24, 48), (64, 32)]:
                # Create a random input tensor
                x = torch.randn(4, c, h, w)
                
                # Create CNSN module
                cnsn = CNSN(channels=c, p=1.0)
                
                # Forward pass
                output = cnsn(x)
                
                # Check output shape (should be same as input)
                self.assertEqual(output.shape, x.shape)


if __name__ == '__main__':
    unittest.main()

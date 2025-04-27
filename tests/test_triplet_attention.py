"""
Tests for Triplet Attention module.
"""
import torch
import unittest
from stage2_triplet.models.triplet_attention_fixed import Z_Pool, TripletAttention


class TestTripletAttention(unittest.TestCase):
    """
    Test cases for Triplet Attention module.
    """
    def test_z_pool(self):
        """
        Test Z_Pool module.
        """
        # Create a random input tensor
        x = torch.randn(2, 16, 24, 24)
        
        # Create Z_Pool module
        z_pool = Z_Pool()
        
        # Forward pass
        output = z_pool(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 2, 24, 24))
        
    def test_triplet_attention(self):
        """
        Test TripletAttention module.
        """
        # Create a random input tensor
        x = torch.randn(2, 16, 24, 24)
        
        # Create TripletAttention module
        triplet_attention = TripletAttention(kernel_size=7)
        
        # Forward pass
        output = triplet_attention(x)
        
        # Check output shape (should be same as input)
        self.assertEqual(output.shape, x.shape)
        
    def test_triplet_attention_different_sizes(self):
        """
        Test TripletAttention module with different input sizes.
        """
        # Test with different input sizes
        for c in [3, 16, 64]:
            for h, w in [(32, 32), (24, 48), (64, 32)]:
                # Create a random input tensor
                x = torch.randn(2, c, h, w)
                
                # Create TripletAttention module
                triplet_attention = TripletAttention(kernel_size=7)
                
                # Forward pass
                output = triplet_attention(x)
                
                # Check output shape (should be same as input)
                self.assertEqual(output.shape, x.shape)


if __name__ == '__main__':
    unittest.main()

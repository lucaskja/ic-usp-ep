"""
Tests for training utility functions.
"""
import sys
import os
import unittest
import torch
import torch.nn as nn
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_utils import accuracy, save_checkpoint, EarlyStopping


class TestTrainingUtils(unittest.TestCase):
    """Test cases for training utility functions."""
    
    def test_accuracy(self):
        """Test accuracy calculation function."""
        # Create sample outputs and targets
        outputs = torch.tensor([
            [0.1, 0.2, 0.7],  # Class 2 has highest probability
            [0.8, 0.1, 0.1],  # Class 0 has highest probability
            [0.1, 0.8, 0.1],  # Class 1 has highest probability
            [0.3, 0.3, 0.4]   # Class 2 has highest probability
        ])
        
        targets = torch.tensor([2, 0, 0, 2])  # Correct for indices 0, 1, 3
        
        # Test top-1 accuracy
        acc1 = accuracy(outputs, targets, topk=(1,))
        self.assertEqual(acc1[0].item(), 75.0)  # 3 out of 4 correct = 75%
        
        # Test top-2 accuracy
        acc2 = accuracy(outputs, targets, topk=(2,))
        self.assertEqual(acc2[0].item(), 75.0)  # Still 3 out of 4 correct within top 2
        
        # Test both top-1 and top-2
        acc1, acc2 = accuracy(outputs, targets, topk=(1, 2))
        self.assertEqual(acc1.item(), 75.0)
        self.assertEqual(acc2.item(), 75.0)
        
        # Test with k > number of classes
        acc_large = accuracy(outputs, targets, topk=(5,))
        self.assertEqual(acc_large[0].item(), 100.0)  # All correct when k > num_classes
    
    def test_save_checkpoint(self):
        """Test checkpoint saving function."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create dummy state
            state = {
                'epoch': 10,
                'state_dict': {'weight': torch.tensor([1.0, 2.0, 3.0])},
                'optimizer': {'lr': 0.01}
            }
            
            # Save checkpoint
            save_checkpoint(state, False, temp_dir)
            
            # Check if file exists
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'checkpoint.pth')))
            
            # Save as best
            save_checkpoint(state, True, temp_dir)
            
            # Check if best file exists
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'model_best.pth')))
            
            # Load and verify
            loaded = torch.load(os.path.join(temp_dir, 'checkpoint.pth'))
            self.assertEqual(loaded['epoch'], 10)
            self.assertTrue(torch.all(loaded['state_dict']['weight'] == torch.tensor([1.0, 2.0, 3.0])))
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        # Create early stopping with patience 3
        early_stopping = EarlyStopping(patience=3, verbose=False)
        
        # Test improvement
        self.assertFalse(early_stopping(50.0))  # First call, should not stop
        self.assertFalse(early_stopping(55.0))  # Improvement, should not stop
        
        # Test no improvement
        self.assertFalse(early_stopping(54.0))  # No improvement, counter = 1
        self.assertFalse(early_stopping(54.0))  # No improvement, counter = 2
        self.assertFalse(early_stopping(54.0))  # No improvement, counter = 3
        self.assertTrue(early_stopping(54.0))   # No improvement, counter = 4, should stop
        
        # Verify early stop flag
        self.assertTrue(early_stopping.early_stop)
        
        # Test reset behavior with new instance
        early_stopping = EarlyStopping(patience=2, verbose=False)
        self.assertFalse(early_stopping(50.0))
        self.assertFalse(early_stopping(49.0))  # No improvement, counter = 1
        self.assertFalse(early_stopping(51.0))  # Improvement, counter reset to 0
        self.assertFalse(early_stopping(50.0))  # No improvement, counter = 1
        self.assertFalse(early_stopping(49.0))  # No improvement, counter = 2
        self.assertTrue(early_stopping(48.0))   # No improvement, counter = 3, should stop


if __name__ == '__main__':
    unittest.main()

"""
Tests for HybridTrainer.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memtorch_cnn.models import MemTorchCNN
from memtorch_cnn.utils.trainer import HybridTrainer


class TestHybridTrainer(unittest.TestCase):
    """Test cases for HybridTrainer."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 10
        self.batch_size = 2
        self.input_shape = (self.batch_size, 3, 224, 224)
        
        # Create model
        self.model = MemTorchCNN(num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        
        # Create trainer
        self.trainer = HybridTrainer(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            checkpoint_dir=self.temp_dir
        )
        
        # Create dummy data
        self.inputs = torch.randn(self.input_shape).to(self.device)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        
        # Create dummy data loader
        self.dummy_loader = [(self.inputs, self.targets)]
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        self.assertIsInstance(self.trainer, HybridTrainer)
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.optimizer, self.optimizer)
        self.assertEqual(self.trainer.device, self.device)
        self.assertEqual(self.trainer.checkpoint_dir, self.temp_dir)
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        # Train for one epoch
        loss, acc = self.trainer._train_epoch(self.dummy_loader)
        
        # Check if loss and accuracy are computed
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
    
    def test_validate(self):
        """Test validation."""
        # Validate
        loss, acc = self.trainer._validate(self.dummy_loader)
        
        # Check if loss and accuracy are computed
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        # Save checkpoint
        self.trainer._save_checkpoint(epoch=0, is_best=True)
        
        # Check if checkpoint files are created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'checkpoint_epoch_0.pth')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'model_best.pth')))
    
    def test_ex_situ_train(self):
        """Test ex-situ training."""
        # Skip full training, just test if it runs
        try:
            history = self.trainer.ex_situ_train(
                train_loader=self.dummy_loader,
                val_loader=self.dummy_loader,
                epochs=1
            )
            
            # Check if history is returned
            self.assertIsInstance(history, dict)
            self.assertIn('train_loss', history)
            self.assertIn('train_acc', history)
            self.assertIn('val_loss', history)
            self.assertIn('val_acc', history)
        except Exception as e:
            self.fail(f"ex_situ_train raised {type(e).__name__} unexpectedly: {e}")
    
    def test_in_situ_train(self):
        """Test in-situ training."""
        # Skip if memtorch is not available
        try:
            import memtorch
        except ImportError:
            self.skipTest("memtorch not available")
        
        # Convert model to memristive
        self.model.convert_to_memristive(device=self.device)
        
        # Skip full training, just test if it runs
        try:
            history = self.trainer.in_situ_train(
                train_loader=self.dummy_loader,
                val_loader=self.dummy_loader,
                epochs=1
            )
            
            # Check if history is returned
            self.assertIsInstance(history, dict)
            self.assertIn('train_loss', history)
            self.assertIn('train_acc', history)
            self.assertIn('val_loss', history)
            self.assertIn('val_acc', history)
        except Exception as e:
            self.fail(f"in_situ_train raised {type(e).__name__} unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()

"""
Tests for training utilities.
"""
import os
import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.training_utils import (
    AverageMeter,
    accuracy,
    train_one_epoch,
    validate,
    save_checkpoint
)


def test_average_meter():
    """Test AverageMeter class."""
    meter = AverageMeter('test')
    
    # Test single update
    meter.update(1.0)
    assert meter.val == 1.0
    assert meter.avg == 1.0
    assert meter.sum == 1.0
    assert meter.count == 1
    
    # Test multiple updates
    meter.update(2.0)
    assert meter.val == 2.0
    assert meter.avg == 1.5
    assert meter.sum == 3.0
    assert meter.count == 2
    
    # Test reset
    meter.reset()
    assert meter.val == 0
    assert meter.avg == 0
    assert meter.sum == 0
    assert meter.count == 0


def test_accuracy():
    """Test accuracy calculation."""
    output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # 2 samples, 2 classes
    target = torch.tensor([1, 0])  # Correct classes
    
    acc1, = accuracy(output, target, topk=(1,))
    assert acc1.item() == 100.0  # Both predictions are correct


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


def test_train_one_epoch():
    """Test train_one_epoch function."""
    # Create a simple model and data
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy data
    x = torch.randn(4, 10)  # 4 samples, 10 features
    y = torch.tensor([0, 1, 0, 1])  # Binary classification
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    
    # Train for one epoch
    metrics = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device='cpu',
        epoch=0
    )
    
    # Check that metrics are returned
    assert 'loss' in metrics
    assert 'acc1' in metrics
    assert 'acc5' in metrics


def test_validate():
    """Test validate function."""
    # Create a simple model and data
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy data
    x = torch.randn(4, 10)  # 4 samples, 10 features
    y = torch.tensor([0, 1, 0, 1])  # Binary classification
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    
    # Validate
    metrics = validate(
        model,
        loader,
        criterion,
        device='cpu'
    )
    
    # Check that metrics are returned
    assert 'loss' in metrics
    assert 'acc1' in metrics
    assert 'acc5' in metrics


def test_save_checkpoint(tmp_path):
    """Test save_checkpoint function."""
    # Create a simple model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create checkpoint
    checkpoint = {
        'epoch': 1,
        'state_dict': model.state_dict(),
        'best_acc1': 90.0,
        'optimizer': optimizer.state_dict(),
    }
    
    # Save checkpoint
    checkpoint_dir = str(tmp_path)
    save_checkpoint(checkpoint, True, checkpoint_dir)
    
    # Check that files are created
    assert os.path.exists(os.path.join(checkpoint_dir, 'checkpoint.pth'))
    assert os.path.exists(os.path.join(checkpoint_dir, 'model_best.pth'))
    
    # Load checkpoint and verify contents
    loaded = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth'))
    assert loaded['epoch'] == 1
    assert loaded['best_acc1'] == 90.0
    assert 'state_dict' in loaded
    assert 'optimizer' in loaded

"""
Tests for data utilities.
"""
import os
import torch
import pytest
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from utils.data_utils import get_transforms, load_dataset


def test_get_transforms():
    """Test get_transforms function."""
    train_transforms, val_transforms = get_transforms(img_size=224)
    
    # Check that transforms are created
    assert train_transforms is not None
    assert val_transforms is not None
    
    # Check that transforms are of correct type
    assert isinstance(train_transforms, transforms.Compose)
    assert isinstance(val_transforms, transforms.Compose)
    
    # Check that train transforms include data augmentation
    train_transform_list = train_transforms.transforms
    has_random_transform = any(isinstance(t, (transforms.RandomHorizontalFlip, 
                                             transforms.RandomRotation, 
                                             transforms.ColorJitter)) 
                              for t in train_transform_list)
    assert has_random_transform
    
    # Check that both transforms include normalization
    assert any(isinstance(t, transforms.Normalize) for t in train_transforms.transforms)
    assert any(isinstance(t, transforms.Normalize) for t in val_transforms.transforms)


@pytest.mark.skipif(not os.path.exists('datasets/sample'), 
                   reason="Sample dataset not available")
def test_load_dataset():
    """Test load_dataset function with a sample dataset."""
    # This test requires a sample dataset to be available
    # Skip if not available
    
    train_loader, val_loader, num_classes = load_dataset(
        'datasets/sample',
        img_size=224,
        batch_size=4,
        val_split=0.2,
        num_workers=0  # Use 0 workers for testing
    )
    
    # Check that loaders are created
    assert train_loader is not None
    assert val_loader is not None
    
    # Check that loaders are of correct type
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # Check that num_classes is positive
    assert num_classes > 0
    
    # Check that batch size is correct
    for images, labels in train_loader:
        assert images.shape[0] <= 4  # Less than or equal because last batch might be smaller
        break

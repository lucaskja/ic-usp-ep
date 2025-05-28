"""
Data utilities for MemTorch-based CNN.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

def get_leaf_disease_dataloaders(data_dir, batch_size=32, enhanced_augmentation=False, test_split=0.1, val_split=0.2):
    """
    Get data loaders for leaf disease dataset.
    
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for data loaders.
        enhanced_augmentation (bool): Whether to use enhanced data augmentation.
        test_split (float): Fraction of data to use for testing.
        val_split (float): Fraction of remaining data to use for validation.
        
    Returns:
        tuple: Train, validation, and test data loaders.
    """
    # Define transforms
    if enhanced_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check if dataset is already split
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Dataset is already split
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)
        
        # Split train into train and validation
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update transform for validation dataset
        val_dataset.dataset.transform = val_transform
    else:
        # Load the entire dataset
        full_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
        
        # Split into train, validation, and test
        test_size = int(test_split * len(full_dataset))
        train_val_size = len(full_dataset) - test_size
        train_size = int((1 - val_split) * train_val_size)
        val_size = train_val_size - train_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update transforms
        train_dataset.dataset.transform = train_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_names(data_dir):
    """
    Get class names from the dataset.
    
    Args:
        data_dir (str): Path to the dataset directory.
        
    Returns:
        list: List of class names.
    """
    # Check if dataset is already split
    train_dir = os.path.join(data_dir, 'train')
    
    if os.path.exists(train_dir):
        # Dataset is already split
        dataset = datasets.ImageFolder(root=train_dir)
    else:
        # Load the entire dataset
        dataset = datasets.ImageFolder(root=data_dir)
    
    return dataset.classes

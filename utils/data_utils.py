"""
Utility functions for data loading and preprocessing.
"""
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


def get_transforms(img_size=224):
    """
    Get standard data transformations for training and validation.
    
    Args:
        img_size (int): Size to resize images to
        
    Returns:
        tuple: (train_transforms, val_transforms)
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def load_dataset(data_dir, img_size=224, batch_size=32, val_split=0.2, num_workers=4):
    """
    Load and prepare datasets for training and validation.
    
    Args:
        data_dir (str): Path to dataset directory
        img_size (int): Size to resize images to
        batch_size (int): Batch size for dataloaders
        val_split (float): Proportion of data to use for validation
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    train_transforms, val_transforms = get_transforms(img_size)
    
    # Load the full dataset with training transforms
    full_dataset = ImageFolder(root=data_dir, transform=train_transforms)
    
    # Calculate sizes for train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation set with validation transforms
    val_dataset.dataset = ImageFolder(root=data_dir, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(full_dataset.classes)
    
    return train_loader, val_loader, num_classes

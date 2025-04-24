"""
Utility functions for data loading and preprocessing.
"""
import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import logging
from utils.deprecated import deprecated


@deprecated(reason="Use load_dataset from the unified training script instead", 
          alternative="utils.enhanced_data_utils.load_enhanced_dataset")
def get_transforms(img_size=224):
    """
    Get data transforms for training and validation.
    
    Args:
        img_size (int): Size of the input image (default: 224)
        
    Returns:
        tuple: (train_transforms, val_transforms)
    """
    # Define normalization parameters (ImageNet mean and std)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # 256 / 224 = 1.14
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transforms, val_transforms


@deprecated(reason="Use load_dataset from the unified training script instead", 
          alternative="utils.enhanced_data_utils.load_enhanced_dataset")
def load_dataset(data_dir, img_size=224, batch_size=32, val_split=0.2, num_workers=4, debug=False):
    """
    Load dataset from directory.
    
    Args:
        data_dir (str): Path to dataset directory
        img_size (int): Size of the input image (default: 224)
        batch_size (int): Batch size (default: 32)
        val_split (float): Validation split ratio (default: 0.2)
        num_workers (int): Number of workers for data loading (default: 4)
        debug (bool): If True, use a small subset of data for debugging
        
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    # Get transforms
    train_transforms, val_transforms = get_transforms(img_size)
    
    # Check if dataset has predefined train/val split
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # Use predefined split
        train_dataset = ImageFolder(train_dir, transform=train_transforms)
        val_dataset = ImageFolder(val_dir, transform=val_transforms)
    else:
        # Create split from single directory
        full_dataset = ImageFolder(data_dir, transform=train_transforms)
        
        # Calculate split sizes
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Apply different transforms to validation set
        val_dataset.dataset = ImageFolder(data_dir, transform=val_transforms)
    
    # If in debug mode, use only a small subset of the data
    if debug:
        logging.info("Running in debug mode with reduced dataset size")
        train_indices = list(range(min(100, len(train_dataset))))
        val_indices = list(range(min(50, len(val_dataset))))
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        logging.info(f"Debug dataset sizes: {len(train_dataset)} training, {len(val_dataset)} validation")
    
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
    
    # Get number of classes
    # For leaf disease dataset, we know there are 3 classes: Bacterial Blight, Brown Spot, and Leaf Smut
    if 'leaf_disease' in data_dir:
        num_classes = 3
        logging.info("Leaf disease dataset detected, using 3 classes")
    elif hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'classes'):
        num_classes = len(train_dataset.dataset.classes)
    else:
        # If classes attribute is not available, infer from the data
        if hasattr(train_dataset, 'targets'):
            num_classes = len(set(train_dataset.targets))
        elif hasattr(train_dataset.dataset, 'targets'):
            num_classes = len(set(train_dataset.dataset.targets))
        else:
            # Default to 1000 (ImageNet) if we can't determine
            num_classes = 1000
            logging.warning("Could not determine number of classes, defaulting to 1000")
    
    # Verify that we have the correct number of classes for leaf disease dataset
    if 'leaf_disease' in data_dir and num_classes != 3:
        logging.warning(f"Expected 3 classes for leaf disease dataset, but found {num_classes}. Overriding to 3 classes.")
        num_classes = 3
    
    logging.info(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    logging.info(f"Number of classes: {num_classes}")
    
    return train_loader, val_loader, num_classes

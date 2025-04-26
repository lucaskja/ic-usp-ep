"""
Utility functions for data loading and preprocessing.
This module handles splitting data into test, train, and validation sets.
"""
import os
import shutil
import random
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def get_transforms(img_size=224):
    """
    Get data transforms for training and validation.
    
    Args:
        img_size (int): Size of the input image (default: 224)
        
    Returns:
        tuple: (train_transforms, val_transforms, test_transforms)
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
    
    # Validation and test transforms (no augmentation)
    eval_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # 256 / 224 = 1.14
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transforms, eval_transforms, eval_transforms


def split_dataset(data_dir, output_dir=None, test_ratio=0.1, val_ratio=0.2, seed=42):
    """
    Split dataset into test, train, and validation sets and save to separate directories.
    
    Args:
        data_dir (str): Path to the source dataset directory
        output_dir (str, optional): Path to save the split dataset. If None, uses data_dir
        test_ratio (float): Ratio of data to use for testing (default: 0.1)
        val_ratio (float): Ratio of remaining data (after test split) to use for validation (default: 0.2)
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dir, val_dir, test_dir) - Paths to the created directories
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # If output_dir is not specified, use data_dir
    if output_dir is None:
        output_dir = data_dir
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    # Check if directories already exist
    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        logging.info(f"Split directories already exist at {output_dir}")
        return train_dir, val_dir, test_dir
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Process each class
    for class_name in class_dirs:
        # Create class directories in train, val, and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all images for this class
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        # Skip if no images found
        if not images:
            logging.warning(f"No images found in {class_path}, skipping")
            continue
            
        # First split: separate test set
        train_val_images, test_images = train_test_split(
            images, 
            test_size=test_ratio, 
            random_state=seed
        )
        
        # Second split: separate train and validation sets from remaining data
        train_images, val_images = train_test_split(
            train_val_images,
            test_size=val_ratio,
            random_state=seed
        )
        
        # Copy images to their respective directories
        for img in train_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(train_dir, class_name, img)
            )
        
        for img in val_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(val_dir, class_name, img)
            )
        
        for img in test_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(test_dir, class_name, img)
            )
        
        logging.info(f"Class {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    logging.info(f"Dataset split complete: {output_dir}")
    return train_dir, val_dir, test_dir


def load_dataset(data_dir, img_size=224, batch_size=32, num_workers=4, debug=False):
    """
    Load dataset from directory, splitting into test, train, and validation if needed.
    
    Args:
        data_dir (str): Path to dataset directory
        img_size (int): Size of the input image (default: 224)
        batch_size (int): Batch size (default: 32)
        num_workers (int): Number of workers for data loading (default: 4)
        debug (bool): If True, use a small subset of data for debugging
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Get transforms
    train_transforms, val_transforms, test_transforms = get_transforms(img_size)
    
    # Check if dataset is already split
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    if not (os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir)):
        # Split the dataset
        logging.info(f"Dataset not split. Splitting into train, val, and test sets...")
        train_dir, val_dir, test_dir = split_dataset(data_dir)
    
    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)
    test_dataset = ImageFolder(test_dir, transform=test_transforms)
    
    # Ensure class indices are consistent across all datasets
    val_dataset.class_to_idx = train_dataset.class_to_idx
    test_dataset.class_to_idx = train_dataset.class_to_idx
    
    # If in debug mode, use only a small subset of the data
    if debug:
        logging.info("Running in debug mode with reduced dataset size")
        train_indices = list(range(min(100, len(train_dataset))))
        val_indices = list(range(min(50, len(val_dataset))))
        test_indices = list(range(min(50, len(test_dataset))))
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)
        
        logging.info(f"Debug dataset sizes: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test")
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get number of classes
    num_classes = len(train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes)
    
    logging.info(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")
    logging.info(f"Number of classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


def get_class_names(data_dir):
    """
    Get class names from the dataset.
    
    Args:
        data_dir (str): Path to dataset directory
        
    Returns:
        list: List of class names
    """
    # Check if dataset is already split
    train_dir = os.path.join(data_dir, 'train')
    
    if os.path.exists(train_dir):
        # Use train directory to get class names
        dataset = ImageFolder(train_dir)
    else:
        # Use main directory
        dataset = ImageFolder(data_dir)
    
    return dataset.classes

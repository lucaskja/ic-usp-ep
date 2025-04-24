"""
Enhanced data utilities with stronger augmentation techniques.
"""
import os
import logging
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def get_enhanced_transforms(img_size=224):
    """
    Get enhanced data transforms with stronger augmentation.
    
    Args:
        img_size (int): Size of the input image (default: 224)
        
    Returns:
        tuple: (train_transforms, val_transforms)
    """
    # Enhanced training transforms with stronger augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    
    # Validation transforms (same as before)
    val_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def load_enhanced_dataset(data_dir, img_size=224, batch_size=32, val_split=0.2, num_workers=4, debug=False):
    """
    Load dataset with enhanced data augmentation.
    
    Args:
        data_dir (str): Path to the dataset directory
        img_size (int): Size of the input image (default: 224)
        batch_size (int): Batch size (default: 32)
        val_split (float): Validation split ratio (default: 0.2)
        num_workers (int): Number of workers for data loading (default: 4)
        debug (bool): Whether to use a small subset of the dataset for debugging (default: False)
        
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    # Get transforms
    train_transforms, val_transforms = get_enhanced_transforms(img_size)
    
    # Check if the dataset is already split into train and test
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Dataset is already split
        train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
        val_dataset = ImageFolder(root=test_dir, transform=val_transforms)
        
        # Ensure class indices are consistent between train and val
        val_dataset.class_to_idx = train_dataset.class_to_idx
        
        logging.info(f"Dataset already split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    else:
        # Dataset needs to be split
        dataset = ImageFolder(root=data_dir, transform=train_transforms)
        
        # Calculate split sizes
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply different transforms to validation set
        val_dataset.dataset.transform = val_transforms
        
        logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Use a small subset for debugging if needed
    if debug:
        debug_train_size = min(100, len(train_dataset))
        debug_val_size = min(50, len(val_dataset))
        
        logging.info(f"Running in debug mode with reduced dataset size")
        logging.info(f"Debug dataset sizes: {debug_train_size} training, {debug_val_size} validation")
        
        # Create subset indices
        train_indices = torch.randperm(len(train_dataset))[:debug_train_size]
        val_indices = torch.randperm(len(val_dataset))[:debug_val_size]
        
        # Create subsets
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
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
    elif hasattr(val_dataset, 'classes'):
        num_classes = len(val_dataset.classes)
    elif hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'classes'):
        num_classes = len(val_dataset.dataset.classes)
    else:
        try:
            # Try to infer from the output size of the first batch
            for images, _ in train_loader:
                break
            num_classes = _.max().item() + 1
            logging.info(f"Inferred number of classes from data: {num_classes}")
        except:
            num_classes = 1000
            logging.warning("Could not determine number of classes, defaulting to 1000")
    
    # Verify that we have the correct number of classes for leaf disease dataset
    if 'leaf_disease' in data_dir and num_classes != 3:
        logging.warning(f"Expected 3 classes for leaf disease dataset, but found {num_classes}. Overriding to 3 classes.")
        num_classes = 3
    
    logging.info(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    logging.info(f"Number of classes: {num_classes}")
    
    return train_loader, val_loader, num_classes

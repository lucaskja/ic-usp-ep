"""
Utility script to visualize data transformations and augmentations.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import get_transforms

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize data transformations')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='experiments/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--num_augmentations', type=int, default=8,
                        help='Number of augmented versions to generate')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for transformations')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced augmentations')
    return parser.parse_args()

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

def denormalize(tensor):
    """
    Denormalize an image tensor that was normalized with ImageNet mean and std.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
        
    Returns:
        torch.Tensor: Denormalized image tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_transforms(image_path, output_dir, num_augmentations=8, img_size=224, enhanced=False):
    """
    Visualize original and transformed images.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save visualizations
        num_augmentations (int): Number of augmented versions to generate
        img_size (int): Image size for transformations
        enhanced (bool): Whether to use enhanced augmentations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Get transforms
    if enhanced:
        train_transforms, val_transforms = get_enhanced_transforms(img_size)
        prefix = "enhanced"
    else:
        train_transforms, val_transforms = get_transforms(img_size)
        prefix = "standard"
    
    # Apply validation transform
    val_tensor = val_transforms(image)
    
    # Apply training transform multiple times
    train_tensors = [train_transforms(image) for _ in range(num_augmentations)]
    
    # Create a grid with the original and transformed images
    all_tensors = [val_tensor] + train_tensors
    grid = make_grid(all_tensors, nrow=4, padding=10)
    
    # Denormalize for visualization
    grid_denorm = denormalize(grid)
    
    # Convert to numpy for matplotlib
    grid_np = grid_denorm.permute(1, 2, 0).numpy()
    grid_np = np.clip(grid_np, 0, 1)
    
    # Plot
    plt.figure(figsize=(15, 10))
    plt.imshow(grid_np)
    plt.title(f"Image Transformations ({prefix})")
    plt.axis('off')
    
    # Add labels
    plt.text(img_size//2, 10, "Original (Val Transform)", 
             horizontalalignment='center', color='white', fontsize=12)
    
    for i in range(num_augmentations):
        row = (i + 1) // 4
        col = (i + 1) % 4
        x_pos = col * (img_size + 10) + img_size//2
        y_pos = row * (img_size + 10) + 10
        plt.text(x_pos, y_pos, f"Aug #{i+1}", 
                 horizontalalignment='center', color='white', fontsize=12)
    
    # Save the visualization
    output_path = os.path.join(output_dir, f"{prefix}_transforms_{os.path.basename(image_path)}")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Save individual augmented images
    for i, tensor in enumerate(all_tensors):
        img_denorm = denormalize(tensor)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.axis('off')
        
        label = "original_val" if i == 0 else f"aug_{i}"
        img_output_path = os.path.join(output_dir, f"{prefix}_{label}_{os.path.basename(image_path)}")
        plt.savefig(img_output_path, bbox_inches='tight')
        plt.close()

def main():
    """Main function."""
    args = parse_args()
    visualize_transforms(
        args.image_path, 
        args.output_dir, 
        args.num_augmentations, 
        args.img_size,
        args.enhanced
    )

if __name__ == "__main__":
    main()

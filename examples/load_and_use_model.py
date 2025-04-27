"""
Example script demonstrating how to load and use a trained model.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model loading utilities
from utils.model_loader import load_best_model, load_model_for_inference


def load_image(image_path, transform=None):
    """
    Load and preprocess an image for model inference.
    
    Args:
        image_path (str): Path to the image file
        transform (callable, optional): Transformation to apply to the image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def main():
    """Main function demonstrating model loading and inference."""
    # Parameters
    model_type = 'cnsn'  # Change to 'base', 'mish', or 'triplet' as needed
    num_classes = 4  # Change to match your model
    checkpoint_dir = os.path.join('checkpoints', model_type)
    image_path = 'datasets/rld/test/class1/img001.jpg'  # Change to your test image
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Method 1: Load the best model directly
    print("\nMethod 1: Loading best model...")
    model = load_best_model(model_type, num_classes, checkpoint_dir, device)
    
    # Method 2: Create an inference function
    print("\nMethod 2: Creating inference function...")
    checkpoint_path = os.path.join(checkpoint_dir, 'best.pth')
    inference_fn = load_model_for_inference(model_type, checkpoint_path, num_classes, device)
    
    # Load a test image
    print(f"\nLoading test image: {image_path}")
    try:
        image_tensor = load_image(image_path)
        image_tensor = image_tensor.to(device)
        
        # Method 1: Direct inference with model
        print("\nRunning inference with Method 1...")
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        print(f"Prediction: Class {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
        # Method 2: Using the inference function
        print("\nRunning inference with Method 2...")
        pred, prob = inference_fn(image_tensor)
        prediction2 = pred.item()
        confidence2 = prob[0, prediction2].item()
        
        print(f"Prediction: Class {prediction2}")
        print(f"Confidence: {confidence2:.4f}")
        
        # Print all class probabilities
        print("\nClass probabilities:")
        for i in range(num_classes):
            print(f"Class {i}: {prob[0, i].item():.4f}")
        
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()

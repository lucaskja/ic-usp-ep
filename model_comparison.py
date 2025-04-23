"""
Compare parameter sizes between PyTorch and TensorFlow MobileNetV2 implementations.
"""
import os
import sys
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Calculate the size of a PyTorch model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def main():
    # PyTorch MobileNetV2
    print("Loading PyTorch MobileNetV2...")
    pytorch_model = mobilenet_v2(weights=None)  # No pre-trained weights
    
    # Count parameters and calculate size
    pytorch_params = count_parameters(pytorch_model)
    pytorch_size_mb = get_model_size_mb(pytorch_model)
    
    print("\nPyTorch MobileNetV2:")
    print(f"Number of parameters: {pytorch_params:,}")
    print(f"Model size: {pytorch_size_mb:.2f} MB")
    
    # Try to import TensorFlow if available
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2 as TFMobileNetV2
        
        # TensorFlow MobileNetV2
        print("\nLoading TensorFlow MobileNetV2...")
        tf_model = TFMobileNetV2(weights=None, include_top=True)  # No pre-trained weights
        
        # Count parameters
        tf_params = tf_model.count_params()
        
        # Calculate size (approximate)
        tf_size_mb = tf_params * 4 / 1024**2  # Assuming float32 (4 bytes)
        
        print("\nTensorFlow MobileNetV2:")
        print(f"Number of parameters: {tf_params:,}")
        print(f"Model size (approx): {tf_size_mb:.2f} MB")
        
        # Compare
        print("\nComparison:")
        print(f"Parameter difference: {abs(pytorch_params - tf_params):,} ({abs(pytorch_params - tf_params) / max(pytorch_params, tf_params) * 100:.2f}%)")
        print(f"Size difference: {abs(pytorch_size_mb - tf_size_mb):.2f} MB ({abs(pytorch_size_mb - tf_size_mb) / max(pytorch_size_mb, tf_size_mb) * 100:.2f}%)")
        
    except ImportError:
        print("\nTensorFlow not available. Only showing PyTorch model information.")
        print("To install TensorFlow: pip install tensorflow")
        
        # Provide theoretical values for comparison
        print("\nFor reference, TensorFlow MobileNetV2 typically has:")
        print("Number of parameters: 3,538,984")
        print("Model size (approx): 13.50 MB")
        
        # Compare with theoretical values
        theoretical_tf_params = 3538984
        theoretical_tf_size = 13.50
        
        print("\nComparison with theoretical TensorFlow values:")
        print(f"Parameter difference: {abs(pytorch_params - theoretical_tf_params):,} ({abs(pytorch_params - theoretical_tf_params) / max(pytorch_params, theoretical_tf_params) * 100:.2f}%)")
        print(f"Size difference: {abs(pytorch_size_mb - theoretical_tf_size):.2f} MB ({abs(pytorch_size_mb - theoretical_tf_size) / max(pytorch_size_mb, theoretical_tf_size) * 100:.2f}%)")

if __name__ == "__main__":
    main()

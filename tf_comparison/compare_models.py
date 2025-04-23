"""
Direct comparison between PyTorch and TensorFlow MobileNetV2 implementations.
"""
import os
import sys
import time
import torch
import tensorflow as tf
import numpy as np
from torchvision.models import mobilenet_v2
from tensorflow.keras.applications import MobileNetV2 as TFMobileNetV2

def count_pytorch_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_pytorch_model_size_mb(model):
    """Calculate the size of a PyTorch model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def analyze_pytorch_layers(model):
    """Analyze the layer types in a PyTorch model."""
    layer_types = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            layer_type = module.__class__.__name__
            if layer_type not in layer_types:
                layer_types[layer_type] = {'count': 0, 'params': 0}
            layer_types[layer_type]['count'] += 1
            layer_types[layer_type]['params'] += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    return layer_types

def analyze_tf_layers(model):
    """Analyze the layer types in a TensorFlow model."""
    layer_types = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type not in layer_types:
            layer_types[layer_type] = {'count': 0, 'params': 0}
        layer_types[layer_type]['count'] += 1
        layer_types[layer_type]['params'] += layer.count_params()
        
        # For nested models like functional models
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                sublayer_type = sublayer.__class__.__name__
                if sublayer_type not in layer_types:
                    layer_types[sublayer_type] = {'count': 0, 'params': 0}
                layer_types[sublayer_type]['count'] += 1
                layer_types[sublayer_type]['params'] += sublayer.count_params()
    
    return layer_types

def benchmark_inference(model, framework, input_shape, num_runs=50):
    """Benchmark inference time for a model."""
    # Warm-up
    if framework == 'pytorch':
        model.eval()
        with torch.no_grad():
            x = torch.randn(input_shape)
            for _ in range(10):
                _ = model(x)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(torch.randn(input_shape))
        end_time = time.time()
    
    elif framework == 'tensorflow':
        x = tf.random.normal(input_shape)
        for _ in range(10):
            _ = model(x)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(tf.random.normal(input_shape))
        end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time * 1000  # Convert to ms

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    
    print("=" * 50)
    print("MobileNetV2 Implementation Comparison")
    print("=" * 50)
    
    # Load PyTorch MobileNetV2
    print("\nLoading PyTorch MobileNetV2...")
    pytorch_model = mobilenet_v2(weights=None)
    
    # Load TensorFlow MobileNetV2
    print("Loading TensorFlow MobileNetV2...")
    tf_model = TFMobileNetV2(weights=None, include_top=True)
    
    # Parameter counts
    pytorch_params = count_pytorch_parameters(pytorch_model)
    tf_params = tf_model.count_params()
    
    # Model sizes
    pytorch_size = get_pytorch_model_size_mb(pytorch_model)
    tf_size = tf_params * 4 / 1024**2  # Approximate size assuming float32 (4 bytes)
    
    print("\n" + "=" * 50)
    print("Parameter Count Comparison")
    print("=" * 50)
    print(f"PyTorch MobileNetV2: {pytorch_params:,} parameters")
    print(f"TensorFlow MobileNetV2: {tf_params:,} parameters")
    print(f"Difference: {abs(pytorch_params - tf_params):,} parameters ({abs(pytorch_params - tf_params) / max(pytorch_params, tf_params) * 100:.2f}%)")
    
    print("\n" + "=" * 50)
    print("Model Size Comparison")
    print("=" * 50)
    print(f"PyTorch MobileNetV2: {pytorch_size:.2f} MB")
    print(f"TensorFlow MobileNetV2: {tf_size:.2f} MB")
    print(f"Difference: {abs(pytorch_size - tf_size):.2f} MB ({abs(pytorch_size - tf_size) / max(pytorch_size, tf_size) * 100:.2f}%)")
    
    # Layer analysis
    pytorch_layers = analyze_pytorch_layers(pytorch_model)
    tf_layers = analyze_tf_layers(tf_model)
    
    print("\n" + "=" * 50)
    print("PyTorch Layer Analysis")
    print("=" * 50)
    for layer_type, data in pytorch_layers.items():
        print(f"{layer_type}: {data['count']} layers, {data['params']:,} parameters ({data['params']/pytorch_params*100:.2f}%)")
    
    print("\n" + "=" * 50)
    print("TensorFlow Layer Analysis")
    print("=" * 50)
    for layer_type, data in tf_layers.items():
        if data['params'] > 0:  # Only show layers with parameters
            print(f"{layer_type}: {data['count']} layers, {data['params']:,} parameters ({data['params']/tf_params*100:.2f}%)")
    
    # Inference benchmarking
    print("\n" + "=" * 50)
    print("Inference Benchmarking")
    print("=" * 50)
    
    batch_size = 1
    input_shape_pytorch = (batch_size, 3, 224, 224)
    input_shape_tf = (batch_size, 224, 224, 3)
    
    pytorch_time = benchmark_inference(pytorch_model, 'pytorch', input_shape_pytorch)
    tf_time = benchmark_inference(tf_model, 'tensorflow', input_shape_tf)
    
    print(f"PyTorch inference time (avg of 50 runs): {pytorch_time:.2f} ms")
    print(f"TensorFlow inference time (avg of 50 runs): {tf_time:.2f} ms")
    print(f"Ratio (TF/PyTorch): {tf_time/pytorch_time:.2f}x")
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"PyTorch MobileNetV2: {pytorch_params:,} parameters, {pytorch_size:.2f} MB, {pytorch_time:.2f} ms inference")
    print(f"TensorFlow MobileNetV2: {tf_params:,} parameters, {tf_size:.2f} MB, {tf_time:.2f} ms inference")

if __name__ == "__main__":
    main()

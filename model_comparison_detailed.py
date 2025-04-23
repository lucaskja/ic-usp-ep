"""
Detailed comparison of MobileNetV2 architecture between PyTorch and theoretical values.
"""
import torch
from torchvision.models import mobilenet_v2
import numpy as np

def count_parameters_by_layer(model):
    """Count parameters by layer in a PyTorch model."""
    results = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            results.append((name, module.__class__.__name__, params))
    return results

def analyze_mobilenetv2_architecture():
    """Analyze the MobileNetV2 architecture in detail."""
    # Load PyTorch MobileNetV2
    print("Loading PyTorch MobileNetV2...")
    model = mobilenet_v2(weights=None)
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    # Get parameters by layer type
    layer_types = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            layer_type = module.__class__.__name__
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print("\nParameters by layer type:")
    for layer_type, count in layer_types.items():
        print(f"{layer_type}: {count:,} ({count/total_params*100:.2f}%)")
    
    # Analyze inverted residual blocks
    print("\nAnalyzing inverted residual blocks:")
    features = model.features
    block_params = []
    current_block = 0
    current_block_params = 0
    
    for i, layer in enumerate(features):
        if i == 0:  # First conv layer
            first_conv_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            print(f"First Conv Layer: {first_conv_params:,} parameters")
            continue
            
        # Check if this is a new block
        if hasattr(layer, 'stride') and isinstance(layer.stride, (list, tuple)) and layer.stride[0] > 1:
            if current_block > 0:
                block_params.append((current_block, current_block_params))
            current_block += 1
            current_block_params = 0
            
        current_block_params += sum(p.numel() for p in layer.parameters() if p.requires_grad)
    
    # Add the last block
    block_params.append((current_block, current_block_params))
    
    for block_num, params in block_params:
        print(f"Inverted Residual Block {block_num}: {params:,} parameters")
    
    # Classifier parameters
    classifier_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(f"\nClassifier: {classifier_params:,} parameters")
    
    # Compare with theoretical TensorFlow MobileNetV2
    print("\nComparison with theoretical TensorFlow MobileNetV2:")
    print(f"PyTorch MobileNetV2: {total_params:,} parameters")
    print(f"TensorFlow MobileNetV2: 3,538,984 parameters")
    print(f"Difference: {abs(total_params - 3538984):,} parameters ({abs(total_params - 3538984)/3538984*100:.2f}%)")
    
    # Detailed layer-by-layer analysis
    print("\nDetailed layer analysis:")
    layer_params = count_parameters_by_layer(model)
    
    # Group by layer type and print summary
    layer_type_summary = {}
    for name, layer_type, params in layer_params:
        if layer_type not in layer_type_summary:
            layer_type_summary[layer_type] = {'count': 0, 'params': 0}
        layer_type_summary[layer_type]['count'] += 1
        layer_type_summary[layer_type]['params'] += params
    
    print("\nLayer type summary:")
    for layer_type, data in layer_type_summary.items():
        print(f"{layer_type}: {data['count']} layers, {data['params']:,} parameters ({data['params']/total_params*100:.2f}%)")

if __name__ == "__main__":
    analyze_mobilenetv2_architecture()

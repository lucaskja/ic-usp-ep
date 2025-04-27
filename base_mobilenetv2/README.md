# Base MobileNetV2 Implementation

This directory contains the implementation of the base MobileNetV2 architecture, which serves as the foundation for all improvements in this project.

## Overview

The base MobileNetV2 architecture features:
- Inverted residual structure
- Depthwise separable convolutions
- Linear bottlenecks
- ReLU6 activation function

## Directory Structure

```
base_mobilenetv2/
└── models/           # Model architecture implementation
    ├── __init__.py
    └── mobilenetv2.py  # Base MobileNetV2 model definition
```

## Usage

The base MobileNetV2 model is now used through the unified training and evaluation scripts at the project root:

### Training

```bash
# Train the base MobileNetV2 model
python train.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 32 --lr 0.001

# Train with enhanced data augmentation
python train.py --data_dir datasets/leaf_disease --model_type base --enhanced_augmentation --epochs 60 --batch_size 32 --lr 0.001

# Train with GPU acceleration
python train.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

### Evaluation

```bash
# Evaluate the base MobileNetV2 model
python evaluate.py --data_dir datasets/leaf_disease --model_type base --checkpoint checkpoints/base/model_best.pth

# Evaluate with enhanced preprocessing
python evaluate.py --data_dir datasets/leaf_disease --model_type base --checkpoint checkpoints/base/model_best.pth --enhanced_preprocessing

# Evaluate with GPU acceleration
python evaluate.py --data_dir datasets/leaf_disease --model_type base --checkpoint checkpoints/base/model_best.pth --device cuda
```

## Implementation Details

The base MobileNetV2 model is implemented in `models/mobilenetv2.py` and includes:

1. **Inverted Residual Block**:
   - Expands the input channels
   - Applies depthwise convolution
   - Projects back to a smaller number of channels

2. **Depthwise Separable Convolutions**:
   - Depthwise convolution: applies a single filter per input channel
   - Pointwise convolution: 1×1 convolution to combine the outputs

3. **Linear Bottlenecks**:
   - Uses linear activation in the bottleneck layers
   - Preserves information in low-dimensional spaces

4. **ReLU6 Activation**:
   - ReLU capped at 6: min(max(0, x), 6)
   - Provides robustness for low-precision computation

## Model Creation

To create a base MobileNetV2 model programmatically:

```python
from base_mobilenetv2.models.mobilenetv2 import create_mobilenetv2

# Create a model for 3 classes
model = create_mobilenetv2(num_classes=3, pretrained=True)
```

Or use the unified model factory:

```python
from utils.model_factory import create_model

# Create a model for 3 classes
model = create_model('base', num_classes=3, pretrained=True)
```

# MobileNetV2 with Mish Activation

This directory contains the implementation of MobileNetV2 with Mish activation function, which is the first improvement stage in this project.

## Overview

The MobileNetV2 with Mish activation features:
- All components of the base MobileNetV2 architecture
- Replacement of ReLU6 with Mish activation function
- Improved gradient flow and feature representation

## Directory Structure

```
stage1_mish/
└── models/           # Model architecture implementation
    ├── __init__.py
    ├── mish.py       # Mish activation function implementation
    └── mobilenetv2_mish.py  # MobileNetV2 with Mish model definition
```

## Usage

The MobileNetV2 with Mish model is now used through the unified training and evaluation scripts at the project root:

### Training

```bash
# Train the MobileNetV2 with Mish model
python train.py --data_dir datasets/leaf_disease --model_type mish --epochs 60 --batch_size 32 --lr 0.001

# Train with enhanced data augmentation
python train.py --data_dir datasets/leaf_disease --model_type mish --enhanced_augmentation --epochs 60 --batch_size 32 --lr 0.001

# Train with GPU acceleration
python train.py --data_dir datasets/leaf_disease --model_type mish --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

### Evaluation

```bash
# Evaluate the MobileNetV2 with Mish model
python evaluate.py --data_dir datasets/leaf_disease --model_type mish --checkpoint checkpoints/mish/model_best.pth

# Evaluate with enhanced preprocessing
python evaluate.py --data_dir datasets/leaf_disease --model_type mish --checkpoint checkpoints/mish/model_best.pth --enhanced_preprocessing

# Evaluate with GPU acceleration
python evaluate.py --data_dir datasets/leaf_disease --model_type mish --checkpoint checkpoints/mish/model_best.pth --device cuda
```

## Implementation Details

### Mish Activation Function

Mish is a self-regularized non-monotonic activation function defined as:

```
f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
```

### Key Advantages Over ReLU6

1. **Smoothness**: Mish is smooth and non-monotonic, allowing better information flow
2. **Unbounded Above**: Unlike ReLU6, Mish has no upper bound, preserving important features
3. **Bounded Below**: Slightly bounded below, providing regularization benefits
4. **Better Gradient Flow**: Reduces the vanishing gradient problem
5. **Preserves Slight Negative Values**: Can help preserve important negative information

### Implementation

The Mish activation function is implemented in `models/mish.py` and integrated into the MobileNetV2 architecture in `models/mobilenetv2_mish.py`.

## Model Creation

To create a MobileNetV2 with Mish model programmatically:

```python
from stage1_mish.models.mobilenetv2_mish import create_mobilenetv2_mish

# Create a model for 3 classes
model = create_mobilenetv2_mish(num_classes=3, pretrained=True)
```

Or use the unified model factory:

```python
from utils.model_factory import create_model

# Create a model for 3 classes
model = create_model('mish', num_classes=3, pretrained=True)
```

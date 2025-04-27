# MobileNetV2 with Mish and Triplet Attention

This directory contains the implementation of MobileNetV2 with Mish activation and Triplet Attention, which is the second improvement stage in this project.

## Overview

The MobileNetV2 with Mish and Triplet Attention features:
- All components of the MobileNetV2 with Mish architecture
- Addition of Triplet Attention mechanism after inverted residual blocks
- Improved feature representation through cross-dimension interactions

## Directory Structure

```
stage2_triplet/
└── models/           # Model architecture implementation
    ├── __init__.py
    ├── triplet_attention.py  # Triplet Attention implementation
    └── mobilenetv2_triplet.py  # MobileNetV2 with Mish and Triplet Attention model definition
```

## Usage

The MobileNetV2 with Mish and Triplet Attention model is now used through the unified training and evaluation scripts at the project root:

### Training

```bash
# Train the MobileNetV2 with Mish and Triplet Attention model
python train.py --data_dir datasets/leaf_disease --model_type triplet --epochs 60 --batch_size 32 --lr 0.001

# Train with enhanced data augmentation
python train.py --data_dir datasets/leaf_disease --model_type triplet --enhanced_augmentation --epochs 60 --batch_size 32 --lr 0.001

# Train with GPU acceleration
python train.py --data_dir datasets/leaf_disease --model_type triplet --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

### Evaluation

```bash
# Evaluate the MobileNetV2 with Mish and Triplet Attention model
python evaluate.py --data_dir datasets/leaf_disease --model_type triplet --checkpoint checkpoints/triplet/model_best.pth

# Evaluate with enhanced preprocessing
python evaluate.py --data_dir datasets/leaf_disease --model_type triplet --checkpoint checkpoints/triplet/model_best.pth --enhanced_preprocessing

# Evaluate with GPU acceleration
python evaluate.py --data_dir datasets/leaf_disease --model_type triplet --checkpoint checkpoints/triplet/model_best.pth --device cuda
```

## Implementation Details

### Triplet Attention Mechanism

Triplet Attention is a lightweight attention mechanism that captures cross-dimension interactions through a three-branch structure.

#### Key Components

1. **Three Parallel Branches**:
   - Each branch focuses on different dimensional interactions
   - Branch 1: Channel-Height interaction
   - Branch 2: Channel-Width interaction
   - Branch 3: Standard spatial attention

2. **Z-Pool Operation**:
   - Combines max-pooling and average-pooling along the channel dimension
   - Captures both prominent and overall feature distributions

3. **No Channel Reduction**:
   - Unlike many attention mechanisms, Triplet Attention doesn't reduce channel dimensions
   - Preserves full information flow

### Integration with MobileNetV2

Triplet Attention is added after each inverted residual block in the MobileNetV2 architecture, enhancing the model's ability to capture important features across different dimensions.

## Model Creation

To create a MobileNetV2 with Mish and Triplet Attention model programmatically:

```python
from stage2_triplet.models.mobilenetv2_triplet import create_mobilenetv2_triplet

# Create a model for 3 classes
model = create_mobilenetv2_triplet(num_classes=3, pretrained=True)
```

Or use the unified model factory:

```python
from utils.model_factory import create_model

# Create a model for 3 classes
model = create_model('triplet', num_classes=3, pretrained=True)
```

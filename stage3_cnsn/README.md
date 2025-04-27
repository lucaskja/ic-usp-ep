# MobileNetV2 with Mish, Triplet Attention, and CNSN

This directory contains the implementation of MobileNetV2 with Mish activation, Triplet Attention, and CrossNorm-SelfNorm (CNSN) modules, which is the third and final improvement stage in this project.

## Overview

The MobileNetV2 with Mish, Triplet Attention, and CNSN features:
- All components of the MobileNetV2 with Mish and Triplet Attention architecture
- Addition of CrossNorm (CN) and SelfNorm (SN) modules
- Improved generalization through feature distribution manipulation

## Directory Structure

```
stage3_cnsn/
└── models/           # Model architecture implementation
    ├── __init__.py
    ├── cnsn.py       # CrossNorm and SelfNorm implementation
    └── mobilenetv2_cnsn.py  # MobileNetV2 with Mish, Triplet Attention, and CNSN model definition
```

## Usage

The MobileNetV2 with Mish, Triplet Attention, and CNSN model is now used through the unified training and evaluation scripts at the project root:

### Training

```bash
# Train the MobileNetV2 with Mish, Triplet Attention, and CNSN model
python train.py --data_dir datasets/leaf_disease --model_type cnsn --epochs 60 --batch_size 32 --lr 0.001

# Train with enhanced data augmentation
python train.py --data_dir datasets/leaf_disease --model_type cnsn --enhanced_augmentation --epochs 60 --batch_size 32 --lr 0.001

# Train with GPU acceleration
python train.py --data_dir datasets/leaf_disease --model_type cnsn --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

### Evaluation

```bash
# Evaluate the MobileNetV2 with Mish, Triplet Attention, and CNSN model
python evaluate.py --data_dir datasets/leaf_disease --model_type cnsn --checkpoint checkpoints/cnsn/model_best.pth

# Evaluate with enhanced preprocessing
python evaluate.py --data_dir datasets/leaf_disease --model_type cnsn --checkpoint checkpoints/cnsn/model_best.pth --enhanced_preprocessing

# Evaluate with GPU acceleration
python evaluate.py --data_dir datasets/leaf_disease --model_type cnsn --checkpoint checkpoints/cnsn/model_best.pth --device cuda
```

## Implementation Details

### CNSN (CrossNorm and SelfNorm)

CNSN combines two complementary normalization techniques:

#### 1. CrossNorm (CN)

- **Purpose**: Enlarges the training distribution by exchanging channel-wise statistics between feature maps
- **Operation**: Exchanges channel-wise mean and variance between feature maps
- **Variants**:
  - 1-instance mode: Exchange between channels within same instance
  - 2-instance mode: Exchange between corresponding channels of different instances
  - Crop mode: Apply to specific spatial regions
- **Training Only**: CrossNorm is only active during training

#### 2. SelfNorm (SN)

- **Purpose**: Bridges train-test distribution gap
- **Operation**: Recalibrates statistics using attention mechanism
- **Components**:
  - Two FC networks for attention functions f and g
  - Processes each channel independently
  - No dimensionality reduction
- **Always Active**: SelfNorm works during both training and testing

### Integration with MobileNetV2

CNSN modules are added after each inverted residual block, following the Triplet Attention module. This combination provides both improved feature representation (from Triplet Attention) and better generalization (from CNSN).

## Model Creation

To create a MobileNetV2 with Mish, Triplet Attention, and CNSN model programmatically:

```python
from stage3_cnsn.models.mobilenetv2_cnsn import create_mobilenetv2_cnsn

# Create a model for 3 classes
model = create_mobilenetv2_cnsn(num_classes=3, pretrained=True)
```

Or use the unified model factory:

```python
from utils.model_factory import create_model

# Create a model for 3 classes
model = create_model('cnsn', num_classes=3, pretrained=True)
```

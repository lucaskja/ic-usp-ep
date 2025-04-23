# MobileNetV2 Improvements for Leaf Disease Classification - Project Summary

## Project Overview

This research project focuses on enhancing the MobileNetV2 architecture for leaf disease classification tasks. The improvements are implemented in three progressive stages, each building upon the previous one, with the goal of improving classification accuracy while maintaining computational efficiency.

## Workspace Structure Analysis

### Directory Organization
The project is organized into several key directories:
- **Main Implementation Stages**:
  - `base_mobilenetv2/`: Baseline MobileNetV2 implementation
  - `stage1_mish/`: MobileNetV2 with Mish activation
  - `stage2_triplet/`: MobileNetV2 with Mish and Triplet Attention
  - `stage3_cnsn/`: MobileNetV2 with Mish, Triplet Attention, and CNSN
- **Support Directories**:
  - `datasets/`: Contains leaf disease datasets including "rice leaf diseases dataset"
  - `utils/`: Shared utility functions
  - `tests/`: Test cases for model implementations
  - `experiments/`: Results and logs from model training
  - `docs/`: Documentation including model comparison details

### Virtual Environments
The project uses two separate virtual environments:
1. **Main PyTorch Environment** (`venv/`):
   - Primary environment for PyTorch-based model implementations
   - Contains dependencies: torch, torchvision, numpy, matplotlib, scikit-learn, tqdm, pillow, seaborn
   - Used for all main model implementations and evaluations

2. **TensorFlow Comparison Environment** (`tf_comparison/tf_venv/`):
   - Dedicated environment for TensorFlow implementation comparison
   - Contains TensorFlow 2.19.0 for framework comparison purposes
   - Used exclusively for the comparison study in the `tf_comparison/` directory

### Framework Comparison Study
The project includes a dedicated comparison between PyTorch and TensorFlow implementations:
- Located in the `tf_comparison/` directory
- Compares parameter counts, model sizes, layer distributions, and inference performance
- Key findings show PyTorch implementation is ~20% faster for inference
- Parameter difference is minimal (0.96%) between frameworks

### Optimization Opportunities
1. **Environment Consolidation**: Consider merging the environments if framework comparison is complete
2. **Shared Code Refactoring**: Extract common code patterns across implementation stages
3. **Standardized Evaluation Pipeline**: Create unified evaluation scripts across all stages
4. **Automated Testing**: Expand test coverage for model implementations
5. **Documentation Integration**: Better integrate framework comparison findings with main documentation

## Core Components

### Base Model: MobileNetV2
- **Architecture**: Lightweight CNN with inverted residual structure
- **Key Features**:
  - Depthwise separable convolutions
  - Linear bottlenecks
  - ReLU6 activation function
- **Purpose**: Serves as the baseline for comparison

### Stage 1: MobileNetV2 with Mish Activation
- **Improvement**: Replaces ReLU6 with Mish activation function
- **Benefits**:
  - Smoother activation curve
  - Better gradient propagation
  - No upper bound truncation (unlike ReLU6's limit at 6)
  - Preserves negative information
- **Formula**: `f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))`

### Stage 2: MobileNetV2 with Mish and Triplet Attention
- **Improvement**: Adds Triplet Attention mechanism after inverted residual blocks
- **Architecture**:
  - Three parallel branches focusing on different dimensional interactions:
    1. Branch 1: Channel-Height interaction
    2. Branch 2: Channel-Width interaction
    3. Branch 3: Standard spatial attention
  - Z-Pool operation (combines max-pooling and average-pooling)
  - No channel reduction
- **Benefits**:
  - Captures cross-dimension interactions
  - Parameter-efficient
  - Minimal computational overhead

### Stage 3: MobileNetV2 with Mish, Triplet Attention, and CNSN
- **Improvement**: Integrates CrossNorm and SelfNorm modules
- **Components**:
  1. **CrossNorm (CN)**:
     - Enlarges training distribution by exchanging channel-wise statistics
     - Only active during training
     - Modes: 1-instance, 2-instance, and crop
  2. **SelfNorm (SN)**:
     - Bridges train-test distribution gap
     - Uses attention mechanism to recalibrate statistics
     - Active during both training and testing
- **Benefits**:
  - Improved generalization
  - Better feature representation
  - Reduced overfitting

## Evaluation Methodology

### K-Fold Cross-Validation
- **Implementation**: 5-fold cross-validation (k=5)
- **Process**:
  1. Dataset divided into 5 equal parts (folds)
  2. For each fold:
     - Current fold used as validation set
     - Remaining 4 folds used as training set
     - Model trained and evaluated
  3. Process repeated 5 times
  4. Final metrics averaged across all 5 runs
- **Benefits**:
  - Robust evaluation
  - Reduced variance
  - Efficient data usage
  - Fair comparison between models

### Dataset Structure
- **Organization**:
  ```
  datasets/
  └── leaf_disease/
      ├── train/
      │   ├── class1/
      │   ├── class2/
      │   └── ...
      └── test/
          ├── class1/
          ├── class2/
          └── ...
  ```
- **Preprocessing**:
  - Resizing to 224×224 pixels
  - Normalization using ImageNet mean and standard deviation
  - Data augmentation (flips, rotations, color jitter)

## Implementation Details

### Code Structure
```
mobilenetv2_improvements/
├── base_mobilenetv2/     # Base MobileNetV2 implementation
├── stage1_mish/          # MobileNetV2 with Mish activation
├── stage2_triplet/       # MobileNetV2 with Mish and Triplet Attention
├── stage3_cnsn/          # MobileNetV2 with Mish, Triplet Attention, and CNSN
├── datasets/             # Dataset storage
├── utils/                # Utility functions
├── tests/                # Test cases
└── experiments/          # Experiment results and logs
```

### Training Configuration
- **Optimizer**: SGD with momentum 0.9 and weight decay 1e-4
- **Learning Rate**: Initial 0.001 with step decay (factor 0.1 every 10 epochs)
- **Epochs**: 50-60
- **Batch Size**: 32

### Metrics Tracked
- Model accuracy
- Parameter count
- Computational overhead (FLOPs)
- Training time
- Memory usage
- Inference speed

## Expected Performance Improvements

| Model | Expected Improvement |
|-------|----------------------|
| Base MobileNetV2 | Baseline |
| MobileNetV2 + Mish | +1-2% accuracy, better convergence |
| MobileNetV2 + Mish + Triplet Attention | +2-3% accuracy, better feature representation |
| MobileNetV2 + Mish + Triplet Attention + CNSN | +3-5% accuracy, better generalization |

## Environment Setup

### Virtual Environment
- Python virtual environment required for all operations
- Activation: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Set PYTHONPATH to project root when needed: `PYTHONPATH=/Users/lucaskle/Documents/USP/ic-usp-ep`

### Running Models
- Training commands available for each stage
- Evaluation scripts for testing model performance
- Visualization tools for comparing results

## Development Guidelines
- Follow PEP 8 style guidelines
- Write docstrings for all functions and classes
- Add unit tests for new functionality
- Use git branches for new features
- Run tests before committing changes

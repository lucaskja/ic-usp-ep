# MobileNetV2 Improvements for Leaf Disease Classification

This project implements and evaluates several improvements to the MobileNetV2 architecture for leaf disease classification tasks. The improvements are implemented in stages, allowing for incremental testing and evaluation.

## Project Structure

The project has been reorganized with a unified structure:

```
mobilenetv2_improvements/
├── base_mobilenetv2/     # Base MobileNetV2 model definition
│   └── models/           # Model architecture implementation
├── stage1_mish/          # MobileNetV2 with Mish activation
│   └── models/           # Model architecture implementation
├── stage2_triplet/       # MobileNetV2 with Mish and Triplet Attention
│   └── models/           # Model architecture implementation
├── stage3_cnsn/          # MobileNetV2 with Mish, Triplet Attention, and CNSN
│   └── models/           # Model architecture implementation
├── configs/              # Unified configuration files
│   └── model_configs.py  # Configuration for all model variants
├── utils/                # Unified utility functions
│   ├── data_utils.py     # Standard data loading utilities
│   ├── enhanced_data_utils.py # Enhanced data augmentation
│   ├── evaluator.py      # Unified model evaluation
│   ├── model_factory.py  # Factory pattern for model creation
│   ├── model_utils.py    # Model utility functions
│   ├── training_utils.py # Training utility functions
│   └── visualize_transforms.py # Visualization tools for data transforms
├── datasets/             # Dataset storage (not tracked by git)
├── checkpoints/          # Model checkpoints (organized by model type)
├── experiments/          # Experiment results and logs
├── logs/                 # Training and evaluation logs
├── docs/                 # Documentation
├── tests/                # Test cases
├── train.py              # Unified training script for all models
└── evaluate.py           # Unified evaluation script for all models
```

## Model Improvements

### 1. Base MobileNetV2

The foundation model with:
- Inverted residual structure
- Depthwise separable convolutions
- Linear bottlenecks
- ReLU6 activation function

### 2. MobileNetV2 with Mish Activation (Stage 1)

Replaces ReLU6 with Mish activation function to enhance nonlinear characteristics:
- Smoother activation function with formula: `f(x) = x * tanh(softplus(x))`
- Better gradient propagation with no vanishing gradient issues
- No upper bound truncation (unlike ReLU6's limit at 6)
- Allows negative outputs, preserving important negative information

### 3. MobileNetV2 with Mish and Triplet Attention (Stage 2)

Adds a three-branch attention structure for capturing cross-dimension interactions:
- Three parallel branches focusing on different dimensional interactions:
  - Branch 1: Channel-Height interaction
  - Branch 2: Channel-Width interaction
  - Branch 3: Standard spatial attention
- Z-Pool operation combines max-pooling and average-pooling
- No channel reduction, preserving full information flow
- Parameter-efficient design with minimal computational overhead

### 4. MobileNetV2 with Mish, Triplet Attention, and CNSN (Stage 3)

Integrates CrossNorm and SelfNorm modules:
- CrossNorm: Enlarges training distribution by exchanging channel-wise statistics
  - Only active during training
  - Creates diverse feature representations
  - Multiple modes: 1-instance, 2-instance, and crop
- SelfNorm: Bridges train-test distribution gap using an attention mechanism
  - Active during both training and testing
  - Uses attention functions to recalibrate statistics
  - Processes each channel independently

## Setup Instructions

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mobilenetv2_improvements
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Acceleration Setup

For faster training with NVIDIA GPUs:

1. Install CUDA Toolkit and cuDNN:
   - Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.8 or 12.1 recommended)
   - Download and install [cuDNN](https://developer.nvidia.com/cudnn) (requires free NVIDIA account)

2. Install GPU-enabled PyTorch:
```bash
# Activate your virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# For CUDA 11.8
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. Verify GPU detection:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Dataset Preparation

1. Download a leaf disease dataset (e.g., Plant Village, PlantDoc, Rice Leaf Disease Dataset)
2. Organize it according to the following structure:
```
datasets/
└── leaf_disease/
    ├── class1/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    ├── class2/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    └── ...
```

The dataset will be automatically split into train, validation, and test sets when first used.

## Running the Models

### Training

All model variants are now trained using the unified `train.py` script:

#### Basic Training Commands

```bash
# Base MobileNetV2
python train.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish
python train.py --data_dir datasets/leaf_disease --model_type mish --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish and Triplet Attention
python train.py --data_dir datasets/leaf_disease --model_type triplet --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish, Triplet Attention, and CNSN
python train.py --data_dir datasets/leaf_disease --model_type cnsn --epochs 60 --batch_size 32 --lr 0.001
```

#### Training with Enhanced Data Augmentation

```bash
# Add --enhanced_augmentation flag to use enhanced data augmentation
python train.py --data_dir datasets/leaf_disease --model_type base --enhanced_augmentation --epochs 60 --batch_size 32 --lr 0.001
```

#### GPU Training Commands

```bash
# Add --device cuda to use GPU acceleration
python train.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

#### Training All Models Sequentially

```bash
# Use --train_all flag to train all models in sequence (base -> mish -> triplet -> cnsn)
python train.py --data_dir datasets/leaf_disease --train_all --epochs 60 --batch_size 32 --lr 0.001 --device cuda
```

### Evaluation

All model variants are evaluated using the unified `evaluate.py` script:

```bash
# Base MobileNetV2
python evaluate.py --data_dir datasets/leaf_disease --model_type base --checkpoint checkpoints/base/model_best.pth

# MobileNetV2 with Mish
python evaluate.py --data_dir datasets/leaf_disease --model_type mish --checkpoint checkpoints/mish/model_best.pth

# MobileNetV2 with Mish and Triplet Attention
python evaluate.py --data_dir datasets/leaf_disease --model_type triplet --checkpoint checkpoints/triplet/model_best.pth

# MobileNetV2 with Mish, Triplet Attention, and CNSN
python evaluate.py --data_dir datasets/leaf_disease --model_type cnsn --checkpoint checkpoints/cnsn/model_best.pth
```

#### Evaluation with Enhanced Preprocessing

```bash
# Add --enhanced_preprocessing flag to use enhanced preprocessing
python evaluate.py --data_dir datasets/leaf_disease --model_type base --checkpoint checkpoints/base/model_best.pth --enhanced_preprocessing
```

#### GPU Evaluation Commands

```bash
# Add --device cuda to use GPU acceleration
python evaluate.py --data_dir datasets/leaf_disease --model_type base --checkpoint checkpoints/base/model_best.pth --device cuda
```

### Data Augmentation Visualization

To visualize how data augmentation affects your training images:

```bash
# Visualize standard augmentations
python utils/visualize_transforms.py --image_path datasets/leaf_disease/class1/img001.jpg --output_dir experiments/visualizations

# Visualize enhanced augmentations
python utils/visualize_transforms.py --image_path datasets/leaf_disease/class1/img001.jpg --output_dir experiments/visualizations --enhanced
```

## Dataset Split Strategy

This project uses a three-way split strategy:
1. **Test set**: 10% of the entire dataset
2. **Training set**: 72% of the entire dataset (80% of the remaining 90%)
3. **Validation set**: 18% of the entire dataset (20% of the remaining 90%)

The split is performed automatically when loading the dataset for the first time, creating separate directories for each split while maintaining the class structure.

## Testing

Run tests to verify the implementation:

```bash
pytest
```

## Development Guidelines

1. Follow PEP 8 style guidelines
2. Write docstrings for all functions and classes
3. Add unit tests for new functionality
4. Use git branches for new features
5. Run tests before committing changes

## License

[MIT License](LICENSE)

# MobileNetV2 Improvements for Leaf Disease Classification

This project implements and evaluates several improvements to the MobileNetV2 architecture for leaf disease classification tasks. The improvements are implemented in stages, allowing for incremental testing and evaluation.

## Project Structure

```
mobilenetv2_improvements/
├── base_mobilenetv2/     # Base MobileNetV2 implementation
├── stage1_mish/          # MobileNetV2 with Mish activation
├── stage2_triplet/       # MobileNetV2 with Mish and Triplet Attention
├── stage3_cnsn/          # MobileNetV2 with Mish, Triplet Attention, and CNSN
├── datasets/             # Dataset storage (not tracked by git)
├── docs/                 # Documentation
│   └── model_comparison/ # Framework comparison documentation
├── experiments/          # Experiment results and logs
│   ├── model_comparison/ # Framework comparison scripts
│   ├── enhanced/         # Results with enhanced augmentation
│   └── visualizations/   # Data augmentation visualizations
├── utils/                # Utility functions
│   ├── data_utils.py     # Standard data loading utilities
│   ├── enhanced_data_utils.py # Enhanced data augmentation
│   └── visualize_transforms.py # Visualization tools for data transforms
├── train_enhanced.py     # Unified training script with enhanced augmentation
└── tests/                # Test cases
```

## Recent Updates

### Enhanced Data Augmentation

We've added enhanced data augmentation techniques to improve model generalization and prevent overfitting:

- **Stronger Augmentations**: More aggressive transformations including perspective changes, affine transformations, random erasing, and stronger color jittering
- **Visualization Tools**: New utilities to visualize how data augmentation affects training images
- **Unified Training Script**: A single script to train any model variant with enhanced augmentation

#### Enhanced Augmentation Techniques

The enhanced data augmentation pipeline includes:

- RandomResizedCrop with scale variation (70-100%)
- RandomHorizontalFlip (50% probability)
- RandomVerticalFlip (30% probability)
- RandomRotation (up to 20 degrees)
- ColorJitter (brightness, contrast, saturation, hue)
- RandomAffine (translation and scaling)
- RandomPerspective (perspective transformations)
- RandomGrayscale (10% probability)
- RandomErasing (simulates occlusion)

These techniques significantly increase the effective size of the training dataset by creating diverse variations of each image, helping the model learn more robust features.

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

For more detailed descriptions of each model, see [model_descriptions.md](model_descriptions.md).

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

### GPU Acceleration Setup (Windows with NVIDIA GPUs)

For faster training with NVIDIA GPUs (e.g., RTX 4060 Ti):

1. Install CUDA Toolkit and cuDNN:
   - Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.8 or 12.1 recommended)
   - Download and install [cuDNN](https://developer.nvidia.com/cudnn) (requires free NVIDIA account)

2. Install GPU-enabled PyTorch:
```bash
# Activate your virtual environment first
venv\Scripts\activate

# For CUDA 11.8
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. Verify GPU detection:
```bash
# Create and run a test script
echo "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" > test_gpu.py
python test_gpu.py
```

### Dataset Preparation

1. Download a leaf disease dataset (e.g., Plant Village, PlantDoc)
2. Organize it according to the structure in [dataset_structure.md](dataset_structure.md)

## Running the Models Locally

### Training

To train the models with Holdout validation (80% training, 20% validation):

#### Basic Training Commands

```bash
# Base MobileNetV2
cd base_mobilenetv2
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 50 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish
cd stage1_mish
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 50 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish and Triplet Attention
cd stage2_triplet
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 50 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish, Triplet Attention, and CNSN
cd stage3_cnsn
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 50 --batch_size 32 --lr 0.001
```

#### GPU Training Commands (Linux/macOS)

For systems with GPU acceleration:

```bash
# Base MobileNetV2 with GPU
cd base_mobilenetv2
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 50 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish
cd stage1_mish
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 50 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and Triplet Attention
cd stage2_triplet
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 50 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN
cd stage3_cnsn
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 50 --batch_size 64 --lr 0.001 --device cuda
```

#### Windows GPU Training Commands

For Windows with GPU acceleration:

```bash
# Base MobileNetV2 with GPU
cd base_mobilenetv2
venv\Scripts\activate && python train.py --data_dir ..\datasets\leaf_disease --epochs 50 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish
cd stage1_mish
venv\Scripts\activate && python train.py --data_dir ..\datasets\leaf_disease --epochs 50 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and Triplet Attention
cd stage2_triplet
venv\Scripts\activate && python train.py --data_dir ..\datasets\leaf_disease --epochs 50 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN
cd stage3_cnsn
venv\Scripts\activate && python train.py --data_dir ..\datasets\leaf_disease --epochs 50 --batch_size 64 --lr 0.001 --device cuda
```

#### Advanced Training with Enhanced Data Augmentation

For training with enhanced data augmentation (recommended for better generalization):

```bash
# Linux/macOS commands

# Base MobileNetV2 with enhanced augmentation
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish and enhanced augmentation
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type mish --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish and Triplet Attention and enhanced augmentation
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type triplet --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish, Triplet Attention, and CNSN and enhanced augmentation
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type cnsn --epochs 60 --batch_size 32 --lr 0.001
```

#### GPU Training with Enhanced Augmentation (Linux/macOS)

```bash
# Base MobileNetV2 with enhanced augmentation on GPU
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and enhanced augmentation on GPU
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type mish --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and Triplet Attention and enhanced augmentation on GPU
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type triplet --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN and enhanced augmentation on GPU
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type cnsn --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

#### Windows GPU Training with Enhanced Augmentation

```bash
# Base MobileNetV2 with enhanced augmentation on GPU
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type base --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and enhanced augmentation on GPU
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type mish --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and Triplet Attention and enhanced augmentation on GPU
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type triplet --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN and enhanced augmentation on GPU
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type cnsn --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

### Evaluation

To evaluate the models on the test set:

```bash
# Base MobileNetV2
cd base_mobilenetv2
source ../venv/bin/activate && python evaluate.py --data_dir ../datasets/leaf_disease/test --checkpoint checkpoints/best.pth

# MobileNetV2 with Mish
cd stage1_mish
source ../venv/bin/activate && python evaluate.py --data_dir ../datasets/leaf_disease/test --checkpoint checkpoints/best.pth

# MobileNetV2 with Mish and Triplet Attention
cd stage2_triplet
source ../venv/bin/activate && python evaluate.py --data_dir ../datasets/leaf_disease/test --checkpoint checkpoints/best.pth

# MobileNetV2 with Mish, Triplet Attention, and CNSN
cd stage3_cnsn
source ../venv/bin/activate && python evaluate.py --data_dir ../datasets/leaf_disease/test --checkpoint checkpoints/best.pth
```

#### GPU Evaluation Commands (Linux/macOS)

```bash
# Base MobileNetV2 with GPU
cd base_mobilenetv2
source ../venv/bin/activate && python evaluate.py --data_dir ../datasets/leaf_disease/test --checkpoint checkpoints/best.pth --device cuda

# MobileNetV2 with Mish
cd stage1_mish
source ../venv/bin/activate && python evaluate.py --data_dir ../datasets/leaf_disease/test --checkpoint checkpoints/best.pth --device cuda

# MobileNetV2 with Mish and Triplet Attention
cd stage2_triplet
source ../venv/bin/activate && python evaluate.py --data_dir ../datasets/leaf_disease/test --checkpoint checkpoints/best.pth --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN
cd stage3_cnsn
source ../venv/bin/activate && python evaluate.py --data_dir ../datasets/leaf_disease/test --checkpoint checkpoints/best.pth --device cuda
```

#### Windows GPU Evaluation Commands

```bash
# Base MobileNetV2 with GPU
cd base_mobilenetv2
venv\Scripts\activate && python evaluate.py --data_dir ..\datasets\leaf_disease\test --checkpoint checkpoints\best.pth --device cuda

# MobileNetV2 with Mish
cd stage1_mish
venv\Scripts\activate && python evaluate.py --data_dir ..\datasets\leaf_disease\test --checkpoint checkpoints\best.pth --device cuda

# MobileNetV2 with Mish and Triplet Attention
cd stage2_triplet
venv\Scripts\activate && python evaluate.py --data_dir ..\datasets\leaf_disease\test --checkpoint checkpoints\best.pth --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN
cd stage3_cnsn
venv\Scripts\activate && python evaluate.py --data_dir ..\datasets\leaf_disease\test --checkpoint checkpoints\best.pth --device cuda
```

### Visualization and Analysis

To generate visualizations and performance comparisons:

```bash
# Linux/macOS
source venv/bin/activate && python analyze.py --results_dir experiments/results --output_dir experiments/visualizations

# Windows
venv\Scripts\activate && python analyze.py --results_dir experiments\results --output_dir experiments\visualizations
```

### Data Augmentation Visualization

To visualize how data augmentation affects your training images:

```bash
# Linux/macOS commands
# Visualize standard augmentations
source venv/bin/activate && python utils/visualize_transforms.py --image_path datasets/leaf_disease/train/Bacterialblight/BACTERAILBLIGHT3_001.jpg --output_dir experiments/visualizations

# Visualize enhanced augmentations
source venv/bin/activate && python utils/visualize_transforms.py --image_path datasets/leaf_disease/train/Bacterialblight/BACTERAILBLIGHT3_001.jpg --output_dir experiments/visualizations --enhanced

# Windows commands
# Visualize standard augmentations
venv\Scripts\activate && python utils\visualize_transforms.py --image_path datasets\leaf_disease\train\Bacterialblight\BACTERAILBLIGHT3_001.jpg --output_dir experiments\visualizations

# Visualize enhanced augmentations
venv\Scripts\activate && python utils\visualize_transforms.py --image_path datasets\leaf_disease\train\Bacterialblight\BACTERAILBLIGHT3_001.jpg --output_dir experiments\visualizations --enhanced
```

This will generate visualizations showing the original image alongside multiple augmented versions, helping you understand how your data is being transformed during training.

#### Linux/macOS Training Commands with Enhanced Augmentation

```bash
# Base MobileNetV2 with enhanced augmentation
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish and enhanced augmentation
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type mish --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish and Triplet Attention and enhanced augmentation
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type triplet --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish, Triplet Attention, and CNSN and enhanced augmentation
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type cnsn --epochs 60 --batch_size 32 --lr 0.001
```

#### Linux/macOS GPU Training Commands with Enhanced Augmentation

```bash
# Base MobileNetV2 with enhanced augmentation on GPU
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type base --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and enhanced augmentation on GPU
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type mish --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and Triplet Attention and enhanced augmentation on GPU
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type triplet --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN and enhanced augmentation on GPU
source venv/bin/activate && python train_enhanced.py --data_dir datasets/leaf_disease --model_type cnsn --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

#### Windows Training Commands with Enhanced Augmentation

```bash
# Base MobileNetV2 with enhanced augmentation
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type base --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish and enhanced augmentation
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type mish --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish and Triplet Attention and enhanced augmentation
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type triplet --epochs 60 --batch_size 32 --lr 0.001

# MobileNetV2 with Mish, Triplet Attention, and CNSN and enhanced augmentation
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type cnsn --epochs 60 --batch_size 32 --lr 0.001
```

#### Windows GPU Training Commands with Enhanced Augmentation

```bash
# Base MobileNetV2 with enhanced augmentation on GPU
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type base --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and enhanced augmentation on GPU
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type mish --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish and Triplet Attention and enhanced augmentation on GPU
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type triplet --epochs 60 --batch_size 64 --lr 0.001 --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN and enhanced augmentation on GPU
venv\Scripts\activate && python train_enhanced.py --data_dir datasets\leaf_disease --model_type cnsn --epochs 60 --batch_size 64 --lr 0.001 --device cuda
```

## Validation Strategy

This project uses the Holdout method for model validation, with 80% of the data used for training and 20% for validation. This provides a simple and effective way to evaluate model performance while maintaining computational efficiency.

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

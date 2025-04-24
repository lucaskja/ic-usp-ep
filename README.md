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
│   └── model_comparison/ # Framework comparison scripts
├── utils/                # Utility functions
└── tests/                # Test cases
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

#### Advanced Training with SGD and Learning Rate Decay

For training with SGD optimizer and learning rate decay (recommended for optimal results):

```bash
# Base MobileNetV2 with SGD and LR decay
cd base_mobilenetv2
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 60 --batch_size 32 --lr 0.001 --device cuda

# MobileNetV2 with Mish
cd stage1_mish
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 60 --batch_size 32 --lr 0.001 --device cuda

# MobileNetV2 with Mish and Triplet Attention
cd stage2_triplet
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 60 --batch_size 32 --lr 0.001 --device cuda

# MobileNetV2 with Mish, Triplet Attention, and CNSN
cd stage3_cnsn
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 60 --batch_size 32 --lr 0.001 --device cuda
```

#### Quick Testing Mode

For quick testing to ensure everything is working correctly:

```bash
# Run with a small subset of data and just 1 epoch
cd base_mobilenetv2
source ../venv/bin/activate && python train.py --data_dir ../datasets/leaf_disease --epochs 1 --batch_size 8 --debug --device cuda
```

Note: The learning rate scheduler is already configured in the default configuration files with:
- SGD optimizer with momentum 0.9 and weight decay 1e-4
- Step learning rate scheduler that reduces the learning rate by a factor of 0.1 every 10 epochs
- You can modify these settings in the respective config files in each model's directory
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
python analyze.py --results_dir experiments/results --output_dir experiments/visualizations
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

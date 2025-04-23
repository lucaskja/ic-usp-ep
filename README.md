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
├── utils/                # Utility functions
├── tests/                # Test cases
└── experiments/          # Experiment results and logs
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

### Dataset Preparation

1. Download a leaf disease dataset (e.g., Plant Village, PlantDoc)
2. Organize it according to the structure in [dataset_structure.md](dataset_structure.md)

## Running the Models Locally

### Training

To train the models with K-Fold cross-validation:

#### Basic Training Commands

```bash
# Base MobileNetV2
python train.py --model base --data_dir datasets/leaf_disease --epochs 50 --batch_size 32 --lr 0.001 --k_folds 5

# MobileNetV2 with Mish
python train.py --model mish --data_dir datasets/leaf_disease --epochs 50 --batch_size 32 --lr 0.001 --k_folds 5

# MobileNetV2 with Mish and Triplet Attention
python train.py --model triplet --data_dir datasets/leaf_disease --epochs 50 --batch_size 32 --lr 0.001 --k_folds 5

# MobileNetV2 with Mish, Triplet Attention, and CNSN
python train.py --model cnsn --data_dir datasets/leaf_disease --epochs 50 --batch_size 32 --lr 0.001 --k_folds 5
```

#### Advanced Training with SGD and Learning Rate Decay

For training with SGD optimizer and learning rate decay (recommended for optimal results):

```bash
# Base MobileNetV2 with SGD and LR decay
python train.py --model base --data_dir datasets/leaf_disease --epochs 60 --batch_size 32 --optimizer sgd --lr 0.001 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.1 --loss cross_entropy --k_folds 5

# MobileNetV2 with Mish
python train.py --model mish --data_dir datasets/leaf_disease --epochs 60 --batch_size 32 --optimizer sgd --lr 0.001 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.1 --loss cross_entropy --k_folds 5

# MobileNetV2 with Mish and Triplet Attention
python train.py --model triplet --data_dir datasets/leaf_disease --epochs 60 --batch_size 32 --optimizer sgd --lr 0.001 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.1 --loss cross_entropy --k_folds 5

# MobileNetV2 with Mish, Triplet Attention, and CNSN
python train.py --model cnsn --data_dir datasets/leaf_disease --epochs 60 --batch_size 32 --optimizer sgd --lr 0.001 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.1 --loss cross_entropy --k_folds 5
```

These commands configure:
- 60 training epochs
- SGD optimizer with initial learning rate of 0.001
- Learning rate decay by a factor of 0.1 every 20 epochs
- Cross-entropy loss function
- 5-fold cross-validation
```

### Evaluation

To evaluate the models on the test set:

```bash
# Base MobileNetV2
python evaluate.py --model base --data_dir datasets/leaf_disease/test --checkpoint checkpoints/base_best.pth

# MobileNetV2 with Mish
python evaluate.py --model mish --data_dir datasets/leaf_disease/test --checkpoint checkpoints/mish_best.pth

# MobileNetV2 with Mish and Triplet Attention
python evaluate.py --model triplet --data_dir datasets/leaf_disease/test --checkpoint checkpoints/triplet_best.pth

# MobileNetV2 with Mish, Triplet Attention, and CNSN
python evaluate.py --model cnsn --data_dir datasets/leaf_disease/test --checkpoint checkpoints/cnsn_best.pth
```

### Visualization and Analysis

To generate visualizations and performance comparisons:

```bash
python analyze.py --results_dir experiments/results --output_dir experiments/visualizations
```

## Cross-Validation

This project uses K-Fold cross-validation for robust model evaluation. For details on the implementation, see [data_validation.md](data_validation.md).

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

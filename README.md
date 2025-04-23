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

## Improvements

### 1. Mish Activation Function

Replaces ReLU6 with Mish activation function to enhance nonlinear characteristics:
- Smoother activation function
- Better gradient propagation
- No upper bound truncation
- Allows negative outputs

### 2. Triplet Attention

Adds a three-branch attention structure for capturing cross-dimension interactions:
- No channel reduction
- Parameter-efficient
- Captures interactions across different dimensions

### 3. CNSN (CrossNorm and SelfNorm)

Integrates CrossNorm and SelfNorm modules:
- CrossNorm: Enlarges training distribution by exchanging channel-wise statistics
- SelfNorm: Bridges train-test distribution gap using an attention mechanism

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

1. Download your leaf disease dataset
2. Organize it in the following structure:
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

## Usage

### Training

To train the models:

```bash
# Base MobileNetV2
python -m mobilenetv2_improvements.base_mobilenetv2.train --data_dir datasets/leaf_disease

# MobileNetV2 with Mish
python -m mobilenetv2_improvements.stage1_mish.train --data_dir datasets/leaf_disease

# MobileNetV2 with Mish and Triplet Attention
python -m mobilenetv2_improvements.stage2_triplet.train --data_dir datasets/leaf_disease

# MobileNetV2 with Mish, Triplet Attention, and CNSN
python -m mobilenetv2_improvements.stage3_cnsn.train --data_dir datasets/leaf_disease
```

### Evaluation

To evaluate the models:

```bash
# Base MobileNetV2
python -m mobilenetv2_improvements.base_mobilenetv2.evaluate --data_dir datasets/leaf_disease --checkpoint path/to/checkpoint.pth

# MobileNetV2 with Mish
python -m mobilenetv2_improvements.stage1_mish.evaluate --data_dir datasets/leaf_disease --checkpoint path/to/checkpoint.pth

# MobileNetV2 with Mish and Triplet Attention
python -m mobilenetv2_improvements.stage2_triplet.evaluate --data_dir datasets/leaf_disease --checkpoint path/to/checkpoint.pth

# MobileNetV2 with Mish, Triplet Attention, and CNSN
python -m mobilenetv2_improvements.stage3_cnsn.evaluate --data_dir datasets/leaf_disease --checkpoint path/to/checkpoint.pth
```

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

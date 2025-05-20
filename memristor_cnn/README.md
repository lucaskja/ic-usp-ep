# MemristorCNN: Memristor-based CNN for Leaf Disease Detection

This project implements a Memristor-based Convolutional Neural Network (mCNN) architecture optimized for leaf disease detection using TTN-MobileNetV2 (Triplet attention, CNSN normalization, and Mish activation).

## Architecture Overview

The MemristorCNN architecture combines several key innovations:

1. **Base Model**: TTN-MobileNetV2 with:
   - Mish activation function replacing ReLU6
   - Triplet Attention for enhanced spatial-channel attention
   - CNSN (CrossNorm and SelfNorm) for improved feature normalization

2. **Memristor Integration**:
   - 2048-cell (128×16 1T1R) memristor arrays
   - 15-level fixed-point conductance states
   - Differential conductance pairs for weight representation
   - 0.2V read voltage and 50ns programming pulses

3. **Hybrid Training Approach**:
   - Phase 1: Ex-situ training (conventional backpropagation)
   - Weight transfer with closed-loop programming
   - Phase 2: In-situ training (threshold-based updates for FC layer only)

## Key Features

- **Energy Efficiency**: ~110x more efficient than GPU-based implementations
- **Latency Reduction**: 3x faster with parallel convolvers
- **Accuracy Target**: >96% on leaf disease classification tasks
- **Error Tolerance**: Hybrid training compensation for device variations

## Directory Structure

```
memristor_cnn/
├── models/                  # Model architecture implementation
│   ├── memristor_crossbar.py  # Memristor crossbar array implementation
│   ├── memristor_cnn.py       # Main model architecture
│   ├── memristor_mapping.py   # Mapping layers to memristor arrays
│   └── memristor_pe.py        # Processing element implementation
├── utils/                   # Utility functions
│   ├── data_utils.py          # Data loading utilities
│   ├── evaluation_utils.py    # Model evaluation utilities
│   ├── memristor_utils.py     # Memristor-specific utilities
│   └── training_utils.py      # Training utilities
├── tests/                   # Test cases
│   └── test_memristor_cnn.py  # Tests for model components
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
└── visualize.py             # Visualization script
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.1.0+
- CUDA (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd memristor_cnn
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scikit-learn tqdm pandas seaborn torchviz graphviz
```

## Usage

### Training

To train the model using the hybrid training approach:

```bash
# Basic training
python train.py --data_dir datasets/leaf_disease --batch_size 100 --ex_situ_epochs 50 --in_situ_epochs 10

# With enhanced data augmentation
python train.py --data_dir datasets/leaf_disease --enhanced_augmentation --batch_size 100

# With GPU acceleration
python train.py --data_dir datasets/leaf_disease --device cuda --batch_size 100

# Skip ex-situ training (if you have a pre-trained model)
python train.py --data_dir datasets/leaf_disease --skip_ex_situ --checkpoint checkpoints/memristor_cnn/model_best.pth
```

### Evaluation

To evaluate a trained model:

```bash
# Basic evaluation
python evaluate.py --data_dir datasets/leaf_disease --checkpoint checkpoints/memristor_cnn/model_best.pth

# With GPU acceleration
python evaluate.py --data_dir datasets/leaf_disease --checkpoint checkpoints/memristor_cnn/model_best.pth --device cuda
```

### Visualization

To visualize model architecture, training history, and performance metrics:

```bash
# Visualize all aspects
python visualize.py --results_dir results/memristor_cnn --visualize_all

# Visualize specific aspects
python visualize.py --results_dir results/memristor_cnn --visualize_model --visualize_metrics
```

## GPU Training on Windows

To train the model on a Windows machine with GPU support:

### Setup for Windows with GPU

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Note: For CUDA 12.1, use:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install other dependencies**:
   ```bash
   pip install numpy matplotlib scikit-learn tqdm pandas seaborn torchviz graphviz
   ```

4. **Verify GPU detection**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
   ```

### Training on Windows GPU

1. **Set PYTHONPATH to the project root**:
   ```bash
   set PYTHONPATH=C:\path\to\your\project
   ```

2. **Run the training script with GPU**:
   ```bash
   python train.py --data_dir datasets/leaf_disease --device cuda --batch_size 64 --ex_situ_epochs 50 --in_situ_epochs 10
   ```

3. **For faster experimentation**, you can start with:
   ```bash
   python train.py --data_dir datasets/leaf_disease --device cuda --batch_size 64 --ex_situ_epochs 10 --in_situ_epochs 5
   ```

### GPU Requirements

- NVIDIA GPU with CUDA support
- At least 4GB VRAM (8GB+ recommended for larger batch sizes)
- Updated NVIDIA drivers
- Compatible CUDA version (11.8 or 12.1 recommended)

## Memristor Mapping

The model maps neural network layers to memristor arrays as follows:

1. **Convolutional Layers**:
   - First convolutional layer (C1): 3×3 kernels, 8 channels → PE1
   - Third convolutional layer (C3): 3×3×8 kernels, 12 channels → PE1, PE3

2. **Fully Connected Layers**:
   - Input size: 192, Output size: 10 → PE5, PE7

## Performance Metrics

Expected performance metrics:

- **Accuracy**: >96% on leaf disease classification
- **Energy Efficiency**: 110x compared to GPU implementation
- **Latency Reduction**: 3x with parallel convolvers
- **Memory Footprint**: Significantly reduced compared to conventional models

## Testing

Run the test suite to verify the implementation:

```bash
python -m unittest discover -s tests
```

## References

1. MobileNetV2: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
2. Triplet Attention: "Rotate to Attend: Convolutional Triplet Attention Module"
3. CNSN: "CrossNorm and SelfNorm for Generalization under Distribution Shifts"
4. Mish: "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
5. Memristor CNN: "Fully hardware-implemented memristor convolutional neural network"

## License

[MIT License](LICENSE)
## Troubleshooting

### CUDA Device Mismatch Error

If you encounter the following error:
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

This error occurs when tensors on different devices (CPU and GPU) are used together. The fix has been implemented in the latest version, which ensures all model components are moved to the same device.

If you're still experiencing this issue:

1. Make sure you're using the latest version of the code
2. Verify that your GPU has enough memory for the model
3. Try reducing the batch size with `--batch_size 32` or even smaller
4. Check that your CUDA installation is working correctly with:
   ```python
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
   ```

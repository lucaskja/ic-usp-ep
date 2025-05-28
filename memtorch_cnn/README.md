# MemTorch-based CNN for Leaf Disease Classification

This project implements a MemTorch-based Convolutional Neural Network (CNN) for leaf disease classification. The implementation leverages memristor crossbar arrays for efficient computation, providing significant improvements in energy efficiency and latency compared to conventional CNN implementations.

## Features

- **MemTorch Integration**: Uses MemTorch for accurate memristor device modeling
- **Hybrid Training Approach**: Ex-situ training followed by in-situ fine-tuning
- **Hardware-Aware Training**: Accounts for memristor constraints during training
- **Energy Efficiency Analysis**: Compares energy consumption with conventional CNNs
- **Latency Analysis**: Compares inference latency with conventional CNNs
- **Non-Ideality Modeling**: Simulates device variations and state drift

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ic-usp-ep
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r memtorch_cnn/requirements.txt
```

## Usage

### Training

To train the MemTorch-based CNN model:

```bash
python memtorch_cnn/train.py --data_dir datasets/leaf_disease --device cuda --batch_size 32 --ex_situ_epochs 50 --in_situ_epochs 10
```

#### Training Options

- `--data_dir`: Path to the dataset directory
- `--enhanced_augmentation`: Use enhanced data augmentation
- `--width_mult`: Width multiplier for the network (default: 0.75)
- `--tile_shape`: Shape of memristor crossbar tiles (default: 128 128)
- `--adc_resolution`: ADC resolution in bits (default: 8)
- `--dac_resolution`: DAC resolution in bits (default: 8)
- `--max_input_voltage`: Maximum input voltage (default: 0.3)
- `--batch_size`: Batch size for training (default: 32)
- `--ex_situ_epochs`: Number of ex-situ training epochs (default: 50)
- `--in_situ_epochs`: Number of in-situ training epochs (default: 10)
- `--lr`: Initial learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--threshold`: Threshold for in-situ weight updates (default: 0.1)
- `--device`: Device to use (cuda or cpu)
- `--checkpoint_dir`: Directory to save checkpoints
- `--results_dir`: Directory to save results
- `--skip_ex_situ`: Skip ex-situ training phase
- `--skip_in_situ`: Skip in-situ training phase
- `--resume`: Path to checkpoint to resume from
- `--debug`: Enable debug mode (reduced dataset size)

### Evaluation

To evaluate a trained model:

```bash
python memtorch_cnn/evaluate.py --data_dir datasets/leaf_disease --checkpoint checkpoints/memtorch_cnn/model_best_in_situ.pth --device cuda
```

#### Evaluation Options

- `--data_dir`: Path to the dataset directory
- `--width_mult`: Width multiplier for the network (default: 0.75)
- `--checkpoint`: Path to the checkpoint to evaluate
- `--device`: Device to use (cuda or cpu)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--results_dir`: Directory to save results

## Model Architecture

The MemTorch-based CNN uses a MobileNetV2-like architecture with the following components:

1. **First Conv Layer**: Standard convolutional layer (kept as digital)
2. **Inverted Residual Blocks**: Converted to memristive layers
3. **Last Conv Layer**: Converted to memristive layer
4. **Global Average Pooling**: Standard pooling layer (kept as digital)
5. **Classifier**: Fully connected layer converted to memristive layer

### Memristor Configuration

- **Device Model**: LinearIonDrift
- **Crossbar Array Size**: 128×128
- **Resistance Range**: 100Ω (ON) to 16kΩ (OFF)
- **ADC/DAC Resolution**: 8 bits
- **Weight Quantization**: 4 bits (16 levels)

## Hybrid Training Approach

The training process consists of two phases:

1. **Ex-situ Training**:
   - Conventional training on GPU/CPU
   - All layers are trainable
   - Standard backpropagation

2. **Weight Transfer**:
   - Convert model to memristive
   - Apply weight quantization
   - Apply non-idealities

3. **In-situ Training**:
   - Freeze convolutional layers
   - Only update FC layer weights
   - Threshold-based updates
   - Hardware-aware training

## Performance Analysis

The MemTorch-based CNN provides significant improvements over conventional CNNs:

- **Energy Efficiency**: 10-100× improvement
- **Latency**: 2-5× improvement
- **Model Size**: Similar to conventional CNN

## Testing

To run the tests:

```bash
python -m unittest discover memtorch_cnn/tests
```

## Comparison with Custom Memristor Implementation

This MemTorch-based implementation offers several advantages over the custom memristor implementation:

1. **More Accurate Device Models**: MemTorch includes realistic memristor models that capture actual device physics
2. **Built-in Non-idealities**: Simulates device-to-device variations, state drift, and other non-ideal characteristics
3. **Seamless PyTorch Integration**: Extends PyTorch's Module class for easy model conversion
4. **Hardware-Aware Training**: Accounts for memristor constraints during training
5. **Energy and Performance Analysis**: Built-in tools for analyzing energy consumption and performance

## License

[MIT License](LICENSE)

## Acknowledgments

- MemTorch: https://github.com/coreylammie/MemTorch
- PyTorch: https://pytorch.org/
- MobileNetV2: https://arxiv.org/abs/1801.04381

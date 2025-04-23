# MobileNetV2 Framework Comparison Scripts

This directory contains scripts for comparing MobileNetV2 implementations between PyTorch and TensorFlow.

## Scripts

- `model_comparison.py`: Basic comparison of PyTorch MobileNetV2 with theoretical TensorFlow values
- `model_comparison_detailed.py`: Detailed analysis of PyTorch MobileNetV2 architecture
- `compare_models.py`: Direct comparison between PyTorch and TensorFlow MobileNetV2 implementations

## Usage

### Basic Comparison (PyTorch only)

```bash
python model_comparison.py
```

### Detailed PyTorch Analysis

```bash
python model_comparison_detailed.py
```

### Direct Framework Comparison

Requires Python 3.12 with both PyTorch and TensorFlow installed:

```bash
python compare_models.py
```

## Environment Setup for TensorFlow Comparison

To run the direct comparison with TensorFlow:

1. Create a Python 3.12 virtual environment:
   ```bash
   python3.12 -m venv tf_venv
   ```

2. Activate the environment:
   ```bash
   source tf_venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install tensorflow torch torchvision
   ```

4. Run the comparison script:
   ```bash
   python compare_models.py
   ```

## Results

The comparison results are documented in the `docs/model_comparison/` directory.

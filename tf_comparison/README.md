# MobileNetV2 Framework Comparison

This directory contains code and results for comparing MobileNetV2 implementations between PyTorch and TensorFlow.

## Environment Setup

- Python 3.12
- TensorFlow 2.19.0
- PyTorch 2.7.0

## Files

- `compare_models.py`: Script to directly compare PyTorch and TensorFlow MobileNetV2 implementations
- `detailed_comparison.md`: Comprehensive analysis of the comparison results

## Key Findings

1. **Parameter Count**:
   - PyTorch MobileNetV2: 3,504,872 parameters
   - TensorFlow MobileNetV2: 3,538,984 parameters
   - Difference: 34,112 parameters (0.96%)

2. **Model Size**:
   - Both implementations: 13.50 MB

3. **Layer Distribution**:
   - PyTorch: 52 Conv2d, 52 BatchNorm2d, 1 Linear
   - TensorFlow: 35 Conv2D, 17 DepthwiseConv2D, 52 BatchNormalization, 1 Dense

4. **Inference Performance**:
   - PyTorch: 49.62 ms per inference
   - TensorFlow: 59.58 ms per inference
   - PyTorch is approximately 20% faster on this hardware

## Running the Comparison

```bash
# Activate the virtual environment
source tf_venv/bin/activate

# Run the comparison script
python compare_models.py
```

## Conclusion

The PyTorch and TensorFlow implementations of MobileNetV2 are functionally equivalent with minor differences in parameter count (0.96%) and implementation details. The main architectural difference is in how batch normalization is implemented and how depthwise convolutions are handled.

For research purposes, both implementations can be considered equivalent, with PyTorch showing a slight performance advantage in inference speed on this particular hardware.

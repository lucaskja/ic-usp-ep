# MobileNetV2 Framework Comparison

This directory contains documentation and analysis of the comparison between PyTorch and TensorFlow implementations of MobileNetV2.

## Files

- `detailed_comparison.md`: Comprehensive analysis of the comparison results from direct testing
- `model_comparison_results.md`: Initial comparison results and implications for the research project

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

## Conclusion

The PyTorch and TensorFlow implementations of MobileNetV2 are functionally equivalent with minor differences in parameter count (0.96%) and implementation details. The main architectural difference is in how batch normalization is implemented and how depthwise convolutions are handled.

For research purposes, both implementations can be considered equivalent, with PyTorch showing a slight performance advantage in inference speed on this particular hardware.

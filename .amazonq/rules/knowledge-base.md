# Knowledge Base for MobileNetV2 Improvements Project

## Base Model: MobileNetV2
- Source: PyTorch's torchvision.models.mobilenet_v2
- Architecture features:
  - Inverted residual structure
  - Lightweight deepwise separable convolutions
  - Initial processing with spatial features through individual kernels
  - 1x1 pointwise convolutions for channel combination
  - Uses ReLU6 activation function

## Improvement 1: Mish Activation Function
Replacing ReLU6 with Mish to enhance nonlinear characteristics.

### Mish Function Characteristics:
- Smoother activation function
- Allows negative outputs unlike ReLU6
- Better gradient propagation
- No truncation at upper bound (unlike ReLU6's limit at 6)
- Maintains robust gradient flow during training
- Helps prevent gradient vanishing

### Implementation Details:
- Replace all ReLU6 activations in MobileNetV2
- No architectural changes needed
- Requires custom Mish implementation in PyTorch

## Improvement 2: Triplet Attention
A three-branch structure for capturing cross-dimension interactions.

### Key Features:
- No channel reduction
- Parameter-efficient
- Captures interactions across different dimensions

### Architecture:
1. Branch 1 (Channel-Height Interaction):
   - Input rotation 90° anti-clockwise along H axis
   - Z-pool operation (max and avg pool concatenation)
   - Conv2D (k×k kernel)
   - Batch Normalization
   - Sigmoid activation

2. Branch 2 (Channel-Width Interaction):
   - Input rotation 90° anti-clockwise along W axis
   - Z-pool operation
   - Conv2D (k×k kernel)
   - Batch Normalization
   - Sigmoid activation

3. Branch 3 (Spatial Attention):
   - Z-pool operation
   - Conv2D (k×k kernel)
   - Batch Normalization
   - Sigmoid activation

### Integration:
- Output = Average(Branch1, Branch2, Branch3)
- Add after each inverted residual block in MobileNetV2

## Improvement 3: CNSN (CrossNorm and SelfNorm)

### CrossNorm:
- Purpose: Enlarge training distribution
- Operation: Exchanges channel-wise mean and variance between feature maps
- Variants:
  1. 1-instance mode: Exchange between channels within same instance
  2. 2-instance mode: Exchange between corresponding channels of different instances
  3. Crop mode: Apply to specific spatial regions

### SelfNorm:
- Purpose: Bridge train-test distribution gap
- Operation: Recalibrates statistics using attention mechanism
- Components:
  - Two FC networks for attention functions f and g
  - Processes each channel independently
  - No dimensionality reduction

### Integration:
- CrossNorm works only during training
- SelfNorm works during both training and testing
- Complementary operation:
  - CrossNorm: Expands feature distribution
  - SelfNorm: Reduces feature distribution variations

## Implementation Order:
1. Replace ReLU6 with Mish
2. Add Triplet Attention after inverted residual blocks
3. Integrate CNSN modules

## Testing Requirements:
1. Base MobileNetV2 performance baseline
2. Individual improvement validations
3. Cumulative improvement measurements
4. Ablation studies for each component

## Metrics to Track:
- Model accuracy
- Parameter count
- Computational overhead (FLOPs)
- Training time
- Memory usage
- Inference speed

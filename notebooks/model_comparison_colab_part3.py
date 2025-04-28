# Helper class for InvertedResidual with Triplet Attention
class InvertedResidualWithTripletAttention(nn.Module):
    """
    Inverted Residual block with Triplet Attention.
    Adds Triplet Attention after the depthwise convolution.
    """
    def __init__(self, inverted_residual_block, kernel_size=7):
        """
        Initialize wrapper for inverted residual block.
        
        Args:
            inverted_residual_block: Original inverted residual block
            kernel_size (int): Kernel size for Triplet Attention
        """
        super(InvertedResidualWithTripletAttention, self).__init__()
        self.block = inverted_residual_block
        self.attention = TripletAttention(kernel_size=kernel_size)
        self.use_res_connect = self.block.use_res_connect
        
        # Extract the layers from the original block
        if hasattr(self.block, 'conv'):
            # For blocks with residual connection
            layers = list(self.block.conv)
            
            # Find the depthwise convolution layer
            dw_conv_idx = None
            for i, layer in enumerate(layers):
                if isinstance(layer, nn.Conv2d) and layer.groups > 1:
                    dw_conv_idx = i
                    break
            
            if dw_conv_idx is not None:
                # Split the layers before and after depthwise conv
                self.layers_before_dw = nn.Sequential(*layers[:dw_conv_idx+1])
                self.layers_after_dw = nn.Sequential(*layers[dw_conv_idx+1:])
            else:
                # Fallback if depthwise conv not found
                self.layers_before_dw = self.block.conv
                self.layers_after_dw = nn.Identity()
        else:
            # Fallback for other types of blocks
            self.layers_before_dw = self.block
            self.layers_after_dw = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.use_res_connect:
            # Apply layers before depthwise conv
            out = self.layers_before_dw(x)
            
            # Apply Triplet Attention
            out = self.attention(out)
            
            # Apply layers after depthwise conv
            out = self.layers_after_dw(out)
            
            # Add residual connection
            return x + out
        else:
            # Apply layers before depthwise conv
            out = self.layers_before_dw(x)
            
            # Apply Triplet Attention
            out = self.attention(out)
            
            # Apply layers after depthwise conv
            out = self.layers_after_dw(out)
            
            return out

# Add Triplet Attention to MobileNetV2
def add_triplet_attention_to_mobilenetv2(model, kernel_size=7):
    """
    Add Triplet Attention to MobileNetV2 model.
    
    Args:
        model (nn.Module): MobileNetV2 model
        kernel_size (int): Kernel size for Triplet Attention
        
    Returns:
        nn.Module: MobileNetV2 model with Triplet Attention
    """
    # Add Triplet Attention after each inverted residual block
    for i, layer in enumerate(model.features):
        if hasattr(layer, 'conv'):  # Check if it's an inverted residual block
            model.features[i] = InvertedResidualWithTripletAttention(layer, kernel_size=kernel_size)
    
    return model

# MobileNetV2 with Mish and Triplet Attention
def create_mobilenetv2_triplet(num_classes, pretrained=True, triplet_attention_kernel_size=7, width_mult=0.75):
    """
    Create a MobileNetV2 model with Mish activation and Triplet Attention.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        triplet_attention_kernel_size (int): Kernel size for Triplet Attention
        width_mult (float): Width multiplier for the network (default: 0.75)
        
    Returns:
        nn.Module: MobileNetV2 model with Mish and Triplet Attention
    """
    model = create_mobilenetv2_mish(num_classes, pretrained, width_mult)
    model = add_triplet_attention_to_mobilenetv2(model, kernel_size=triplet_attention_kernel_size)
    return model

# Helper function to get output channels
def get_output_channels(layer):
    """
    Get the number of output channels from a layer.
    
    Args:
        layer (nn.Module): Layer to analyze
        
    Returns:
        int: Number of output channels
    """
    # For inverted residual blocks with residual connection
    if hasattr(layer, 'conv'):
        # Navigate through the conv sequential to find the last conv layer
        for m in reversed(list(layer.conv.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
    
    # For other blocks, try to find the last layer with out_channels
    for m in reversed(list(layer.modules())):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    
    # Fallback: use a default value
    return 32

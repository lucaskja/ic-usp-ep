# Helper class for InvertedResidual with Triplet Attention and CNSN
class InvertedResidualWithTripletAttentionAndCNSN(nn.Module):
    """
    Inverted Residual block with Triplet Attention and CNSN.
    Adds Triplet Attention and CNSN after the depthwise convolution.
    """
    def __init__(self, inverted_residual_block, num_channels, kernel_size=7, p=0.5):
        """
        Initialize wrapper for inverted residual block.
        
        Args:
            inverted_residual_block: Original inverted residual block
            num_channels (int): Number of output channels
            kernel_size (int): Kernel size for Triplet Attention
            p (float): Probability of applying CrossNorm during training
        """
        super(InvertedResidualWithTripletAttentionAndCNSN, self).__init__()
        self.block = inverted_residual_block
        self.attention = TripletAttention(kernel_size=kernel_size)
        self.cnsn = CNSN(num_channels, p=p)
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
            
            # Apply CNSN
            out = self.cnsn(out)
            
            # Apply layers after depthwise conv
            out = self.layers_after_dw(out)
            
            # Add residual connection
            return x + out
        else:
            # Apply layers before depthwise conv
            out = self.layers_before_dw(x)
            
            # Apply Triplet Attention
            out = self.attention(out)
            
            # Apply CNSN
            out = self.cnsn(out)
            
            # Apply layers after depthwise conv
            out = self.layers_after_dw(out)
            
            return out

# Add Triplet Attention and CNSN to MobileNetV2
def add_triplet_attention_and_cnsn_to_mobilenetv2(model, kernel_size=7, p=0.5):
    """
    Add Triplet Attention and CNSN to MobileNetV2 model.
    
    Args:
        model (nn.Module): MobileNetV2 model
        kernel_size (int): Kernel size for Triplet Attention
        p (float): Probability of applying CrossNorm during training
        
    Returns:
        nn.Module: MobileNetV2 model with Triplet Attention and CNSN
    """
    # Add Triplet Attention and CNSN after each inverted residual block
    for i, layer in enumerate(model.features):
        if hasattr(layer, 'conv'):  # Check if it's an inverted residual block
            # Get number of output channels
            num_channels = get_output_channels(layer)
            model.features[i] = InvertedResidualWithTripletAttentionAndCNSN(
                layer, 
                num_channels, 
                kernel_size=kernel_size,
                p=p
            )
    
    return model

# MobileNetV2 with Mish, Triplet Attention, and CNSN
def create_mobilenetv2_cnsn(num_classes, pretrained=True, kernel_size=7, p=0.5, width_mult=0.75):
    """
    Create a MobileNetV2 model with Mish activation, Triplet Attention, and CNSN.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        kernel_size (int): Kernel size for Triplet Attention
        p (float): Probability of applying CrossNorm during training
        width_mult (float): Width multiplier for the network (default: 0.75)
        
    Returns:
        nn.Module: MobileNetV2 model with Mish, Triplet Attention, and CNSN
    """
    model = create_mobilenetv2_mish(num_classes, pretrained, width_mult)
    model = add_triplet_attention_and_cnsn_to_mobilenetv2(model, kernel_size=kernel_size, p=p)
    return model

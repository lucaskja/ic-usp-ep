#######################
# Main Execution
#######################

# Upload dataset
def upload_dataset():
    """Upload dataset from local machine."""
    print("Please upload your dataset as a zip file.")
    uploaded = files.upload()
    
    if len(uploaded) == 0:
        print("No file uploaded. Please try again.")
        return None
    
    # Get the filename of the uploaded file
    filename = list(uploaded.keys())[0]
    
    # Extract the zip file
    !mkdir -p /content/datasets
    !unzip -q -o "{filename}" -d /content/datasets
    
    # Find the dataset directory
    import glob
    dataset_dirs = glob.glob('/content/datasets/*/')
    
    if len(dataset_dirs) == 0:
        print("No directories found in the uploaded zip file.")
        return None
    
    # Use the first directory as the dataset directory
    data_dir = dataset_dirs[0]
    print(f"Dataset extracted to: {data_dir}")
    
    return data_dir

# Main execution
if __name__ == "__main__":
    # Option 1: Upload dataset
    data_dir = upload_dataset()
    
    # Option 2: Use a sample dataset (uncomment to use)
    # !wget -q https://example.com/leaf_disease_dataset.zip
    # !unzip -q leaf_disease_dataset.zip -d /content/datasets
    # data_dir = '/content/datasets/leaf_disease'
    
    if data_dir:
        # Run model comparison
        run_model_comparison(
            data_dir=data_dir,
            enhanced_augmentation=True,  # Set to False for standard augmentation
            epochs=30,                   # Adjust as needed
            batch_size=32,               # Adjust based on GPU memory
            lr=0.001,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    else:
        print("Please provide a dataset to continue.")
# SelfNorm
class SelfNorm(nn.Module):
    """
    SelfNorm module for bridging train-test distribution gap.
    
    Learns channel statistics recalibration using FC layers.
    Active during both training and inference.
    """
    def __init__(self, channels):
        """
        Initialize SelfNorm module.
        
        Args:
            channels (int): Number of input channels
        """
        super(SelfNorm, self).__init__()
        
        # FC layers for mean weight generation
        self.fc_mean = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # FC layers for std weight generation
        self.fc_std = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        B, C, H, W = x.size()
        
        # Calculate mean and std across spatial dimensions
        mean = x.mean(dim=[2, 3])  # [B, C]
        std = torch.sqrt(x.var(dim=[2, 3]) + 1e-5)  # [B, C]
        
        # Process each channel independently
        mean_weights = []
        std_weights = []
        
        for c in range(C):
            # Get statistics for this channel
            channel_stats = torch.stack([mean[:, c], std[:, c]], dim=1)  # [B, 2]
            
            # Generate weights for this channel
            mean_weight = self.fc_mean(channel_stats)  # [B, 1]
            std_weight = self.fc_std(channel_stats)  # [B, 1]
            
            mean_weights.append(mean_weight)
            std_weights.append(std_weight)
        
        # Stack weights for all channels
        mean_weights = torch.stack(mean_weights, dim=1)  # [B, C, 1]
        std_weights = torch.stack(std_weights, dim=1)  # [B, C, 1]
        
        # Reshape for broadcasting
        mean = mean.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        std = std.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        mean_weights = mean_weights.unsqueeze(-1)  # [B, C, 1, 1]
        std_weights = std_weights.unsqueeze(-1)  # [B, C, 1, 1]
        
        # Apply recalibration
        x_norm = (x - mean) / std
        x_selfnorm = x_norm * std_weights + mean * mean_weights
        
        return x_selfnorm

# CNSN
class CNSN(nn.Module):
    """
    CNSN (CrossNorm and SelfNorm) module.
    
    Combines CrossNorm for training distribution expansion and
    SelfNorm for bridging train-test distribution gap.
    """
    def __init__(self, channels, p=0.5):
        """
        Initialize CNSN module.
        
        Args:
            channels (int): Number of input channels
            p (float): Probability of applying CrossNorm during training
        """
        super(CNSN, self).__init__()
        self.crossnorm = CrossNorm(p=p)
        self.selfnorm = SelfNorm(channels)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Apply CrossNorm (only during training with probability p)
        x = self.crossnorm(x)
        
        # Apply SelfNorm (both training and testing)
        x = self.selfnorm(x)
        
        return x

#######################
# MobileNetV2 Models
#######################

# Base MobileNetV2
def create_mobilenetv2(num_classes, pretrained=True, width_mult=0.75):
    """
    Create a MobileNetV2 model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        width_mult (float): Width multiplier for the network (default: 0.75)
        
    Returns:
        nn.Module: MobileNetV2 model
    """
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    
    if pretrained and width_mult == 1.0:
        # Pretrained weights are only available for width_mult=1.0
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        # For custom width_mult, we can't use pretrained weights
        model = mobilenet_v2(weights=None, width_mult=width_mult)
        if pretrained and width_mult != 1.0:
            print(f"Warning: Pretrained weights are only available for width_mult=1.0. Using random initialization for width_mult={width_mult}.")
    
    # Modify the classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model

# MobileNetV2 with Mish
def create_mobilenetv2_mish(num_classes, pretrained=True, width_mult=0.75):
    """
    Create a MobileNetV2 model with Mish activation.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        width_mult (float): Width multiplier for the network (default: 0.75)
        
    Returns:
        nn.Module: MobileNetV2 model with Mish activation
    """
    model = create_mobilenetv2(num_classes, pretrained, width_mult)
    model = replace_relu_with_mish(model)
    return model
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
#######################
# Utility Functions
#######################

# Get model size in MB
def get_model_size(model):
    """
    Calculate the size of a model in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Size of the model in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_in_mb = (param_size + buffer_size) / 1024**2
    return size_in_mb

# Count parameters
def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Average meter for tracking metrics
class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#######################
# Data Loading
#######################

def load_dataset(data_dir, img_size=224, batch_size=32, num_workers=2):
    """
    Load and prepare dataset for training and evaluation.
    
    Args:
        data_dir (str): Path to the dataset directory
        img_size (int): Input image size
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    try:
        full_dataset = ImageFolder(root=data_dir, transform=train_transform)
        
        # Get number of classes
        num_classes = len(full_dataset.classes)
        print(f"Dataset loaded with {num_classes} classes")
        
        # Split dataset into train, validation, and test sets
        dataset_size = len(full_dataset)
        test_size = int(0.1 * dataset_size)
        val_size = int(0.2 * (dataset_size - test_size))
        train_size = dataset_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create validation and test datasets with appropriate transforms
        val_dataset_with_transform = ImageFolder(root=data_dir, transform=val_transform)
        val_dataset.dataset = val_dataset_with_transform
        val_dataset.indices = val_dataset.indices
        
        test_dataset_with_transform = ImageFolder(root=data_dir, transform=val_transform)
        test_dataset.dataset = test_dataset_with_transform
        test_dataset.indices = test_dataset.indices
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")
        
        return train_loader, val_loader, test_loader, num_classes
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please upload a dataset with the following structure:")
        print("data_dir/")
        print("├── class1/")
        print("│   ├── img001.jpg")
        print("│   └── ...")
        print("├── class2/")
        print("│   ├── img001.jpg")
        print("│   └── ...")
        print("└── ...")
        raise
def load_enhanced_dataset(data_dir, img_size=224, batch_size=32, num_workers=2):
    """
    Load and prepare dataset with enhanced augmentation for training and evaluation.
    
    Args:
        data_dir (str): Path to the dataset directory
        img_size (int): Input image size
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Define enhanced transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    try:
        full_dataset = ImageFolder(root=data_dir, transform=train_transform)
        
        # Get number of classes
        num_classes = len(full_dataset.classes)
        print(f"Dataset loaded with {num_classes} classes")
        
        # Split dataset into train, validation, and test sets
        dataset_size = len(full_dataset)
        test_size = int(0.1 * dataset_size)
        val_size = int(0.2 * (dataset_size - test_size))
        train_size = dataset_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create validation and test datasets with appropriate transforms
        val_dataset_with_transform = ImageFolder(root=data_dir, transform=val_transform)
        val_dataset.dataset = val_dataset_with_transform
        val_dataset.indices = val_dataset.indices
        
        test_dataset_with_transform = ImageFolder(root=data_dir, transform=val_transform)
        test_dataset.dataset = test_dataset_with_transform
        test_dataset.indices = test_dataset.indices
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")
        print("Using enhanced data augmentation")
        
        return train_loader, val_loader, test_loader, num_classes
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please upload a dataset with the following structure:")
        print("data_dir/")
        print("├── class1/")
        print("│   ├── img001.jpg")
        print("│   └── ...")
        print("├── class2/")
        print("│   ├── img001.jpg")
        print("│   └── ...")
        print("└── ...")
        raise

#######################
# Training and Evaluation
#######################

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=0, total_epochs=0):
    """
    Train model for one epoch with progress bar.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use
        epoch (int): Current epoch number
        total_epochs (int): Total number of epochs
        
    Returns:
        dict: Training metrics
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.5f')
    
    # Switch to train mode
    model.train()
    
    # Create progress bar
    desc = f"Epoch {epoch+1}/{total_epochs}" if total_epochs > 0 else "Training"
    
    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader, desc=desc)):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        images = images.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(images)
        loss = criterion(output, target)
        
        # Measure accuracy and record loss
        _, predicted = output.max(1)
        correct = predicted.eq(target).sum().item()
        acc = 100.0 * correct / target.size(0)
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc, images.size(0))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    print(f"Train Epoch: {epoch+1}/{total_epochs} Loss: {losses.avg:.4f} Acc@1: {top1.avg:.5f}%")
    
    return {'loss': losses.avg, 'acc1': top1.avg}
def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set with progress bar.
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        
    Returns:
        dict: Validation metrics
    """
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.5f')
    
    # Switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        for images, target in tqdm(val_loader, desc="Validating"):
            # Move data to device
            images = images.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            _, predicted = output.max(1)
            correct = predicted.eq(target).sum().item()
            acc = 100.0 * correct / target.size(0)
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))
    
    print(f"Validation: Loss: {losses.avg:.4f} Acc@1: {top1.avg:.5f}%")
    
    return {'loss': losses.avg, 'acc1': top1.avg}

def evaluate(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Measure inference time
    inference_start = time.time()
    
    with torch.no_grad():
        for images, target in tqdm(test_loader, desc="Testing"):
            # Move data to device
            images = images.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(images)
            probs = torch.nn.functional.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate inference time per sample
    inference_time = (time.time() - inference_start) / len(test_loader.dataset)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate accuracy
    accuracy = 100.0 * np.mean(all_preds == all_targets)
    
    # Calculate precision, recall, and F1 score
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted'
    )
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'inference_time_ms': inference_time * 1000,  # Convert to milliseconds
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }
    
    print(f"Test accuracy: {accuracy:.5f}%")
    print(f"Precision: {precision*100:.5f}%, Recall: {recall*100:.5f}%, F1: {f1*100:.5f}%")
    print(f"Inference time: {inference_time*1000:.3f} ms per sample")
    
    return results

def plot_confusion_matrix(cm, classes, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        classes (list): Class names
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size
    """
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
#######################
# Model Comparison
#######################

def train_and_evaluate_model(model_name, model_creator, num_classes, train_loader, val_loader, test_loader, 
                            epochs, lr, device, checkpoint_dir, output_dir, width_mult=None):
    """
    Train and evaluate a model.
    
    Args:
        model_name (str): Name of the model
        model_creator (function): Function to create the model
        num_classes (int): Number of classes
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        epochs (int): Number of epochs to train
        lr (float): Learning rate
        device (torch.device): Device to use
        checkpoint_dir (str): Directory to save checkpoints
        output_dir (str): Directory to save results
        width_mult (float, optional): Width multiplier for the model
        
    Returns:
        dict: Dictionary with model metrics
    """
    start_time = time.time()
    
    # Create model
    if width_mult is not None:
        model = model_creator(num_classes=num_classes, pretrained=False, width_mult=width_mult)
    else:
        model = model_creator(num_classes=num_classes, pretrained=False)
    
    model = model.to(device)
    
    # Get model info
    model_size_mb = get_model_size(model)
    num_params = count_parameters(model)
    
    print(f"\n{'='*80}\nTraining {model_name}\n{'='*80}")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Number of parameters: {num_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    # Initialize tracking variables
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    
    # Create model directory
    model_checkpoint_dir = os.path.join(checkpoint_dir, model_name.replace(" ", "_").lower())
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            epochs
        )
        
        # Evaluate on validation set
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")
        
        # Store results
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['acc1'])
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['acc1'])
        
        # Check if this is the best model so far
        is_best = val_metrics['acc1'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['acc1']
            best_epoch = epoch
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(model_checkpoint_dir, 'best.pth'))
            print(f"New best model saved with validation accuracy: {best_val_acc:.5f}%")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds. Best validation accuracy: {best_val_acc:.5f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_training_curves.png"))
    plt.show()
    
    # Load best model for evaluation
    best_model_path = os.path.join(model_checkpoint_dir, 'best.pth')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    
    # Evaluate on test set
    print(f"\nEvaluating {model_name} on test set...")
    test_results = evaluate(model, test_loader, device)
    
    # Plot confusion matrix
    if hasattr(test_loader.dataset, 'classes'):
        classes = test_loader.dataset.classes
    elif hasattr(test_loader.dataset, 'dataset') and hasattr(test_loader.dataset.dataset, 'classes'):
        classes = test_loader.dataset.dataset.classes
    else:
        classes = [str(i) for i in range(num_classes)]
    
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        classes,
        save_path=os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    )
    
    # Compile results
    results = {
        'model_name': model_name,
        'width_mult': width_mult if width_mult is not None else 1.0,
        'model_size_mb': model_size_mb,
        'num_params': num_params,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_results['accuracy'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'test_f1': test_results['f1'],
        'training_time': training_time,
        'inference_time_ms': test_results['inference_time_ms'],
        'checkpoint_path': best_model_path
    }
    
    return results
def run_model_comparison(data_dir, enhanced_augmentation=False, epochs=30, batch_size=32, lr=0.001, device=None):
    """
    Run model comparison for different MobileNetV2 variants.
    
    Args:
        data_dir (str): Path to the dataset directory
        enhanced_augmentation (bool): Whether to use enhanced data augmentation
        epochs (int): Number of epochs to train
        batch_size (int): Batch size
        lr (float): Learning rate
        device (torch.device): Device to use
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    output_dir = '/content/experiments/comparison'
    checkpoint_dir = '/content/checkpoints/comparison'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load dataset
    if enhanced_augmentation:
        train_loader, val_loader, test_loader, num_classes = load_enhanced_dataset(
            data_dir,
            img_size=224,
            batch_size=batch_size,
            num_workers=2
        )
    else:
        train_loader, val_loader, test_loader, num_classes = load_dataset(
            data_dir,
            img_size=224,
            batch_size=batch_size,
            num_workers=2
        )
    
    # Define models to compare
    models_to_compare = [
        {
            'name': 'Base MobileNetV2 (width_mult=1.0)',
            'creator': create_mobilenetv2,
            'width_mult': 1.0
        },
        {
            'name': 'Base MobileNetV2 (width_mult=0.75)',
            'creator': create_mobilenetv2,
            'width_mult': 0.75
        },
        {
            'name': 'MobileNetV2 with Mish (width_mult=0.75)',
            'creator': create_mobilenetv2_mish,
            'width_mult': 0.75
        },
        {
            'name': 'MobileNetV2 with Mish and Triplet Attention (width_mult=0.75)',
            'creator': create_mobilenetv2_triplet,
            'width_mult': 0.75
        },
        {
            'name': 'MobileNetV2 with Mish, Triplet Attention, and CNSN (width_mult=0.75)',
            'creator': create_mobilenetv2_cnsn,
            'width_mult': 0.75
        }
    ]
    
    # Train and evaluate each model
    results = []
    for model_config in models_to_compare:
        model_results = train_and_evaluate_model(
            model_name=model_config['name'],
            model_creator=model_config['creator'],
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            width_mult=model_config['width_mult']
        )
        results.append(model_results)
    
    # Print results summary
    print("\n" + "="*120)
    print("COMPARISON RESULTS SUMMARY")
    print("="*120)
    print(f"{'Model':<50} | {'Size (MB)':<10} | {'Params':<12} | {'Val Acc (%)':<12} | {'Test Acc (%)':<12} | {'Inf Time (ms)':<14}")
    print("-"*120)
    
    for result in results:
        print(
            f"{result['model_name']:<50} | "
            f"{result['model_size_mb']:<10.2f} | "
            f"{result['num_params']:<12,} | "
            f"{result['best_val_acc']:<12.5f} | "
            f"{result['test_acc']:<12.5f} | "
            f"{result['inference_time_ms']:<14.3f}"
        )
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, 'model_comparison_results.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'model_name', 'width_mult', 'model_size_mb', 'num_params', 
            'best_val_acc', 'best_epoch', 'test_acc', 'test_precision', 
            'test_recall', 'test_f1', 'training_time', 'inference_time_ms', 
            'checkpoint_path'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Detailed comparison results saved to {csv_path}")
    
    # Plot comparison bar chart
    plt.figure(figsize=(15, 10))
    
    # Models
    models = [r['model_name'] for r in results]
    x = np.arange(len(models))
    width = 0.15
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.bar(x, [r['test_acc'] for r in results], width, label='Test Accuracy')
    plt.bar(x + width, [r['best_val_acc'] for r in results], width, label='Validation Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.xticks(x + width/2, [f"Model {i+1}" for i in range(len(models))], rotation=0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot model size
    plt.subplot(2, 2, 2)
    plt.bar(x, [r['model_size_mb'] for r in results], width)
    plt.xlabel('Model')
    plt.ylabel('Size (MB)')
    plt.title('Model Size Comparison')
    plt.xticks(x, [f"Model {i+1}" for i in range(len(models))], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot inference time
    plt.subplot(2, 2, 3)
    plt.bar(x, [r['inference_time_ms'] for r in results], width)
    plt.xlabel('Model')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time Comparison')
    plt.xticks(x, [f"Model {i+1}" for i in range(len(models))], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot parameters
    plt.subplot(2, 2, 4)
    plt.bar(x, [r['num_params'] for r in results], width)
    plt.xlabel('Model')
    plt.ylabel('Parameters')
    plt.title('Parameter Count Comparison')
    plt.xticks(x, [f"Model {i+1}" for i in range(len(models))], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_charts.png'))
    plt.show()
    
    # Add legend for model numbers
    print("\nModel Legend:")
    for i, model in enumerate(models):
        print(f"Model {i+1}: {model}")
    
    # Save results to Google Drive
    try:
        drive_output_dir = '/content/drive/MyDrive/mobilenetv2_improvements'
        os.makedirs(drive_output_dir, exist_ok=True)
        
        # Copy CSV results
        import shutil
        shutil.copy(csv_path, os.path.join(drive_output_dir, 'model_comparison_results.csv'))
        
        # Copy charts
        shutil.copy(
            os.path.join(output_dir, 'model_comparison_charts.png'),
            os.path.join(drive_output_dir, 'model_comparison_charts.png')
        )
        
        print(f"Results saved to Google Drive: {drive_output_dir}")
    except Exception as e:
        print(f"Could not save to Google Drive: {e}")
        print("Please download the results manually.")

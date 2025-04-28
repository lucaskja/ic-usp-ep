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

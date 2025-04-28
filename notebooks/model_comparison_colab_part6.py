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

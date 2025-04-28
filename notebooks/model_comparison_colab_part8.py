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

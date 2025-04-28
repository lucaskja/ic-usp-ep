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

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

"""
Utility functions for model training and evaluation.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm


def setup_logging(model_name, log_dir='logs'):
    """
    Set up logging configuration.
    
    Args:
        model_name (str): Name of the model for log file naming
        log_dir (str): Directory to save log files
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized for {model_name}")


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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    
    Args:
        output (torch.Tensor): Model output (logits)
        target (torch.Tensor): Target labels
        topk (tuple): Tuple of k values for top-k accuracy
        
    Returns:
        list: List of top-k accuracies
    """
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))  # Ensure k doesn't exceed number of classes
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k <= maxk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                # If k exceeds number of classes, use the max available k
                correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # Switch to train mode
    model.train()
    
    # Create progress bar
    desc = f"Epoch {epoch+1}/{total_epochs}" if total_epochs > 0 else "Training"
    pbar = tqdm(train_loader, desc=desc)
    
    end = time.time()
    for i, (images, target) in enumerate(pbar):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        images = images.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(images)
        loss = criterion(output, target)
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'acc@1': f"{top1.avg:.2f}%",
            'time': f"{batch_time.avg:.3f}s"
        })
    
    # Log epoch results
    logging.info(f"Train Epoch: {epoch+1}/{total_epochs if total_epochs > 0 else '?'} "
                f"Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%")
    
    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg}


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
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # Switch to evaluate mode
    model.eval()
    
    # Create progress bar
    pbar = tqdm(val_loader, desc="Validating")
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(pbar):
            # Move data to device
            images = images.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'acc@1': f"{top1.avg:.2f}%"
            })
    
    # Log validation results
    logging.info(f"Validation: Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%")
    
    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg}


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    """
    Save model checkpoint.
    
    Args:
        state (dict): State dictionary to save
        is_best (bool): Whether this is the best model so far
        checkpoint_dir (str): Directory to save checkpoint
        filename (str): Filename for the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(state, best_path)
        logging.info(f"Best model saved to {best_path}")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience (int): Number of epochs to wait after last improvement
        min_delta (float): Minimum change to qualify as improvement
        verbose (bool): If True, prints a message for each improvement
    """
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False
            
        if val_score > self.best_score + self.min_delta:
            if self.verbose:
                logging.info(f"Validation score improved from {self.best_score:.4f} to {val_score:.4f}")
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"Validation score did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    logging.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                self.early_stop = True
                return True
            return False

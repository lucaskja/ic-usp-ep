"""
Utility functions for model training and evaluation.
"""
import os
import time
import torch
import numpy as np
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value"""
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
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # Switch to train mode
    model.train()
    
    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to the same device as model
        images = images.to(device)
        target = target.to(device)
        
        # Compute output
        output = model(images)
        loss = criterion(output, target)
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg}


def validate(model, val_loader, criterion, device):
    """Validate the model on the validation set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # Switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            target = target.to(device)
            
            # Compute output
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg}


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """Save checkpoint to disk"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filepath)

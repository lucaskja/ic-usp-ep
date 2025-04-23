# K-Fold Cross-Validation for Model Evaluation

## Overview

This project uses K-Fold Cross-Validation to ensure robust evaluation of our MobileNetV2 improvements. This document explains how the data is separated and processed during training and evaluation.

## K-Fold Cross-Validation Implementation

### What is K-Fold Cross-Validation?

K-Fold Cross-Validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called 'k' that refers to the number of groups that a given data sample is to be split into.

### Our Implementation

We use 5-fold cross-validation (k=5) in our experiments, which means:

1. The dataset is divided into 5 equal parts (folds)
2. For each fold:
   - The fold is treated as the validation set
   - The remaining 4 folds are used as the training set
   - The model is trained on the training set and evaluated on the validation set
3. The process is repeated 5 times, with each fold used exactly once as the validation data
4. The final performance metrics are averaged across all 5 runs

### Implementation Details

```python
from sklearn.model_selection import KFold

# Initialize the K-Fold cross-validator
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Assuming 'dataset' is your complete dataset
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Training fold {fold+1}/5...")
    
    # Create data loaders for this fold
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_subsampler
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_subsampler
    )
    
    # Initialize model
    model = initialize_model()
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs)
    
    # Evaluate model
    accuracy = evaluate_model(model, val_loader)
    fold_results.append(accuracy)

# Calculate average performance across all folds
average_accuracy = sum(fold_results) / len(fold_results)
print(f"Average accuracy across all folds: {average_accuracy:.4f}")
```

## Benefits of K-Fold Cross-Validation in Our Project

1. **Robust Evaluation**: By training and testing on different subsets of the data, we get a more reliable estimate of the model's performance.

2. **Reduced Variance**: The averaging of results across multiple folds reduces the variance of the performance estimate.

3. **Efficient Data Usage**: All data points are used for both training and validation, making the most of limited datasets.

4. **Fair Comparison**: When comparing different model architectures (base MobileNetV2 vs. our improvements), using the same folds ensures a fair comparison.

## Dataset Split Visualization

```
Dataset (100%)
│
├── Fold 1 (20%) ──┐
│                  │
├── Fold 2 (20%) ──┤
│                  ├── Training Set (80%) for Fold 5 validation
├── Fold 3 (20%) ──┤
│                  │
└── Fold 4 (20%) ──┘
│
└── Fold 5 (20%) ──── Validation Set (20%) for Fold 5
```

This process is repeated 5 times, with each fold serving as the validation set once.

## Final Test Evaluation

After selecting the best model based on the K-Fold cross-validation results, we perform a final evaluation on a separate test set that was not used during the cross-validation process. This provides an unbiased estimate of the model's performance on unseen data.

"""
Default configuration for MobileNetV2 training and evaluation.
"""

# Training configuration
TRAIN_CONFIG = {
    'epochs': 30,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_scheduler': 'step',  # 'step' or 'cosine'
    'lr_step_size': 10,
    'lr_gamma': 0.1,
}

# Data configuration
DATA_CONFIG = {
    'img_size': 224,
    'batch_size': 32,
    'val_split': 0.2,
    'num_workers': 4,
}

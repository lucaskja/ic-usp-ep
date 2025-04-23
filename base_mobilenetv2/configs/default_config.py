"""
Default configuration for base MobileNetV2 model.
"""

# Dataset configuration
DATA_CONFIG = {
    'img_size': 224,
    'batch_size': 32,
    'val_split': 0.2,
    'num_workers': 4
}

# Model configuration
MODEL_CONFIG = {
    'pretrained': True
}

# Training configuration
TRAIN_CONFIG = {
    'epochs': 30,
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine',  # 'step', 'cosine', 'plateau'
    'lr_step_size': 7,
    'lr_gamma': 0.1,
    'early_stopping_patience': 5
}

# Logging configuration
LOG_CONFIG = {
    'log_interval': 10,
    'use_wandb': False,
    'wandb_project': 'mobilenetv2-improvements',
    'wandb_entity': None,
    'wandb_name': 'base-mobilenetv2'
}

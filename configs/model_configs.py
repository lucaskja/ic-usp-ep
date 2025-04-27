"""
Unified configuration for all MobileNetV2 variants.

This module provides configuration settings for:
- Base MobileNetV2
- MobileNetV2 with Mish activation
- MobileNetV2 with Mish and Triplet Attention
- MobileNetV2 with Mish, Triplet Attention, and CNSN
"""

# Common training configuration
TRAIN_CONFIG = {
    'epochs': 60,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_scheduler': 'step',  # 'step' or 'cosine'
    'lr_step_size': 10,
    'lr_gamma': 0.1,
}

# Common data configuration
DATA_CONFIG = {
    'img_size': 224,
    'batch_size': 32,
    'val_split': 0.2,
    'num_workers': 4,
}

# Model-specific configurations
MODEL_CONFIGS = {
    'base': {
        'name': 'Base MobileNetV2',
        'description': 'Standard MobileNetV2 architecture with ReLU6 activation',
        'params': {
            # No additional parameters for base model
        }
    },
    'mish': {
        'name': 'MobileNetV2 with Mish',
        'description': 'MobileNetV2 with Mish activation function replacing ReLU6',
        'params': {
            # No additional parameters for mish model
        }
    },
    'triplet': {
        'name': 'MobileNetV2 with Mish and Triplet Attention',
        'description': 'MobileNetV2 with Mish activation and Triplet Attention mechanism',
        'params': {
            'triplet_attention_kernel_size': 7,
        }
    },
    'cnsn': {
        'name': 'MobileNetV2 with Mish, Triplet Attention, and CNSN',
        'description': 'MobileNetV2 with Mish, Triplet Attention, and CrossNorm-SelfNorm modules',
        'params': {
            'triplet_attention_kernel_size': 7,
            'cn_mode': '2-instance',  # CrossNorm mode: '1-instance', '2-instance', or 'crop'
        }
    }
}

# Evaluation configuration
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'confusion_matrix': True,
    'per_class_metrics': True,
    'save_predictions': True,
}

# Enhanced data augmentation configuration
ENHANCED_AUGMENTATION_CONFIG = {
    'random_resized_crop_scale': (0.7, 1.0),
    'random_horizontal_flip_prob': 0.5,
    'random_vertical_flip_prob': 0.3,
    'random_rotation_degrees': 20,
    'color_jitter': {
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.3,
        'hue': 0.1,
    },
    'random_affine': {
        'degrees': 0,
        'translate': (0.1, 0.1),
        'scale': (0.9, 1.1),
    },
    'random_perspective_distortion_scale': 0.2,
    'random_perspective_prob': 0.5,
    'random_grayscale_prob': 0.1,
    'random_erasing_prob': 0.2,
}

# Standard data augmentation configuration
STANDARD_AUGMENTATION_CONFIG = {
    'random_resized_crop_scale': (0.8, 1.0),
    'random_horizontal_flip_prob': 0.5,
    'random_rotation_degrees': 15,
    'color_jitter': {
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.0,
    },
}

def get_model_config(model_type):
    """
    Get configuration for a specific model type.
    
    Args:
        model_type (str): Type of model ('base', 'mish', 'triplet', or 'cnsn')
        
    Returns:
        dict: Model configuration
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[model_type]

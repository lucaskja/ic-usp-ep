# Dataset Structure for Leaf Disease Classification

## Directory Organization

The dataset should be organized in the following structure for proper functioning of the training and evaluation scripts:

```
datasets/
└── leaf_disease/
    ├── train/
    │   ├── class1/
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   ├── class2/
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   └── ...
    │
    └── test/
        ├── class1/
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── ...
        ├── class2/
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── ...
        └── ...
```

## Dataset Preparation

### Recommended Datasets

This project is designed to work with leaf disease classification datasets such as:

1. **Plant Village Dataset**: Contains 38 classes of plant diseases and healthy plants
2. **PlantDoc Dataset**: A dataset of plant disease images for various crops
3. **Rice Leaf Diseases Dataset**: Focused on rice plant diseases
4. **Cassava Leaf Disease Dataset**: Contains images of cassava leaves with various diseases

### Data Preprocessing

Before training, the data undergoes the following preprocessing steps:

1. **Resizing**: All images are resized to 224×224 pixels to match MobileNetV2's input requirements
2. **Normalization**: Pixel values are normalized using ImageNet mean and standard deviation
3. **Data Augmentation**: During training, the following augmentations are applied:
   - Random horizontal and vertical flips
   - Random rotations (±10 degrees)
   - Random color jitter (brightness, contrast, saturation)
   - Random crops

Example preprocessing code:

```python
from torchvision import transforms

# Training transforms with augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/test transforms without augmentation
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## Class Distribution

For optimal training, ensure that:

1. Each class has a sufficient number of samples (at least 100 images per class is recommended)
2. The class distribution is relatively balanced
3. If the dataset is imbalanced, consider using techniques such as:
   - Weighted sampling
   - Class weights in the loss function
   - Oversampling minority classes
   - Undersampling majority classes

## Dataset Loading

The dataset is loaded using PyTorch's `ImageFolder` and `DataLoader` classes:

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Load datasets
train_dataset = ImageFolder(root='datasets/leaf_disease/train', transform=train_transforms)
test_dataset = ImageFolder(root='datasets/leaf_disease/test', transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```

## K-Fold Cross-Validation

For model development and hyperparameter tuning, we use K-Fold cross-validation as described in the `data_validation.md` file. This ensures robust evaluation of our models before final testing.

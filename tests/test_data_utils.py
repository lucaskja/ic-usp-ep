"""
Tests for data_utils.py and enhanced_data_utils.py
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

from utils.data_utils import split_dataset, load_dataset, get_transforms
from utils.enhanced_data_utils import load_enhanced_dataset, get_enhanced_transforms


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple dataset structure
        self.classes = ['class1', 'class2', 'class3']
        self.num_images_per_class = 20
        
        # Create class directories and dummy images
        for class_name in self.classes:
            class_dir = os.path.join(self.test_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create dummy images
            for i in range(self.num_images_per_class):
                # Create a simple image (10x10 with random values)
                img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
                img.save(os.path.join(class_dir, f'img_{i}.jpg'))
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_split_dataset(self):
        """Test dataset splitting functionality"""
        # Split the dataset
        train_dir, val_dir, test_dir = split_dataset(
            self.test_dir,
            test_ratio=0.1,
            val_ratio=0.2,
            seed=42
        )
        
        # Check that directories were created
        self.assertTrue(os.path.exists(train_dir))
        self.assertTrue(os.path.exists(val_dir))
        self.assertTrue(os.path.exists(test_dir))
        
        # Check that class directories were created in each split
        for class_name in self.classes:
            self.assertTrue(os.path.exists(os.path.join(train_dir, class_name)))
            self.assertTrue(os.path.exists(os.path.join(val_dir, class_name)))
            self.assertTrue(os.path.exists(os.path.join(test_dir, class_name)))
        
        # Check split ratios
        total_images = self.num_images_per_class * len(self.classes)
        expected_test = int(total_images * 0.1)
        expected_val = int((total_images - expected_test) * 0.2)
        expected_train = total_images - expected_test - expected_val
        
        # Count actual images in each split
        train_count = sum(len(os.listdir(os.path.join(train_dir, c))) for c in self.classes)
        val_count = sum(len(os.listdir(os.path.join(val_dir, c))) for c in self.classes)
        test_count = sum(len(os.listdir(os.path.join(test_dir, c))) for c in self.classes)
        
        # Check that counts are close to expected (may not be exact due to rounding)
        self.assertAlmostEqual(train_count, expected_train, delta=len(self.classes))
        self.assertAlmostEqual(val_count, expected_val, delta=len(self.classes))
        self.assertAlmostEqual(test_count, expected_test, delta=len(self.classes))
        
        # Check that all images were used
        self.assertEqual(train_count + val_count + test_count, total_images)
    
    def test_load_dataset(self):
        """Test dataset loading functionality"""
        # Create a temporary output directory for the split dataset
        output_dir = tempfile.mkdtemp()
        
        try:
            # Split the dataset to the output directory
            train_dir, val_dir, test_dir = split_dataset(
                self.test_dir,
                output_dir=output_dir,
                test_ratio=0.1,
                val_ratio=0.2,
                seed=42
            )
            
            # Load the dataset from the output directory
            train_loader, val_loader, test_loader, num_classes = load_dataset(
                output_dir,
                batch_size=4,
                num_workers=0  # Use 0 workers for testing
            )
            
            # Check number of classes
            self.assertEqual(num_classes, len(self.classes))
            
            # Check that data loaders contain data
            self.assertTrue(len(train_loader) > 0)
            self.assertTrue(len(val_loader) > 0)
            self.assertTrue(len(test_loader) > 0)
            
            # Check batch shape
            for images, labels in train_loader:
                self.assertEqual(images.dim(), 4)  # [batch_size, channels, height, width]
                self.assertEqual(images.shape[1], 3)  # 3 channels (RGB)
                self.assertTrue(torch.max(labels) < num_classes)  # Labels are within range
                break
        finally:
            # Clean up
            shutil.rmtree(output_dir)
    
    def test_enhanced_transforms(self):
        """Test enhanced transforms"""
        train_transforms, val_transforms, test_transforms = get_enhanced_transforms()
        
        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        # Apply transforms
        train_img = train_transforms(img)
        val_img = val_transforms(img)
        test_img = test_transforms(img)
        
        # Check output shapes
        self.assertEqual(train_img.shape[0], 3)  # 3 channels
        self.assertEqual(val_img.shape[0], 3)
        self.assertEqual(test_img.shape[0], 3)
        
        # Check that train transform produces different results (due to randomness)
        train_img2 = train_transforms(img)
        # Due to random erasing, we can't directly compare the entire tensors
        # Instead, check that they're not identical (high probability they differ)
        self.assertFalse(torch.all(train_img == train_img2).item())
        
        # Check that val and test transforms are deterministic
        val_img2 = val_transforms(img)
        test_img2 = test_transforms(img)
        self.assertTrue(torch.all(val_img == val_img2).item())
        self.assertTrue(torch.all(test_img == test_img2).item())


if __name__ == '__main__':
    unittest.main()

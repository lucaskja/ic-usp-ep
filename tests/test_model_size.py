"""
Tests for model size reduction with width_mult parameter.
"""
import torch
import unittest
from utils.model_utils import get_model_size
from base_mobilenetv2.models.mobilenetv2 import create_mobilenetv2
from stage1_mish.models.mobilenetv2_mish import create_mobilenetv2_mish
from stage2_triplet.models.mobilenetv2_triplet_fixed import create_mobilenetv2_triplet
from stage3_cnsn.models.mobilenetv2_cnsn_fixed import create_mobilenetv2_cnsn


class TestModelSize(unittest.TestCase):
    """
    Test cases for model size reduction with width_mult parameter.
    """
    def test_base_model_size_reduction(self):
        """
        Test size reduction of base MobileNetV2 model.
        """
        # Create models with different width multipliers
        model_full = create_mobilenetv2(num_classes=10, pretrained=False, width_mult=1.0)
        model_reduced = create_mobilenetv2(num_classes=10, pretrained=False, width_mult=0.75)
        
        # Calculate model sizes
        size_full = get_model_size(model_full)
        size_reduced = get_model_size(model_reduced)
        
        # Check that reduced model is smaller
        self.assertLess(size_reduced, size_full)
        print(f"Base MobileNetV2 size reduction: {size_full:.2f}MB -> {size_reduced:.2f}MB ({(1 - size_reduced/size_full)*100:.2f}% reduction)")
        
    def test_mish_model_size_reduction(self):
        """
        Test size reduction of MobileNetV2 with Mish.
        """
        # Create models with different width multipliers
        model_full = create_mobilenetv2_mish(num_classes=10, pretrained=False, width_mult=1.0)
        model_reduced = create_mobilenetv2_mish(num_classes=10, pretrained=False, width_mult=0.75)
        
        # Calculate model sizes
        size_full = get_model_size(model_full)
        size_reduced = get_model_size(model_reduced)
        
        # Check that reduced model is smaller
        self.assertLess(size_reduced, size_full)
        print(f"MobileNetV2+Mish size reduction: {size_full:.2f}MB -> {size_reduced:.2f}MB ({(1 - size_reduced/size_full)*100:.2f}% reduction)")
        
    def test_triplet_model_size_reduction(self):
        """
        Test size reduction of MobileNetV2 with Mish and Triplet Attention.
        """
        # Create models with different width multipliers
        model_full = create_mobilenetv2_triplet(num_classes=10, pretrained=False, width_mult=1.0)
        model_reduced = create_mobilenetv2_triplet(num_classes=10, pretrained=False, width_mult=0.75)
        
        # Calculate model sizes
        size_full = get_model_size(model_full)
        size_reduced = get_model_size(model_reduced)
        
        # Check that reduced model is smaller
        self.assertLess(size_reduced, size_full)
        print(f"MobileNetV2+Mish+Triplet size reduction: {size_full:.2f}MB -> {size_reduced:.2f}MB ({(1 - size_reduced/size_full)*100:.2f}% reduction)")
        
    def test_cnsn_model_size_reduction(self):
        """
        Test size reduction of MobileNetV2 with Mish, Triplet Attention, and CNSN.
        """
        # Create models with different width multipliers
        model_full = create_mobilenetv2_cnsn(num_classes=10, pretrained=False, width_mult=1.0)
        model_reduced = create_mobilenetv2_cnsn(num_classes=10, pretrained=False, width_mult=0.75)
        
        # Calculate model sizes
        size_full = get_model_size(model_full)
        size_reduced = get_model_size(model_reduced)
        
        # Check that reduced model is smaller
        self.assertLess(size_reduced, size_full)
        print(f"MobileNetV2+Mish+Triplet+CNSN size reduction: {size_full:.2f}MB -> {size_reduced:.2f}MB ({(1 - size_reduced/size_full)*100:.2f}% reduction)")


if __name__ == '__main__':
    unittest.main()

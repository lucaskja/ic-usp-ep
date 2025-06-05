"""
Models module for MemTorch-based CNN architecture.
"""

import sys
import traceback

try:
    # Try to import the full MemTorch implementation
    import memtorch
    print(f"MemTorch successfully imported from {memtorch.__file__}")
    from .memtorch_cnn import MemTorchCNN, InvertedResidualWithSkip
    print("Using full MemTorch implementation")
except ImportError as e:
    # Fall back to the simplified implementation
    error_info = traceback.format_exc()
    print(f"MemTorch import error in {__file__}:")
    print(f"Error details: {str(e)}")
    print("Using simplified MemTorchCNN implementation (without memtorch bindings)")
    from .memtorch_cnn_simplified import MemTorchCNN, InvertedResidualWithSkip

__all__ = ['MemTorchCNN', 'InvertedResidualWithSkip']

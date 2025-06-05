"""
Models module for MemTorch-based CNN architecture.
"""

try:
    # Try to import the full MemTorch implementation
    from .memtorch_cnn import MemTorchCNN, InvertedResidualWithSkip
    print("Using full MemTorch implementation")
except ImportError:
    # Fall back to the simplified implementation
    print("Using simplified MemTorchCNN implementation (without memtorch bindings)")
    from .memtorch_cnn_simplified import MemTorchCNN, InvertedResidualWithSkip

__all__ = ['MemTorchCNN', 'InvertedResidualWithSkip']

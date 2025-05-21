"""
Models module for Memristor-based CNN architecture.
"""

from .memristor_crossbar import MemristorCrossbar
from .memristor_cnn import MemristorCNN
from .memristor_mapping import MemristorMapping
from .memristor_pe import MemristorPE
from .weight_utils import (
    quantize_15_level,
    map_to_differential_pairs,
    compensate_device_variations,
    apply_closed_loop_programming,
    verify_after_write
)

__all__ = [
    'MemristorCrossbar',
    'MemristorCNN',
    'MemristorMapping',
    'MemristorPE',
    'quantize_15_level',
    'map_to_differential_pairs',
    'compensate_device_variations',
    'apply_closed_loop_programming',
    'verify_after_write'
]

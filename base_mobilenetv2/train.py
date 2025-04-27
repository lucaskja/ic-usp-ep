"""
DEPRECATED: This script is deprecated and will be removed in a future version.

Please use the unified training script at the project root instead:
python train.py --model_type base [other options]

This file is kept for reference purposes only.
"""

import warnings
import sys
import os

warnings.warn(
    "This script is deprecated. Please use the unified training script at the project root instead:\n"
    "python train.py --model_type base [other options]",
    DeprecationWarning, 
    stacklevel=2
)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the unified training script
from train import main

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEPRECATED: This script is deprecated and will be removed in a future version.")
    print("Please use the unified training script at the project root instead:")
    print("python train.py --model_type base [other options]")
    print("="*80 + "\n")
    
    # Run the unified training script with model_type=base
    sys.argv.extend(['--model_type', 'base'])
    main()

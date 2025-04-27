"""
DEPRECATED: This script is deprecated and will be removed in a future version.

Please use the unified evaluation script at the project root instead:
python evaluate.py --model_type mish [other options]

This file is kept for reference purposes only.
"""

import warnings
import sys
import os

warnings.warn(
    "This script is deprecated. Please use the unified evaluation script at the project root instead:\n"
    "python evaluate.py --model_type mish [other options]",
    DeprecationWarning, 
    stacklevel=2
)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the unified evaluation script
from evaluate import main

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEPRECATED: This script is deprecated and will be removed in a future version.")
    print("Please use the unified evaluation script at the project root instead:")
    print("python evaluate.py --model_type mish [other options]")
    print("="*80 + "\n")
    
    # Run the unified evaluation script with model_type=mish
    sys.argv.extend(['--model_type', 'mish'])
    main()

"""
DEPRECATED: This script is deprecated and will be removed in a future version.

Please use the unified training and evaluation scripts at the project root instead:
- For training: python train.py --model_type [base|mish|triplet|cnsn] [other options]
- For evaluation: python evaluate.py --model_type [base|mish|triplet|cnsn] [other options]

This file is kept for reference purposes only.
"""

import warnings
import sys
import os

warnings.warn(
    "This script is deprecated. Please use the unified training and evaluation scripts at the project root instead:\n"
    "- For training: python train.py --model_type [base|mish|triplet|cnsn] [other options]\n"
    "- For evaluation: python evaluate.py --model_type [base|mish|triplet|cnsn] [other options]",
    DeprecationWarning, 
    stacklevel=2
)

import argparse

def main():
    """Main function that redirects to the appropriate unified script."""
    parser = argparse.ArgumentParser(description='MobileNetV2 Improvements (DEPRECATED)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'mish', 'triplet', 'cnsn'],
                        help='Model type to train/evaluate')
    
    # Parse just the mode and model_type arguments
    args, unknown = parser.parse_known_args()
    
    print("\n" + "="*80)
    print("DEPRECATED: This script is deprecated and will be removed in a future version.")
    print("Please use the unified scripts at the project root instead:")
    
    if args.mode == 'train':
        print(f"python train.py --model_type {args.model_type} [other options]")
        
        # Import and run the unified training script
        from train import main as train_main
        sys.argv.remove('--mode')
        sys.argv.remove('train')
        train_main()
        
    elif args.mode == 'evaluate':
        print(f"python evaluate.py --model_type {args.model_type} [other options]")
        
        # Import and run the unified evaluation script
        from evaluate import main as eval_main
        sys.argv.remove('--mode')
        sys.argv.remove('evaluate')
        eval_main()
    
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

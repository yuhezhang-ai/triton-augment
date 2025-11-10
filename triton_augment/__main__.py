"""
CLI entry point for Triton-Augment utilities.

Usage:
    python -m triton_augment.warmup [--batch-sizes SIZES] [--image-sizes SIZES]

Author: yuhezhang-ai
"""

import sys
import argparse
import torch
from .utils import warmup_cache


def main():
    parser = argparse.ArgumentParser(
        prog='python -m triton_augment.warmup',
        description='Warm up the Triton auto-tuning cache for your GPU and image sizes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (batch=32,64; size=224,256,512)
  python -m triton_augment.warmup

  # Custom batch sizes
  python -m triton_augment.warmup --batch-sizes 16,32,128

  # Custom image sizes
  python -m triton_augment.warmup --image-sizes 320,384,640

  # Both custom
  python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 320,512

Note: This only needs to be run ONCE per GPU. The cache is persistent.
        """
    )
    
    parser.add_argument(
        '--batch-sizes',
        type=str,
        default='32,64',
        help='Comma-separated batch sizes to warm up (default: 32,64)'
    )
    
    parser.add_argument(
        '--image-sizes',
        type=str,
        default='224,256,512',
        help='Comma-separated square image sizes to warm up (default: 224,256,512)'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Warmup requires a GPU.", file=sys.stderr)
        sys.exit(1)
    
    # Parse sizes
    try:
        batch_sizes = tuple(int(x.strip()) for x in args.batch_sizes.split(','))
        image_sizes = tuple(int(x.strip()) for x in args.image_sizes.split(','))
    except ValueError as e:
        print(f"Error: Invalid size format. Use comma-separated integers.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate sizes
    if not batch_sizes or not image_sizes:
        print("Error: Must provide at least one batch size and one image size.", file=sys.stderr)
        sys.exit(1)
    
    if any(b <= 0 for b in batch_sizes) or any(s <= 0 for s in image_sizes):
        print("Error: All sizes must be positive integers.", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration
    print("="*80)
    print("Triton-Augment Cache Warmup")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch sizes: {list(batch_sizes)}")
    print(f"Image sizes: {list(image_sizes)}")
    print(f"Total configs: {len(batch_sizes)} × {len(image_sizes)} = {len(batch_sizes) * len(image_sizes)}")
    print("="*80)
    print()
    
    # Run warmup
    warmup_cache(
        batch_sizes=batch_sizes,
        image_sizes=image_sizes,
        verbose=True
    )
    
    print("="*80)
    print("✓ Warmup complete! Your kernels are optimized.")
    print("="*80)
    

if __name__ == '__main__':
    main()


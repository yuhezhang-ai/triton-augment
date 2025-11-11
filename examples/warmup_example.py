"""
Example: Warming up the Triton auto-tuning cache.

This script demonstrates how to pre-populate the kernel cache
to avoid auto-tuning delays during training.

IMPORTANT: Auto-tuning must be enabled for warmup to work!

Recommended approach (CLI):
    export TRITON_AUGMENT_ENABLE_AUTOTUNE=1
    python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 640

Or use this Python API example:
    python examples/warmup_example.py

Author: yuhezhang-ai
"""

import torch
import triton_augment as ta


def example_1_basic_warmup():
    """Example 1: Basic warmup with default settings."""
    print("\n" + "="*80)
    print("Example 1: Basic Warmup (Default Settings)")
    print("="*80)
    print()
    
    # Step 1: Enable auto-tuning (REQUIRED!)
    print("Step 1: Enable auto-tuning...")
    ta.enable_autotune()
    print(f"  ✓ Auto-tuning enabled: {ta.is_autotune_enabled()}")
    print()
    
    # Step 2: Warm up the cache
    print("Step 2: Warming up cache with default sizes...")
    print("  → batch_sizes: (32, 64)")
    print("  → image_sizes: (224, 256, 512)")
    print()
    ta.warmup_cache()
    print()
    print("✓ Default warmup complete!")


def example_2_custom_warmup():
    """Example 2: Custom warmup for specific image sizes."""
    print("\n" + "="*80)
    print("Example 2: Custom Warmup (Your Specific Sizes)")
    print("="*80)
    print()
    
    # Enable auto-tuning (if not already enabled)
    if not ta.is_autotune_enabled():
        ta.enable_autotune()
    
    # Warm up for YOUR specific workload
    print("Warming up for object detection workload:")
    print("  → batch_sizes: (64, 128)")
    print("  → image_sizes: (640,)")
    print()
    
    ta.warmup_cache(
        batch_sizes=(64, 128),
        image_sizes=(640,),
        verbose=True
    )
    print()
    print("✓ Custom warmup complete!")


def example_3_check_cache():
    """Example 3: Check cache status."""
    print("\n" + "="*80)
    print("Example 3: Check Cache Status")
    print("="*80)
    print()
    
    cache_dir = ta.utils.get_triton_cache_dir()
    print(f"Cache location: {cache_dir}")
    print()
    
    # Check if first run
    is_first = ta.utils.is_first_run()
    if is_first:
        print("⚠️  This is your FIRST RUN - cache is empty")
        print("   Run warmup to populate the cache!")
    else:
        print("✓ Cache directory exists - previously warmed up")
    print()


def main():
    """Run all warmup examples."""
    if not torch.cuda.is_available():
        print("\n❌ Error: CUDA is not available. Warmup requires a GPU.\n")
        return
    
    print("="*80)
    print("Triton-Augment Cache Warmup Examples")
    print("="*80)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print()
    print("⚠️  IMPORTANT NOTES:")
    print("   1. Auto-tuning must be ENABLED for warmup to work")
    print("   2. Warmup is SIZE-SPECIFIC - use YOUR training sizes!")
    print("   3. Warming 224×224 won't help 640×640")
    print()
    
    # Check initial status
    example_3_check_cache()
    
    # Run examples
    example_1_basic_warmup()
    example_2_custom_warmup()
    
    # Final summary
    print("\n" + "="*80)
    print("✓ All examples complete!")
    print("="*80)
    print()
    print("Your kernels are now auto-tuned. Training will start instantly!")
    print()
    print("💡 For future runs, use the CLI (easier):")
    print("   export TRITON_AUGMENT_ENABLE_AUTOTUNE=1")
    print("   python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 640")
    print()
    print("📖 More info: See docs/auto-tuning.md")
    print("="*80)
    print()


if __name__ == '__main__':
    main()

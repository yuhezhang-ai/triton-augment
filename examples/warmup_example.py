"""
Example: Warming up the Triton auto-tuning cache.

This script demonstrates how to pre-populate the kernel cache
to avoid auto-tuning delays during training.

Recommended: Use the CLI instead:
    python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 320,640

Or run this example:
    python examples/warmup_example.py

Author: yuhezhang-ai
"""

import torch
import triton_augment as ta


def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Warmup requires a GPU.")
        return
    
    print("="*80)
    print("Triton-Augment Cache Warmup Example")
    print("="*80)
    print()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("⚠️  IMPORTANT: Use YOUR actual training sizes!")
    print("   Auto-tuning is size-specific. Warming 224×224 won't help 320×320.")
    print()
    
    # Option 1: Use default settings (for ImageNet-style training)
    print("Option 1: Default warmup (batch=32,64; size=224,256,512)")
    print("   → Good for: ImageNet classification, standard training")
    print("-" * 80)
    ta.warmup_cache()
    print()
    
    # Option 2: Custom sizes for your specific workload
    print("Option 2: Custom warmup for YOUR specific image sizes")
    print("   → Example: Object detection at 640×640, batch=64,128")
    print("-" * 80)
    ta.warmup_cache(
        batch_sizes=(64, 128),
        image_sizes=(640,),
        verbose=True
    )
    print()
    
    print("="*80)
    print("✓ Warmup complete!")
    print()
    print("Your kernels are now optimized. Training will start instantly!")
    print("Cache location:", ta.utils.get_triton_cache_dir())
    print()
    print("💡 Tip: For future runs, use the CLI:")
    print("   python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 640")
    print("="*80)


if __name__ == '__main__':
    main()


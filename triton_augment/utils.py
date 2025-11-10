"""
Utility functions for Triton-Augment.

Author: yuhezhang-ai
"""

import torch
import os
import sys
from pathlib import Path


def get_triton_cache_dir() -> Path:
    """Get the Triton cache directory path."""
    # Triton uses ~/.triton/cache by default
    cache_dir = Path.home() / ".triton" / "cache"
    return cache_dir


def is_first_run() -> bool:
    """
    Check if this is the first run (cache doesn't exist or is empty).
    
    Returns:
        True if cache doesn't exist or is empty, False otherwise
    """
    cache_dir = get_triton_cache_dir()
    
    if not cache_dir.exists():
        return True
    
    # Check if cache has any files
    try:
        # Check for any .cubin or .json files (Triton cache files)
        cache_files = list(cache_dir.rglob("*.cubin")) + list(cache_dir.rglob("*.json"))
        return len(cache_files) == 0
    except Exception:
        return True


def print_first_run_message():
    """Print a helpful message on first run about auto-tuning."""
    gpu_name = "your GPU"
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    
    message = f"""
{'='*80}
[Triton-Augment] First run detected. Optimizing kernels for {gpu_name}.

This will take 5-10 seconds to auto-tune for optimal performance.
All subsequent runs will be instantaneous (results are cached).

💡 Tip: Pre-warm the cache to avoid delays during training:
   
   # Use defaults (batch=32,64; size=224,256,512)
   python -m triton_augment.warmup
   
   # Or specify YOUR training sizes:
   python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 320,640
   
   Note: Auto-tuning is size-specific! Use your actual training dimensions.
{'='*80}
"""
    print(message, file=sys.stderr)


def warmup_cache(
    batch_sizes=(32, 64),
    image_sizes=(224, 256, 512),
    verbose=True
):
    """
    Pre-populate the auto-tuning cache for common image sizes.
    
    This function runs the fused kernel with various common configurations
    to trigger auto-tuning and cache the optimal settings. This eliminates
    the 5-10 second delay on first use.
    
    Args:
        batch_sizes: Tuple of batch sizes to warm up (default: (32, 64))
        image_sizes: Tuple of square image sizes to warm up (default: (224, 256, 512))
        verbose: Whether to print progress messages (default: True)
        
    Example:
        >>> import triton_augment as ta
        >>> # Warm up cache for common training scenarios
        >>> ta.warmup_cache()
        
        >>> # Custom sizes for your specific use case
        >>> ta.warmup_cache(batch_sizes=(16, 128), image_sizes=(128, 384))
    """
    if not torch.cuda.is_available():
        if verbose:
            print("[Triton-Augment] CUDA not available. Skipping warmup.", file=sys.stderr)
        return
    
    from . import functional as F
    
    if verbose:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Triton-Augment] Warming up cache for {gpu_name}...", file=sys.stderr)
        print(f"  Testing {len(batch_sizes)} batch sizes × {len(image_sizes)} image sizes", file=sys.stderr)
    
    total_configs = len(batch_sizes) * len(image_sizes)
    current = 0
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    for batch_size in batch_sizes:
        for img_size in image_sizes:
            current += 1
            if verbose:
                print(f"  [{current}/{total_configs}] Batch={batch_size}, Size={img_size}×{img_size}...", 
                      file=sys.stderr, end=" ", flush=True)
            
            # Create test image
            img = torch.rand(batch_size, 3, img_size, img_size, device='cuda', dtype=torch.float32)
            
            # Trigger auto-tuning for fused kernel
            _ = F.fused_color_normalize(
                img,
                brightness_factor=1.2,
                contrast_factor=1.1,
                saturation_factor=0.9,
                mean=mean,
                std=std,
            )
            
            # Also warm up individual kernels
            _ = F.adjust_saturation(img, 0.9)
            _ = F.normalize(img, mean=mean, std=std)
            
            if verbose:
                print("✓", file=sys.stderr)
            
            # Free memory
            del img
            torch.cuda.empty_cache()
    
    if verbose:
        print(f"[Triton-Augment] Cache warmup complete! Auto-tuning cache saved to:", file=sys.stderr)
        print(f"  {get_triton_cache_dir()}", file=sys.stderr)
        print("", file=sys.stderr)


__all__ = [
    'warmup_cache',
    'is_first_run',
    'print_first_run_message',
    'get_triton_cache_dir',
]


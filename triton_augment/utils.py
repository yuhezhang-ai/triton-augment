"""
Utility functions for Triton-Augment.

Author: yuhezhang-ai
"""

import torch
import os
import sys
from pathlib import Path
from typing import Tuple


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
    
    from .config import ENABLE_AUTOTUNE
    
    autotune_status = "ENABLED" if ENABLE_AUTOTUNE else "DISABLED (default)"
    autotune_info = ""
    
    if ENABLE_AUTOTUNE:
        autotune_info = """
            ⚠️  Auto-tuning is ENABLED. The fused kernel will auto-tune on first use (2-5 sec).
            You'll see: "[Triton-Augment] Auto-tuning fused_color_normalize_kernel..."
            After tuning, that specific size will be instant (cached).
            """
    else:
        autotune_info = """
            ✓  Auto-tuning is DISABLED (using fixed defaults). No tuning delays!
            To enable auto-tuning for optimal performance:
                import triton_augment as ta
                ta.enable_autotune()
            Or set environment variable: TRITON_AUGMENT_ENABLE_AUTOTUNE=1
            """
    
    message = f"""
        {'='*80}
        [Triton-Augment] First run on {gpu_name}.
        Auto-tuning status: {autotune_status}
        {autotune_info}
        💡 Tip: Pre-warm cache to test different sizes:
        python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 320,640
        {'='*80}
        """
    print(message, file=sys.stderr)


# Cache to track which kernel+config combos we've already checked
# This avoids repeated file system calls during training (performance optimization)
_checked_kernels = set()


def should_show_autotune_message(kernel_name: str, cache_key: tuple) -> bool:
    """
    Check if we should show an auto-tuning message for a kernel.
    
    This checks if the specific kernel with this cache key is already tuned.
    Uses an in-memory cache to avoid repeated file system calls during training.
    
    Args:
        kernel_name: Name of the kernel (e.g., "fused_color_normalize_kernel")
        cache_key: Tuple of dimensions that identify this specific config
        
    Returns:
        True if we should show message (kernel will auto-tune), False otherwise
    """
    # Create a unique key for this kernel+config combo
    check_key = (kernel_name, cache_key)
    
    # If we've already checked this kernel+config, don't show message again
    # This avoids file system overhead on every training step
    if check_key in _checked_kernels:
        return False  # Already checked, don't show message again
    
    # Mark as checked before doing file system operations
    _checked_kernels.add(check_key)
    
    cache_dir = get_triton_cache_dir()
    
    # If cache directory doesn't exist, kernel will definitely auto-tune
    if not cache_dir.exists():
        return True  # Show message: will auto-tune
    
    # Look for cache files matching this kernel name
    # Triton cache files include kernel name in the filename
    try:
        kernel_cache_files = list(cache_dir.rglob(f"*{kernel_name}*"))
        
        # If no cache files exist for this kernel at all, it will auto-tune
        if len(kernel_cache_files) == 0:
            return True  # Show message: will auto-tune
        
        # If cache files exist, we can't easily tell if THIS specific size is cached
        # without parsing Triton's internal cache format. 
        # Conservatively assume it's cached and don't show message.
        return False  # Don't show message: probably cached
        
    except Exception:
        # On error, assume it will auto-tune to be safe
        return True  # Show message: might auto-tune


def warmup_cache(
    batch_sizes: Tuple[int, ...] = (32, 64),
    image_sizes: Tuple[int, ...] = (224, 256, 512),
    verbose: bool = True
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
        ```python
        import triton_augment as ta
        # Warm up cache for common training scenarios
        ta.warmup_cache()
        # Custom sizes for your specific use case
        ta.warmup_cache(batch_sizes=(16, 128), image_sizes=(128, 384))
        ```
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
    'should_show_autotune_message',
]


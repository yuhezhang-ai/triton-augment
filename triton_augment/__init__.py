"""
Triton-Augment: GPU-Accelerated Image Augmentation with Kernel Fusion

A high-performance image augmentation library that leverages OpenAI Triton
to fuse common per-pixel operations, providing significant speedups over
standard PyTorch implementations.

Key Features:
- Fused color jitter (brightness, contrast, saturation) operations
- Fused normalization
- Zero intermediate memory allocations
- Familiar torchvision-like API
- Compatible with PyTorch data loading pipelines

Example:
    ```python
    import torch
    import triton_augment as ta
    
    # Create a fused transform
    transform = ta.TritonFusedAugment(
        crop_size=112,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    # Apply to images
    img = torch.rand(4, 3, 224, 224, device='cuda')
    augmented = transform(img)
    ```
"""

from . import functional
from . import transforms
from . import utils
from . import config

# Import utility functions
from .utils import warmup_cache
from .config import enable_autotune, disable_autotune, is_autotune_enabled

# First-run detection and helpful message
import os
# Allow suppressing the message via environment variable (useful for CI/CD)
if os.getenv('TRITON_AUGMENT_SUPPRESS_FIRST_RUN_MESSAGE') != '1':
    if utils.is_first_run():
        utils.print_first_run_message()

# Import commonly used transforms
from .transforms import (
    # Color transforms
    TritonColorJitter,
    TritonNormalize,
    TritonColorJitterNormalize,
    TritonGrayscale,
    TritonRandomGrayscale,
    # Geometric transforms
    TritonRandomCrop,
    TritonCenterCrop,
    TritonRandomHorizontalFlip,
    TritonRandomCropFlip,
    # Fused augmentation
    TritonFusedAugment,
)

# Import commonly used functional operations
from .functional import (
    # Color operations
    adjust_brightness,
    adjust_contrast,
    adjust_contrast_fast,
    adjust_saturation,
    normalize,
    rgb_to_grayscale,
    apply_brightness,
    apply_contrast,
    apply_saturation,
    apply_normalize,
    # Geometric operations
    crop,
    center_crop,
    horizontal_flip,
    # Fused operations
    fused_augment,
)

__version__ = "0.2.0"

__all__ = [
    # Submodules
    'functional',
    'transforms',
    'utils',
    'config',
    
    # Utilities
    'warmup_cache',
    'enable_autotune',
    'disable_autotune',
    'is_autotune_enabled',
    
    # Transform classes - Color
    'TritonColorJitter',
    'TritonNormalize',
    'TritonColorJitterNormalize',
    'TritonGrayscale',
    'TritonRandomGrayscale',
    # Transform classes - Geometric
    'TritonRandomCrop',
    'TritonCenterCrop',
    'TritonRandomHorizontalFlip',
    'TritonRandomCropFlip',
    # Transform classes - Fused Augmentation
    'TritonFusedAugment',
    
    # Functional API - Color operations
    'adjust_brightness',
    'adjust_contrast',
    'adjust_contrast_fast',
    'adjust_saturation',
    'normalize',
    'rgb_to_grayscale',
    'apply_brightness',
    'apply_contrast',
    'apply_saturation',
    'apply_normalize',
    
    # Functional API - Geometric operations
    'crop',
    'center_crop',
    'horizontal_flip',
    
    # Functional API - Fused operations
    'fused_augment',
]


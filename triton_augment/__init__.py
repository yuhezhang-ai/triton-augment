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
    >>> import torch
    >>> import triton_augment as ta
    >>> 
    >>> # Create a fused transform
    >>> transform = ta.TritonColorJitterNormalize(
    ...     brightness=0.2,
    ...     contrast=0.2,
    ...     saturation=0.2,
    ...     mean=(0.485, 0.456, 0.406),
    ...     std=(0.229, 0.224, 0.225)
    ... )
    >>> 
    >>> # Apply to images
    >>> img = torch.rand(4, 3, 224, 224, device='cuda')
    >>> augmented = transform(img)
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
    TritonColorJitter,
    TritonNormalize,
    TritonColorJitterNormalize,
    TritonGrayscale,
    TritonRandomGrayscale,
)

# Import commonly used functional operations
from .functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_contrast_fast,
    adjust_saturation,
    normalize,
    rgb_to_grayscale,
    random_grayscale,
    apply_brightness,
    apply_contrast,
    apply_saturation,
    apply_normalize,
    fused_color_normalize,
)

__version__ = "0.1.0"

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
    
    # Transform classes
    'TritonColorJitter',
    'TritonNormalize',
    'TritonColorJitterNormalize',
    'TritonGrayscale',
    'TritonRandomGrayscale',
    
    # Functional API
    'adjust_brightness',
    'adjust_contrast',
    'adjust_contrast_fast',
    'adjust_saturation',
    'normalize',
    'rgb_to_grayscale',
    'random_grayscale',
    'apply_brightness',
    'apply_contrast',
    'apply_saturation',
    'apply_normalize',
    'fused_color_normalize',
]


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

# Import commonly used transforms
from .transforms import (
    TritonColorJitter,
    TritonNormalize,
    TritonColorJitterNormalize,
)

# Import commonly used functional operations
from .functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_contrast_fast,
    adjust_saturation,
    normalize,
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
    
    # Transform classes
    'TritonColorJitter',
    'TritonNormalize',
    'TritonColorJitterNormalize',
    
    # Functional API
    'adjust_brightness',
    'adjust_contrast',
    'adjust_contrast_fast',
    'adjust_saturation',
    'normalize',
    'apply_brightness',
    'apply_contrast',
    'apply_saturation',
    'apply_normalize',
    'fused_color_normalize',
]


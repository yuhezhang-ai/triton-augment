"""
Triton kernel implementations for image augmentation.

This package contains the raw Triton JIT-compiled kernels used by the
functional API. These kernels are optimized for performance and kernel fusion.

Author: yuhezhang-ai
"""

from .color_normalize_kernel import (
    brightness_kernel,
    contrast_kernel,
    contrast_fast_kernel,
    saturation_kernel,
    normalize_kernel,
    rgb_to_grayscale_kernel,
)

from .fused_kernel import (
    ultimate_fused_kernel,
)

__all__ = [
    'brightness_kernel',
    'contrast_kernel',
    'contrast_fast_kernel',
    'saturation_kernel',
    'normalize_kernel',
    'rgb_to_grayscale_kernel',
    'ultimate_fused_kernel',
]


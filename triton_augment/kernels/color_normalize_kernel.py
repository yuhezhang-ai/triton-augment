"""
Triton kernels for fused color jitter and normalization operations.

This module contains the raw Triton JIT-compiled kernels that perform
element-wise transformations with kernel fusion to minimize memory bandwidth.

Implementation matches torchvision.transforms.v2 functional API exactly.
Reference: torchvision/transforms/v2/functional/_color.py

Author: yuhezhang-ai
"""

import triton
import triton.language as tl
import torch
from ..config import ENABLE_AUTOTUNE


# Default configuration when auto-tuning is disabled
DEFAULT_CONFIG = triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3)

# Multiple configurations to search when auto-tuning is enabled
# These offer robustness across different GPU architectures (T4, RTX, A100)
AUTOTUNE_CONFIGS = [
    # Config 1: Good for smaller, older GPUs (lower register pressure)
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2), 
    
    # Config 2: Balanced, high occupancy, robust default
    triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
    
    # Config 3: Higher concurrency (num_warps=8) for large data center GPUs
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3), 

    # Config 4: Max block size, testing higher staging for L2 cache
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4), 
]


def _get_autotune_configs():
    """Returns the appropriate list of configurations based on the global flag."""
    if ENABLE_AUTOTUNE:
        # If enabled, Triton searches these configs and caches the fastest one
        return AUTOTUNE_CONFIGS
    else:
        # If disabled, Triton only checks the single default config (zero tuning overhead)
        return [DEFAULT_CONFIG]


@triton.autotune(
    configs=_get_autotune_configs(),
    key=['N'],  # Tune based on total elements
)
@triton.jit
def fused_color_normalize_kernel(
    # Pointers
    input_ptr,
    output_ptr,
    # Total elements (for auto-tuning key)
    N,  # batch_size * channels * height * width
    # Shape parameters
    batch_size,
    channels,
    height,
    width,
    # Channel parameters (for normalization)
    norm_mean_ptr,  # Pointer to normalization mean array [C]
    norm_std_ptr,   # Pointer to normalization std array [C]
    # Color jitter parameters
    brightness_factor,
    contrast_factor,
    saturation_factor,
    # Flags                                
    apply_brightness: tl.constexpr,
    apply_contrast: tl.constexpr,
    apply_saturation: tl.constexpr,
    apply_normalize: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for color jitter and normalization with TWO processing paths.
    
    **OPTIMIZATION**: This kernel uses compile-time branching to choose between:
    - LINEAR processing (when saturation NOT needed): Process N*C*H*W pixels individually
      → Simpler, faster, better memory access
    - SPATIAL processing (when saturation IS needed): Process N*H*W locations with RGB triplets
      → Required for saturation (needs R,G,B together for grayscale computation)
    
    Operations applied in sequence:
    1. Brightness: pixel = pixel * brightness_factor (MULTIPLICATIVE)
    2. Contrast (FAST): pixel = (pixel - 0.5) * contrast_factor + 0.5
    3. Saturation: pixel = blend(pixel, grayscale, saturation_factor) [torchvision-exact]
    4. Normalize: pixel = (pixel - mean) / std
    
    NOTE: Contrast uses FAST centered scaling, NOT torchvision's blend-with-mean.
    For torchvision-exact contrast, use adjust_contrast() separately.
    
    Args:
        input_ptr: Pointer to input tensor (N, C, H, W) flattened
        output_ptr: Pointer to output tensor (N, C, H, W) flattened
        N: Total number of elements (batch_size * channels * height * width)
           Used as auto-tuning key, not in computation
        batch_size, channels, height, width: Tensor dimensions
        norm_mean_ptr, norm_std_ptr: Normalization parameters [C]
        brightness_factor, contrast_factor, saturation_factor: Adjustment factors
        apply_* flags: Compile-time constants to enable/disable operations
        BLOCK_SIZE: Number of elements to process per thread block (auto-tuned)
    """
    spatial_size = height * width
    
    if apply_saturation:
        # ========================================================================
        # SPATIAL PROCESSING PATH (when saturation is needed)
        # ========================================================================
        # Process N*H*W spatial positions, loading RGB triplets together.
        # Required because saturation needs grayscale = 0.2989*R + 0.587*G + 0.114*B
        
        total_spatial = batch_size * spatial_size
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        spatial_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        spatial_mask = spatial_offsets < total_spatial
        
        # Calculate batch and spatial indices
        batch_idx = spatial_offsets // spatial_size
        spatial_idx = spatial_offsets % spatial_size
        
        # Load RGB channels for each spatial location
        # Layout: [N, C, H, W] -> offset = n * C * H * W + c * H * W + spatial_idx
        r_offset = batch_idx * channels * spatial_size + 0 * spatial_size + spatial_idx
        g_offset = batch_idx * channels * spatial_size + 1 * spatial_size + spatial_idx
        b_offset = batch_idx * channels * spatial_size + 2 * spatial_size + spatial_idx
        
        r = tl.load(input_ptr + r_offset, mask=spatial_mask, other=0.0)
        g = tl.load(input_ptr + g_offset, mask=spatial_mask, other=0.0)
        b = tl.load(input_ptr + b_offset, mask=spatial_mask, other=0.0)
        
        # Apply brightness adjustment (MULTIPLICATIVE per torchvision)
        if apply_brightness:
            r = r * brightness_factor
            g = g * brightness_factor
            b = b * brightness_factor
            # Clamp to [0, 1] as torchvision does
            r = tl.maximum(0.0, tl.minimum(1.0, r))
            g = tl.maximum(0.0, tl.minimum(1.0, g))
            b = tl.maximum(0.0, tl.minimum(1.0, b))
        
        # Apply contrast adjustment (FAST version: centered scaling)
        if apply_contrast:
            r = (r - 0.5) * contrast_factor + 0.5
            g = (g - 0.5) * contrast_factor + 0.5
            b = (b - 0.5) * contrast_factor + 0.5
            # Clamp to [0, 1]
            r = tl.maximum(0.0, tl.minimum(1.0, r))
            g = tl.maximum(0.0, tl.minimum(1.0, g))
            b = tl.maximum(0.0, tl.minimum(1.0, b))
        
        # Apply saturation adjustment with proper RGB to grayscale conversion
        # Uses torchvision's weights: 0.2989*R + 0.587*G + 0.114*B
        # Convert to grayscale using exact torchvision formula
        gray = 0.2989 * r + 0.587 * g + 0.114 * b
        
        # blend(color, gray, saturation_factor) = color * saturation_factor + gray * (1 - saturation_factor)
        r = r * saturation_factor + gray * (1.0 - saturation_factor)
        g = g * saturation_factor + gray * (1.0 - saturation_factor)
        b = b * saturation_factor + gray * (1.0 - saturation_factor)
        # Clamp to [0, 1] as torchvision does
        r = tl.maximum(0.0, tl.minimum(1.0, r))
        g = tl.maximum(0.0, tl.minimum(1.0, g))
        b = tl.maximum(0.0, tl.minimum(1.0, b))
        
        # Apply normalization (per-channel)
        if apply_normalize:
            norm_mean_r = tl.load(norm_mean_ptr + 0)
            norm_mean_g = tl.load(norm_mean_ptr + 1)
            norm_mean_b = tl.load(norm_mean_ptr + 2)
            
            norm_std_r = tl.load(norm_std_ptr + 0)
            norm_std_g = tl.load(norm_std_ptr + 1)
            norm_std_b = tl.load(norm_std_ptr + 2)
            
            r = (r - norm_mean_r) / norm_std_r
            g = (g - norm_mean_g) / norm_std_g
            b = (b - norm_mean_b) / norm_std_b
        
        # Store the results
        tl.store(output_ptr + r_offset, r, mask=spatial_mask)
        tl.store(output_ptr + g_offset, g, mask=spatial_mask)
        tl.store(output_ptr + b_offset, b, mask=spatial_mask)
    
    else:
        # ========================================================================
        # LINEAR PROCESSING PATH (when saturation is NOT needed)
        # ========================================================================
        # Process N*C*H*W pixels individually. This is FASTER because:
        # - Simpler index calculations
        # - Better memory coalescing
        # - Less register pressure
        # - No need to load/store RGB triplets together
        
        total_elements = batch_size * channels * spatial_size
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        # Load single pixel value
        pixel = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Apply brightness adjustment (MULTIPLICATIVE per torchvision)
        if apply_brightness:
            pixel = pixel * brightness_factor
            pixel = tl.maximum(0.0, tl.minimum(1.0, pixel))
        
        # Apply contrast adjustment (FAST version: centered scaling)
        if apply_contrast:
            pixel = (pixel - 0.5) * contrast_factor + 0.5
            pixel = tl.maximum(0.0, tl.minimum(1.0, pixel))
        
        # Apply normalization (per-channel)
        if apply_normalize:
            # Calculate which channel this pixel belongs to
            channel_idx = (offsets // spatial_size) % channels
            
            # Load channel-specific normalization parameters
            norm_mean = tl.load(norm_mean_ptr + channel_idx, mask=mask, other=0.0)
            norm_std = tl.load(norm_std_ptr + channel_idx, mask=mask, other=0.0)
            
            pixel = (pixel - norm_mean) / norm_std
        
        # Store the result
        tl.store(output_ptr + offsets, pixel, mask=mask)


@triton.jit
def brightness_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    brightness_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """Brightness adjustment kernel matching torchvision (MULTIPLICATIVE)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pixel = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    pixel = pixel * brightness_factor  # MULTIPLICATIVE per torchvision
    pixel = tl.maximum(0.0, tl.minimum(1.0, pixel))  # Clamp to [0, 1]
    tl.store(output_ptr + offsets, pixel, mask=mask)


@triton.jit
def contrast_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,  # Pre-computed grayscale mean per image [batch_size]
    batch_size,
    channels,
    height,
    width,
    contrast_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """Contrast adjustment kernel matching torchvision.
    
    Applies: pixel = pixel * contrast_factor + mean * (1 - contrast_factor)
    Same formula for all channels, so we process all pixels uniformly.
    """
    spatial_size = height * width
    total_elements = batch_size * channels * spatial_size
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Determine which batch each pixel belongs to
    batch_idx = offsets // (channels * spatial_size)
    
    # Load pixel value
    pixel = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load grayscale mean for this batch
    grayscale_mean = tl.load(mean_ptr + batch_idx, mask=mask, other=0.0)
    
    # Blend with mean (same formula for all channels)
    pixel = pixel * contrast_factor + grayscale_mean * (1.0 - contrast_factor)
    
    # Clamp to [0, 1]
    pixel = tl.maximum(0.0, tl.minimum(1.0, pixel))
    
    tl.store(output_ptr + offsets, pixel, mask=mask)


@triton.jit
def contrast_fast_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    contrast_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast contrast adjustment using simple centered scaling.
    
    This is faster than torchvision's method because it doesn't require
    computing the grayscale mean. It applies: output = (input - 0.5) * contrast + 0.5
    
    Note: This is NOT equivalent to torchvision's adjust_contrast, but provides
    similar perceptual results and is fully fusible with other operations.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pixel = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Center around 0.5, scale, and re-center
    pixel = (pixel - 0.5) * contrast_factor + 0.5
    # Clamp to [0, 1]
    pixel = tl.maximum(0.0, tl.minimum(1.0, pixel))
    tl.store(output_ptr + offsets, pixel, mask=mask)


@triton.jit
def saturation_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    saturation_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """Saturation adjustment kernel matching torchvision."""
    spatial_size = height * width
    total_elements = batch_size * spatial_size
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    spatial_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < total_elements
    
    batch_idx = spatial_offsets // spatial_size
    spatial_idx = spatial_offsets % spatial_size
    
    # Load RGB channels
    r_offset = batch_idx * channels * spatial_size + 0 * spatial_size + spatial_idx
    g_offset = batch_idx * channels * spatial_size + 1 * spatial_size + spatial_idx
    b_offset = batch_idx * channels * spatial_size + 2 * spatial_size + spatial_idx
    
    r = tl.load(input_ptr + r_offset, mask=spatial_mask, other=0.0)
    g = tl.load(input_ptr + g_offset, mask=spatial_mask, other=0.0)
    b = tl.load(input_ptr + b_offset, mask=spatial_mask, other=0.0)
    
    # Convert to grayscale using torchvision formula
    gray = 0.2989 * r + 0.587 * g + 0.114 * b
    
    # Blend with grayscale
    r = r * saturation_factor + gray * (1.0 - saturation_factor)
    g = g * saturation_factor + gray * (1.0 - saturation_factor)
    b = b * saturation_factor + gray * (1.0 - saturation_factor)
    
    # Clamp to [0, 1]
    r = tl.maximum(0.0, tl.minimum(1.0, r))
    g = tl.maximum(0.0, tl.minimum(1.0, g))
    b = tl.maximum(0.0, tl.minimum(1.0, b))
    
    tl.store(output_ptr + r_offset, r, mask=spatial_mask)
    tl.store(output_ptr + g_offset, g, mask=spatial_mask)
    tl.store(output_ptr + b_offset, b, mask=spatial_mask)


@triton.jit
def normalize_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    spatial_size,  # H * W
    n_channels,
    mean_ptr,
    std_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalization kernel for NCHW format.
    
    For NCHW format: offset = n * (C*H*W) + c * (H*W) + h * W + w
    Channel index: c = (offset // spatial_size) % n_channels
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pixel = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel index for NCHW format
    # offset = n * (C*H*W) + c * (H*W) + spatial_pos
    # channel = (offset // spatial_size) % n_channels
    channel_idx = (offsets // spatial_size) % n_channels
    
    # Load mean and std for the channel
    mean = tl.load(mean_ptr + channel_idx, mask=mask)
    std = tl.load(std_ptr + channel_idx, mask=mask)
    
    pixel = (pixel - mean) / std
    tl.store(output_ptr + offsets, pixel, mask=mask)

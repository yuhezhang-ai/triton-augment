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
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    std = tl.load(std_ptr + channel_idx, mask=mask, other=1.0)
    
    pixel = (pixel - mean) / std
    tl.store(output_ptr + offsets, pixel, mask=mask)


@triton.jit
def rgb_to_grayscale_kernel(
    input_ptr,
    output_ptr,
    grayscale_mask_ptr,  # Per-image grayscale decisions [N] (uint8: 0 or 1)
    batch_size,
    channels,
    height,
    width,
    apply_per_image: tl.constexpr,  # Whether to use per-image masks
    BLOCK_SIZE: tl.constexpr,
):
    """
    Convert RGB to grayscale with optional per-image masking.
    
    This kernel converts RGB images to grayscale using torchvision's formula:
    gray = 0.2989*R + 0.587*G + 0.114*B
    
    Supports per-image conditional application:
    - If apply_per_image=True: Uses grayscale_mask_ptr to decide per image
    - If apply_per_image=False: Converts all images
    
    Args:
        input_ptr: Input tensor [N, 3, H, W]
        output_ptr: Output tensor [N, 3, H, W] (grayscale replicated to 3 channels)
        grayscale_mask_ptr: Per-image decisions [N] (uint8: 0=keep original, 1=convert to gray)
        batch_size: Number of images
        channels: Number of channels (must be 3)
        height, width: Image dimensions
        apply_per_image: If True, check mask per image; if False, convert all
        BLOCK_SIZE: Elements to process per thread block
    """
    spatial_size = height * width
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
    
    # Compute grayscale value
    gray = 0.2989 * r + 0.587 * g + 0.114 * b
    
    # Conditionally apply grayscale based on mask
    if apply_per_image:
        # Load per-image decision (bool tensor)
        do_grayscale = tl.load(grayscale_mask_ptr + batch_idx, mask=spatial_mask, other=False)
        # Use tl.where to select gray or original per image
        r_out = tl.where(do_grayscale, gray, r)
        g_out = tl.where(do_grayscale, gray, g)
        b_out = tl.where(do_grayscale, gray, b)
    else:
        # Convert all images to grayscale
        r_out = gray
        g_out = gray
        b_out = gray
    
    # Store results
    tl.store(output_ptr + r_offset, r_out, mask=spatial_mask)
    tl.store(output_ptr + g_offset, g_out, mask=spatial_mask)
    tl.store(output_ptr + b_offset, b_out, mask=spatial_mask)

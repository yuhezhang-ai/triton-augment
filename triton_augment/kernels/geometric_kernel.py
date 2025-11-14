"""
Triton kernels for geometric transformations (crop, flip, etc.)

This module implements GPU-accelerated geometric transformations using Triton.
Operations modify memory indexing rather than pixel values.

Author: yuhezhang-ai
"""

import triton
import triton.language as tl


@triton.jit
def crop_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    top_offsets_ptr,   # Per-image top offsets [N] (int32)
    left_offsets_ptr,  # Per-image left offsets [N] (int32)
    apply_per_image: tl.constexpr,  # Whether to use per-image offsets
    BLOCK_SIZE: tl.constexpr,
):
    """
    Crop a rectangular region from input tensor, with optional per-image crop positions.
    
    This kernel copies pixels from a specific region of the input to the output.
    Supports both uniform crop (all images cropped at same position) and per-image
    crop (each image cropped at different position).
    
    Memory Layout: NCHW (N=batch, C=channels, H=height, W=width)
    
    Args:
        input_ptr: Pointer to input tensor [N, C, input_H, input_W]
        output_ptr: Pointer to output tensor [N, C, output_H, output_W]
        batch_size: Number of images in batch
        channels: Number of channels (e.g., 3 for RGB)
        input_height, input_width: Input image dimensions
        output_height, output_width: Output (cropped) dimensions
        top_offsets_ptr: Pointer to per-image top offsets [N] (int32)
        left_offsets_ptr: Pointer to per-image left offsets [N] (int32)
        apply_per_image: If True, use per-image offsets. If False, all images use offsets[0].
        BLOCK_SIZE: Number of elements to process per thread block
    
    Processing Strategy:
        - Process all N*C*output_H*output_W elements linearly
        - For each output position, calculate corresponding input position
        - Input position = output position + offset (per-image or uniform)
    """
    # Total number of elements in output
    total_elements = batch_size * channels * output_height * output_width
    
    # Get program ID and calculate element offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose flat offset into (n, c, h, w) for output
    output_spatial_size = output_height * output_width
    n = offsets // (channels * output_spatial_size)
    remainder = offsets % (channels * output_spatial_size)
    c = remainder // output_spatial_size
    remainder = remainder % output_spatial_size
    h_out = remainder // output_width
    w_out = remainder % output_width
    
    # Load crop offsets (per-image or uniform)
    if apply_per_image:
        top_offset = tl.load(top_offsets_ptr + n, mask=mask, other=0)
        left_offset = tl.load(left_offsets_ptr + n, mask=mask, other=0)
    else:
        # All images use the same offset (load scalar from first element, no mask needed)
        top_offset = tl.load(top_offsets_ptr)
        left_offset = tl.load(left_offsets_ptr)
    
    # Calculate corresponding input position (apply crop offset)
    h_in = h_out + top_offset
    w_in = w_out + left_offset
    
    # Calculate flat input offset
    input_spatial_size = input_height * input_width
    input_offset = (
        n * channels * input_spatial_size +
        c * input_spatial_size +
        h_in * input_width +
        w_in
    )
    
    # Load from input, store to output
    pixel = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, pixel, mask=mask)


@triton.jit
def horizontal_flip_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    flip_mask_ptr,  # Per-image flip decisions [N] (uint8: 0 or 1), None if all flip
    apply_per_image: tl.constexpr,  # Whether to check flip_mask per image
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flip image horizontally (reverse width dimension), with optional per-image control.
    
    This kernel can either flip all images (when apply_per_image=False) or
    selectively flip based on per-image mask (when apply_per_image=True).
    
    Memory Layout: NCHW (N=batch, C=channels, H=height, W=width)
    
    Args:
        input_ptr: Pointer to input tensor [N, C, H, W]
        output_ptr: Pointer to output tensor [N, C, H, W]
        batch_size: Number of images in batch
        channels: Number of channels (e.g., 3 for RGB)
        height: Image height
        width: Image width
        flip_mask_ptr: Pointer to per-image flip decisions [N] (uint8: 0=no flip, 1=flip)
        apply_per_image: If True, check flip_mask per image. If False, flip all images.
        BLOCK_SIZE: Number of elements to process per thread block
    
    Processing Strategy:
        - Process all N*C*H*W elements linearly
        - For each output position (n, c, h, w):
          - If apply_per_image: check flip_mask[n], conditionally flip
          - Else: flip all images
    """
    # Total number of elements
    total_elements = batch_size * channels * height * width
    
    # Get program ID and calculate element offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose flat offset into (n, c, h, w) for output
    spatial_size = height * width
    n = offsets // (channels * spatial_size)
    remainder = offsets % (channels * spatial_size)
    c = remainder // spatial_size
    remainder = remainder % spatial_size
    h = remainder // width
    w_out = remainder % width
    
    # Determine input width coordinate (flip or not)
    if apply_per_image:
        # Load per-image flip decision (bool tensor)
        do_flip = tl.load(flip_mask_ptr + n, mask=mask, other=False)
        # Conditionally flip: w_in = flip ? (width-1-w_out) : w_out
        w_in = tl.where(do_flip, width - 1 - w_out, w_out)
    else:
        # Flip all images
        w_in = width - 1 - w_out
    
    # Calculate flat input offset
    input_offset = (
        n * channels * spatial_size +
        c * spatial_size +
        h * width +
        w_in
    )
    
    # Load from flipped input position, store to output
    pixel = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, pixel, mask=mask)
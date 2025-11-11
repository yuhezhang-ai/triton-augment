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
    top_offset,
    left_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Crop a rectangular region from input tensor.
    
    This kernel copies pixels from a specific region of the input to the output.
    The region is defined by (top_offset, left_offset) as the top-left corner
    and (output_height, output_width) as the size.
    
    Memory Layout: NCHW (N=batch, C=channels, H=height, W=width)
    
    Args:
        input_ptr: Pointer to input tensor [N, C, input_H, input_W]
        output_ptr: Pointer to output tensor [N, C, output_H, output_W]
        batch_size: Number of images in batch
        channels: Number of channels (e.g., 3 for RGB)
        input_height, input_width: Input image dimensions
        output_height, output_width: Output (cropped) dimensions
        top_offset: Y coordinate of crop's top-left corner in input
        left_offset: X coordinate of crop's top-left corner in input
        BLOCK_SIZE: Number of elements to process per thread block
    
    Processing Strategy:
        - Process all N*C*output_H*output_W elements linearly
        - For each output position, calculate corresponding input position
        - Input position = output position + offset
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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flip image horizontally (reverse width dimension).
    
    This kernel reverses the order of pixels along the width dimension.
    For each output position (n, c, h, w), it reads from input position
    (n, c, h, width-1-w).
    
    Memory Layout: NCHW (N=batch, C=channels, H=height, W=width)
    
    Args:
        input_ptr: Pointer to input tensor [N, C, H, W]
        output_ptr: Pointer to output tensor [N, C, H, W]
        batch_size: Number of images in batch
        channels: Number of channels (e.g., 3 for RGB)
        height: Image height
        width: Image width
        BLOCK_SIZE: Number of elements to process per thread block
    
    Processing Strategy:
        - Process all N*C*H*W elements linearly
        - For each output position (n, c, h, w), read from (n, c, h, W-1-w)
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
    
    # Flip width coordinate for input
    w_in = width - 1 - w_out
    
    # Calculate flat input offset with flipped width
    input_offset = (
        n * channels * spatial_size +
        c * spatial_size +
        h * width +
        w_in
    )
    
    # Load from flipped input position, store to output
    pixel = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, pixel, mask=mask)


@triton.jit
def fused_crop_flip_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    top_offset,
    left_offset,
    do_flip: tl.constexpr,  # Compile-time constant (0 or 1)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused crop + horizontal flip in a single kernel.
    
    This kernel combines crop and flip operations by calculating the final
    input index transformation in one pass, eliminating intermediate memory
    transfers between the two operations.
    
    **OPTIMIZATION**: Uses compile-time branching via tl.constexpr:
    - When do_flip=0: Only crop logic is compiled (no flip overhead)
    - When do_flip=1: Both crop and flip logic are compiled
    
    Processing Order:
    1. Calculate output position (n, c, h, w)
    2. Apply flip if enabled: w = output_width - 1 - w
    3. Apply crop offset: h_in = h + top, w_in = w + left
    4. Single memory read from calculated position
    5. Single memory write to output
    
    Memory Layout: NCHW (N=batch, C=channels, H=height, W=width)
    
    Args:
        input_ptr: Pointer to input tensor [N, C, input_H, input_W]
        output_ptr: Pointer to output tensor [N, C, output_H, output_W]
        batch_size: Number of images in batch
        channels: Number of channels (e.g., 3 for RGB)
        input_height, input_width: Input image dimensions
        output_height, output_width: Output (cropped) dimensions
        top_offset: Y coordinate of crop's top-left corner in input
        left_offset: X coordinate of crop's top-left corner in input
        do_flip: 1 to apply horizontal flip after crop, 0 to skip
        BLOCK_SIZE: Number of elements to process per thread block
    
    Performance:
        - ~1.5-2x faster than sequential crop + flip
        - Eliminates intermediate memory transfers
        - No performance penalty when do_flip=0 (compile-time optimization)
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
    
    # Apply flip logic (compile-time branching - zero overhead when do_flip=0)
    if do_flip:
        w_out = output_width - 1 - w_out
    
    # Apply crop offset
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
    
    # Single memory read + single memory write (no intermediate buffer!)
    pixel = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, pixel, mask=mask)


@triton.jit
def ultimate_fused_kernel(
    # Pointers
    input_ptr,
    output_ptr,
    # Geometric parameters
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    top_offset,
    left_offset,
    do_flip: tl.constexpr,  # Compile-time constant for flip
    # Pixel operation parameters
    norm_mean_ptr,  # Normalization mean [C]
    norm_std_ptr,   # Normalization std [C]
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
    THE ULTIMATE FUSED KERNEL: All augmentations in ONE pass.
    
    This kernel combines:
    - **Geometric Tier**: RandomCrop + RandomHorizontalFlip (index transformations)
    - **Pixel Tier**: Brightness + Contrast + Saturation + Normalize (value operations)
    
    **OPTIMIZATION**: Uses TWO processing paths based on saturation:
    - LINEAR path (no saturation): Process N*C*H*W pixels individually → FASTEST
    - SPATIAL path (with saturation): Process N*H*W locations with RGB triplets → Required for grayscale
    
    **Processing Flow**:
    1. Calculate output position (n, c, h, w)
    2. Apply geometric transforms (crop + optional flip) → get input position
    3. Single memory read from transformed input position
    4. Apply all pixel operations in registers:
       - Brightness (multiplicative)
       - Contrast (fast centered scaling)
       - Saturation (blend with grayscale) [if enabled]
       - Normalize (per-channel)
    5. Single memory write to output
    
    **Zero intermediate global memory access!**
    
    Args:
        input_ptr: Input tensor [N, C, input_H, input_W]
        output_ptr: Output tensor [N, C, output_H, output_W]
        batch_size, channels: Batch and channel dimensions
        input_height, input_width: Input image dimensions
        output_height, output_width: Output (cropped) dimensions
        top_offset, left_offset: Crop offsets
        do_flip: 1 to apply horizontal flip, 0 to skip
        norm_mean_ptr, norm_std_ptr: Normalization parameters
        brightness_factor, contrast_factor, saturation_factor: Adjustment factors
        apply_* flags: Compile-time constants to enable/disable operations
        BLOCK_SIZE: Elements to process per thread block
    
    Performance:
        - ~3-5x faster than torchvision Compose (sequential operations)
        - No intermediate memory buffers
        - Single kernel launch for entire pipeline
    """
    output_spatial_size = output_height * output_width
    
    if apply_saturation:
        # ========================================================================
        # SPATIAL PROCESSING PATH (when saturation is needed)
        # ========================================================================
        # Process N*H*W spatial positions, loading RGB triplets together
        # Required for saturation (needs grayscale = 0.2989*R + 0.587*G + 0.114*B)
        
        total_spatial = batch_size * output_spatial_size
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        spatial_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        spatial_mask = spatial_offsets < total_spatial
        
        # Calculate output batch and spatial indices
        batch_idx = spatial_offsets // output_spatial_size
        spatial_idx = spatial_offsets % output_spatial_size
        h_out = spatial_idx // output_width
        w_out = spatial_idx % output_width
        
        # === GEOMETRIC TIER: Apply flip and crop ===
        if do_flip:
            w_out_transformed = output_width - 1 - w_out
        else:
            w_out_transformed = w_out
        
        h_in = h_out + top_offset
        w_in = w_out_transformed + left_offset
        
        # Calculate input offsets for RGB channels
        input_spatial_size = input_height * input_width
        r_in_offset = batch_idx * channels * input_spatial_size + 0 * input_spatial_size + h_in * input_width + w_in
        g_in_offset = batch_idx * channels * input_spatial_size + 1 * input_spatial_size + h_in * input_width + w_in
        b_in_offset = batch_idx * channels * input_spatial_size + 2 * input_spatial_size + h_in * input_width + w_in
        
        # Single memory read (from transformed geometric position)
        r = tl.load(input_ptr + r_in_offset, mask=spatial_mask, other=0.0)
        g = tl.load(input_ptr + g_in_offset, mask=spatial_mask, other=0.0)
        b = tl.load(input_ptr + b_in_offset, mask=spatial_mask, other=0.0)
        
        # === PIXEL TIER: Apply all color operations ===
        
        # Brightness
        if apply_brightness:
            r = r * brightness_factor
            g = g * brightness_factor
            b = b * brightness_factor
            r = tl.maximum(0.0, tl.minimum(1.0, r))
            g = tl.maximum(0.0, tl.minimum(1.0, g))
            b = tl.maximum(0.0, tl.minimum(1.0, b))
        
        # Contrast (fast)
        if apply_contrast:
            r = (r - 0.5) * contrast_factor + 0.5
            g = (g - 0.5) * contrast_factor + 0.5
            b = (b - 0.5) * contrast_factor + 0.5
            r = tl.maximum(0.0, tl.minimum(1.0, r))
            g = tl.maximum(0.0, tl.minimum(1.0, g))
            b = tl.maximum(0.0, tl.minimum(1.0, b))
        
        # Saturation (grayscale blend)
        gray = 0.2989 * r + 0.587 * g + 0.114 * b
        r = r * saturation_factor + gray * (1.0 - saturation_factor)
        g = g * saturation_factor + gray * (1.0 - saturation_factor)
        b = b * saturation_factor + gray * (1.0 - saturation_factor)
        r = tl.maximum(0.0, tl.minimum(1.0, r))
        g = tl.maximum(0.0, tl.minimum(1.0, g))
        b = tl.maximum(0.0, tl.minimum(1.0, b))
        
        # Normalize
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
        
        # Calculate output offsets
        r_out_offset = batch_idx * channels * output_spatial_size + 0 * output_spatial_size + spatial_idx
        g_out_offset = batch_idx * channels * output_spatial_size + 1 * output_spatial_size + spatial_idx
        b_out_offset = batch_idx * channels * output_spatial_size + 2 * output_spatial_size + spatial_idx
        
        # Single memory write
        tl.store(output_ptr + r_out_offset, r, mask=spatial_mask)
        tl.store(output_ptr + g_out_offset, g, mask=spatial_mask)
        tl.store(output_ptr + b_out_offset, b, mask=spatial_mask)
    
    else:
        # ========================================================================
        # LINEAR PROCESSING PATH (when saturation is NOT needed)
        # ========================================================================
        # Process N*C*H*W pixels individually - FASTEST path
        # Simpler indexing, better coalescing, less register pressure
        
        total_elements = batch_size * channels * output_spatial_size
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        # Decompose output offset into (n, c, h, w)
        n = offsets // (channels * output_spatial_size)
        remainder = offsets % (channels * output_spatial_size)
        c = remainder // output_spatial_size
        remainder = remainder % output_spatial_size
        h_out = remainder // output_width
        w_out = remainder % output_width
        
        # === GEOMETRIC TIER: Apply flip and crop ===
        if do_flip:
            w_out_transformed = output_width - 1 - w_out
        else:
            w_out_transformed = w_out
        
        h_in = h_out + top_offset
        w_in = w_out_transformed + left_offset
        
        # Calculate input offset
        input_spatial_size = input_height * input_width
        input_offset = (
            n * channels * input_spatial_size +
            c * input_spatial_size +
            h_in * input_width +
            w_in
        )
        
        # Single memory read (from transformed geometric position)
        pixel = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # === PIXEL TIER: Apply all color operations ===
        
        # Brightness
        if apply_brightness:
            pixel = pixel * brightness_factor
            pixel = tl.maximum(0.0, tl.minimum(1.0, pixel))
        
        # Contrast (fast)
        if apply_contrast:
            pixel = (pixel - 0.5) * contrast_factor + 0.5
            pixel = tl.maximum(0.0, tl.minimum(1.0, pixel))
        
        # Normalize
        if apply_normalize:
            channel_idx = c
            norm_mean = tl.load(norm_mean_ptr + channel_idx, mask=mask, other=0.0)
            norm_std = tl.load(norm_std_ptr + channel_idx, mask=mask, other=1.0)
            pixel = (pixel - norm_mean) / norm_std
        
        # Single memory write
        tl.store(output_ptr + offsets, pixel, mask=mask)


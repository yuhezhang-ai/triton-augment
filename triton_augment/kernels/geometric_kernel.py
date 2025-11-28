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


@triton.jit
def sample_bilinear(
    input_ptr,
    x_in, y_in,
    base_offset,
    input_width, input_height,
    fill_value,
    mask
):
    """
    Sample a pixel using bilinear interpolation.

    Args:
        input_ptr: Pointer to input tensor
        x_in, y_in: Input coordinates (float32)
        base_offset: Base offset for this batch/channel
        input_width, input_height: Image dimensions
        fill_value: Value for out-of-bounds pixels
        mask: Validity mask for this thread

    Returns:
        Interpolated pixel value
    """
    # Get integer coordinates (floor)
    x0 = tl.math.floor(x_in).to(tl.int32)
    y0 = tl.math.floor(y_in).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Get fractional parts
    dx = x_in - x0
    dy = y_in - y0

    # Check bounds for 4 neighbors
    valid_x0 = (x0 >= 0) & (x0 < input_width)
    valid_y0 = (y0 >= 0) & (y0 < input_height)
    valid_x1 = (x1 >= 0) & (x1 < input_width)
    valid_y1 = (y1 >= 0) & (y1 < input_height)

    # Load 4 neighbors with fill_value for out-of-bounds pixels
    # Top-left (x0, y0)
    mask_00 = mask & valid_x0 & valid_y0
    idx_00 = base_offset + y0 * input_width + x0
    p00 = tl.load(input_ptr + idx_00, mask=mask_00, other=fill_value)

    # Top-right (x1, y0)
    mask_10 = mask & valid_x1 & valid_y0
    idx_10 = base_offset + y0 * input_width + x1
    p10 = tl.load(input_ptr + idx_10, mask=mask_10, other=fill_value)

    # Bottom-left (x0, y1)
    mask_01 = mask & valid_x0 & valid_y1
    idx_01 = base_offset + y1 * input_width + x0
    p01 = tl.load(input_ptr + idx_01, mask=mask_01, other=fill_value)

    # Bottom-right (x1, y1)
    mask_11 = mask & valid_x1 & valid_y1
    idx_11 = base_offset + y1 * input_width + x1
    p11 = tl.load(input_ptr + idx_11, mask=mask_11, other=fill_value)

    # Bilinear interpolation weights
    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    # Perform bilinear interpolation
    return p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11


@triton.jit
def sample_nearest(
    input_ptr,
    x_in, y_in,
    base_offset,
    input_width, input_height,
    fill_value,
    mask
):
    """
    Sample a pixel using nearest neighbor interpolation.

    Args:
        input_ptr: Pointer to input tensor
        x_in, y_in: Input coordinates (float32)
        base_offset: Base offset for this batch/channel
        input_width, input_height: Image dimensions
        fill_value: Value for out-of-bounds pixels
        mask: Validity mask for this thread

    Returns:
        Nearest pixel value
    """
    # Round to nearest integer coordinates
    # Use floor(x + 0.5) to implement rounding
    x_nearest = tl.math.floor(x_in + 0.5).to(tl.int32)
    y_nearest = tl.math.floor(y_in + 0.5).to(tl.int32)

    # Check bounds
    valid = (x_nearest >= 0) & (x_nearest < input_width) & (y_nearest >= 0) & (y_nearest < input_height)

    # Load nearest pixel
    mask_valid = mask & valid
    idx = base_offset + y_nearest * input_width + x_nearest
    return tl.load(input_ptr + idx, mask=mask_valid, other=fill_value)


@triton.jit
def affine_transform_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    matrix_ptr,       # Pointer to inverse affine matrices [N, 6] (float32)
    fill_value,       # Constant value for out-of-bounds pixels
    interpolation_mode,  # 0: nearest, 1: bilinear
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply affine transformation to a batch of images.
    
    The transformation is defined by a 2x3 matrix M (inverse affine matrix).
    For each output pixel (x', y'), we compute the input coordinate (x, y):
        [x, y, 1]^T = M * [x', y', 1]^T
    
    Then we sample the input image at (x, y) using bilinear interpolation.
    
    Args:
        input_ptr: Pointer to input tensor [N, C, H_in, W_in]
        output_ptr: Pointer to output tensor [N, C, H_out, W_out]
        batch_size: Number of images
        channels: Number of channels
        input_height, input_width: Input dimensions
        output_height, output_width: Output dimensions
        matrix_ptr: Pointer to inverse affine matrices [N, 6]
                    Layout: [a, b, c, d, e, f] corresponding to:
                    [[a, b, c],
                     [d, e, f]]
        fill_value: Value to use for out-of-bounds pixels
        BLOCK_SIZE: Block size for Triton kernel
    """
    # Total number of elements in output
    total_elements = batch_size * channels * output_height * output_width
    
    # Get program ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose flat offset into (n, c, h_out, w_out)
    output_spatial_size = output_height * output_width
    n = offsets // (channels * output_spatial_size)
    remainder = offsets % (channels * output_spatial_size)
    c = remainder // output_spatial_size
    remainder = remainder % output_spatial_size
    y_out = remainder // output_width  # Row index (height)
    x_out = remainder % output_width   # Column index (width)
    
    # Load affine matrix for this image
    # Matrix layout: [N, 6] where each row is [a, b, c, d, e, f]
    # Note: Threads may access different matrix rows (non-coalesced), but the small
    # size (24 bytes) and GPU caching make this acceptable.

    
    # Pointer arithmetic for matrix
    # matrix_ptr + n * 6 + offset
    mat_base = matrix_ptr + n * 6
    
    # Load matrix elements
    a = tl.load(mat_base + 0, mask=mask, other=1.0)
    b = tl.load(mat_base + 1, mask=mask, other=0.0)
    c_tx = tl.load(mat_base + 2, mask=mask, other=0.0)
    d = tl.load(mat_base + 3, mask=mask, other=0.0)
    e = tl.load(mat_base + 4, mask=mask, other=1.0)
    f_ty = tl.load(mat_base + 5, mask=mask, other=0.0)
    
    # Calculate input coordinates
    # The matrix from _get_inverse_affine_matrix is designed for torchvision's grid-based system:
    # 1. base_grid coords are centered: x in [-w*0.5 + 0.5, w*0.5 - 0.5]
    # 2. Matrix is rescaled by [0.5*w, 0.5*h] before matmul
    # 3. Result is normalized coords for grid_sample with align_corners=False
    #
    # To match this, we:
    # 1. Convert output pixel coords to centered coords
    # 2. Apply matrix with rescaling
    # 3. Convert normalized result back to pixel coords

    # Cast to float32 for calculation
    x_out_f = x_out.to(tl.float32)
    y_out_f = y_out.to(tl.float32)

    # Step 1: Convert to centered coordinates (matching torchvision's base_grid)
    half_w = output_width * 0.5
    half_h = output_height * 0.5
    x_centered = x_out_f - half_w + 0.5
    y_centered = y_out_f - half_h + 0.5

    # Step 2: Apply matrix with rescaling (as torchvision does)
    # torchvision: rescaled_theta = theta.T / [0.5*w, 0.5*h]
    # output_grid = base_grid @ rescaled_theta
    # This means: x_norm = (a*x + b*y + c) / (0.5*w)
    #             y_norm = (d*x + e*y + f) / (0.5*h)
    x_norm = (a * x_centered + b * y_centered + c_tx) / half_w
    y_norm = (d * x_centered + e * y_centered + f_ty) / half_h

    # Step 3: Convert normalized coords to input pixel coords
    # grid_sample with align_corners=False:
    # pixel = ((normalized + 1) * size - 1) / 2
    # For input image of same size as output:
    x_in = ((x_norm + 1.0) * input_width - 1.0) * 0.5
    y_in = ((y_norm + 1.0) * input_height - 1.0) * 0.5

    input_batch_offset = n * channels * input_height * input_width
    input_channel_offset = c * input_height * input_width
    base_offset = input_batch_offset + input_channel_offset

    # Sample pixel using appropriate interpolation method
    if interpolation_mode == 0:  # Nearest neighbor
        pixel = sample_nearest(
            input_ptr, x_in, y_in, base_offset,
            input_width, input_height, fill_value, mask
        )
    else:  # Bilinear interpolation (mode == 1)
        pixel = sample_bilinear(
            input_ptr, x_in, y_in, base_offset,
            input_width, input_height, fill_value, mask
        )

    # Store result
    tl.store(output_ptr + offsets, pixel, mask=mask)
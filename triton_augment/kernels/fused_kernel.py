"""
THE ULTIMATE FUSED KERNEL - Single kernel for ALL augmentations.

This module contains the ultimate fused kernel that combines geometric transformations
(crop, flip) with pixel operations (brightness, contrast, saturation, grayscale, normalize).

This is the ONLY fused kernel in the codebase - all fused operations use this kernel
with appropriate parameters.
"""

import triton
import triton.language as tl
from ..config import ENABLE_AUTOTUNE
from .geometric_kernel import sample_nearest, sample_bilinear, apply_fused_geometric_transform


# Default configuration when auto-tuning is disabled
DEFAULT_CONFIG = triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3)

# Multiple configurations to search when auto-tuning is enabled
# Expanded search space for better stability across workloads and GPUs
AUTOTUNE_CONFIGS = [
    # Small block sizes - Good for smaller images/batches, lower register pressure
    triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
    
    # Medium block sizes - Balanced approach
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
    
    # Large block sizes - Good for large images/batches
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),  # Robust default
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
    
    # Very large block sizes - For maximum throughput on big workloads
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
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
    key=['N'],  # Tune based on total elements (batch_size * channels * height * width)
)
@triton.jit
def fused_augment_kernel(
    # Pointers
    input_ptr,
    output_ptr,
    # Auto-tuning key
    N,  # Total elements (batch_size * channels * output_height * output_width)
        # Used as auto-tuning key, not directly in computation
    # Geometric parameters (per-image)
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    # Crop/Flip parameters (always used - for fused geometric transform)
    top_offsets_ptr,      # Per-image top offsets [N] (float32)
    left_offsets_ptr,     # Per-image left offsets [N] (float32)
    flip_mask_ptr,        # Per-image flip decisions [N] (bool)
    # Affine parameters (used when has_affine=True)
    matrix_ptr,           # Per-image affine matrix [N, 6] (float32)
    fill_value,           # Fill value for out-of-bounds
    interpolation_mode,   # 0: nearest, 1: bilinear
    # Pixel operation parameters (per-image)
    norm_mean_ptr,  # Normalization mean [C]
    norm_std_ptr,   # Normalization std [C]
    brightness_factors_ptr,  # Per-image brightness factors [N]
    contrast_factors_ptr,    # Per-image contrast factors [N]
    saturation_factors_ptr,  # Per-image saturation factors [N]
    grayscale_mask_ptr,      # Per-image grayscale decisions [N] (uint8: 0 or 1)
    # Flags
    has_affine: tl.constexpr,  # Whether to apply affine transform (in addition to crop/flip)
    apply_brightness: tl.constexpr,
    apply_contrast: tl.constexpr,
    apply_saturation: tl.constexpr,
    apply_grayscale: tl.constexpr,  # Whether to check grayscale mask at all
    apply_normalize: tl.constexpr,
    # Block size (auto-tuned)
    BLOCK_SIZE: tl.constexpr,
):
    """
    FUSED AUGMENT KERNEL: All augmentations in ONE pass.
    
    This unified kernel combines:
    - **Geometric Tier**: Affine Transform (controlled by has_affine flag) + Crop + Flip
    - **Pixel Tier**: Brightness + Contrast + Saturation + Random Grayscale + Normalize (value operations)
    
    **OPTIMIZATION**: Uses TWO processing paths based on saturation/grayscale:
    - LINEAR path (no saturation/grayscale): Process N*C*H*W pixels individually → FASTEST
    - SPATIAL path (with saturation/grayscale): Process N*H*W locations with RGB triplets → Required for grayscale
    
    **Processing Flow**:
    1. Calculate output position (n, c, h, w)
    2. Apply geometric transforms:
       - If has_affine=False: Crop + optional Flip (simple integer ops)
       - If has_affine=True: Affine transform with interpolation + Crop + optional Flip
    3. Sample pixel from transformed input position
    4. Apply all pixel operations in registers:
       - Brightness (multiplicative)
       - Contrast (fast centered scaling)
       - Saturation (blend with grayscale) [if enabled]
       - Random Grayscale (per-image conversion) [if enabled]
       - Normalize (per-channel)
    5. Single memory write to output
    
    **Zero intermediate global memory access!**
    
    **Auto-Tuning**:
    - When ENABLE_AUTOTUNE=True: Tests multiple BLOCK_SIZE configs
    - When ENABLE_AUTOTUNE=False: Uses single default config (1024×4w×3s)
    - Auto-tune key: N (total output elements)
    
    Args:
        input_ptr: Input tensor [N, C, input_H, input_W]
        output_ptr: Output tensor [N, C, output_H, output_W]
        N: Total output elements (batch_size * channels * output_H * output_W)
        batch_size, channels: Batch and channel dimensions
        input_height, input_width: Input image dimensions
        output_height, output_width: Output (cropped) dimensions
        top_offsets_ptr: Per-image crop top offsets [N] (used when has_affine=False)
        left_offsets_ptr: Per-image crop left offsets [N] (used when has_affine=False)
        flip_mask_ptr: Per-image flip decisions [N] (used when has_affine=False)
        matrix_ptr: Per-image affine matrices [N, 6] (used when has_affine=True)
        fill_value: Fill value for out-of-bounds (used when has_affine=True)
        interpolation_mode: 0=nearest, 1=bilinear (used when has_affine=True)
        norm_mean_ptr, norm_std_ptr: Normalization parameters [C]
        brightness_factors_ptr: Per-image brightness factors [N]
        contrast_factors_ptr: Per-image contrast factors [N]
        saturation_factors_ptr: Per-image saturation factors [N]
        grayscale_mask_ptr: Per-image grayscale decisions [N]
        has_affine: Compile-time flag - use affine (True) or crop/flip (False)
        apply_* flags: Compile-time constants to enable/disable operations
        BLOCK_SIZE: Elements to process per thread block (auto-tuned)
    
    Performance:
        - Up to 12x faster on large images
        - No intermediate memory buffers
        - Single kernel launch for entire pipeline
        - Compile-time branching (has_affine) has zero runtime overhead
    """
    output_spatial_size = output_height * output_width
    
    if apply_saturation or apply_grayscale:
        # ========================================================================
        # SPATIAL PROCESSING PATH (when saturation or grayscale is needed)
        # ========================================================================
        # Process N*H*W spatial positions, loading RGB triplets together
        # Required for saturation/grayscale (needs grayscale = 0.2989*R + 0.587*G + 0.114*B)
        
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
        
        # === GEOMETRIC TIER: Apply fused transform (Flip -> Crop -> Affine) ===
        # Load per-image geometric parameters
        top_offset = tl.load(top_offsets_ptr + batch_idx, mask=spatial_mask, other=0.0)
        left_offset = tl.load(left_offsets_ptr + batch_idx, mask=spatial_mask, other=0.0)
        do_flip = tl.load(flip_mask_ptr + batch_idx, mask=spatial_mask, other=False)
        
        # Apply fused geometric transform: Flip -> Crop offset -> Affine
        x_in, y_in = apply_fused_geometric_transform(
            w_out, h_out,
            top_offset, left_offset,
            output_width,
            do_flip,
            matrix_ptr,
            batch_idx,
            input_width, input_height,
            has_affine,
            spatial_mask
        )
        
        # Calculate base offsets for RGB channels
        input_spatial_size = input_height * input_width
        r_base = batch_idx * channels * input_spatial_size + 0 * input_spatial_size
        g_base = batch_idx * channels * input_spatial_size + 1 * input_spatial_size
        b_base = batch_idx * channels * input_spatial_size + 2 * input_spatial_size
        
        if has_affine:
            # Sample RGB channels with interpolation (affine may produce non-integer coords)
            if interpolation_mode == 0:  # Nearest
                r = sample_nearest(input_ptr, x_in, y_in, r_base, input_width, input_height, fill_value, spatial_mask)
                g = sample_nearest(input_ptr, x_in, y_in, g_base, input_width, input_height, fill_value, spatial_mask)
                b = sample_nearest(input_ptr, x_in, y_in, b_base, input_width, input_height, fill_value, spatial_mask)
            else:  # Bilinear
                r = sample_bilinear(input_ptr, x_in, y_in, r_base, input_width, input_height, fill_value, spatial_mask)
                g = sample_bilinear(input_ptr, x_in, y_in, g_base, input_width, input_height, fill_value, spatial_mask)
                b = sample_bilinear(input_ptr, x_in, y_in, b_base, input_width, input_height, fill_value, spatial_mask)
        else:
            # Direct load (crop/flip produces integer coords)
            x_in_int = x_in.to(tl.int32)
            y_in_int = y_in.to(tl.int32)
            r_in_offset = r_base + y_in_int * input_width + x_in_int
            g_in_offset = g_base + y_in_int * input_width + x_in_int
            b_in_offset = b_base + y_in_int * input_width + x_in_int
            
            r = tl.load(input_ptr + r_in_offset, mask=spatial_mask, other=0.0)
            g = tl.load(input_ptr + g_in_offset, mask=spatial_mask, other=0.0)
            b = tl.load(input_ptr + b_in_offset, mask=spatial_mask, other=0.0)
        
        # === PIXEL TIER: Apply all color operations ===
        
        # Load per-image factors
        brightness_factor = tl.load(brightness_factors_ptr + batch_idx, mask=spatial_mask, other=1.0)
        contrast_factor = tl.load(contrast_factors_ptr + batch_idx, mask=spatial_mask, other=1.0)
        saturation_factor = tl.load(saturation_factors_ptr + batch_idx, mask=spatial_mask, other=1.0)
        
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
        if apply_saturation:
            gray = 0.2989 * r + 0.587 * g + 0.114 * b
            r = r * saturation_factor + gray * (1.0 - saturation_factor)
            g = g * saturation_factor + gray * (1.0 - saturation_factor)
            b = b * saturation_factor + gray * (1.0 - saturation_factor)
            r = tl.maximum(0.0, tl.minimum(1.0, r))
            g = tl.maximum(0.0, tl.minimum(1.0, g))
            b = tl.maximum(0.0, tl.minimum(1.0, b))
        
        # Random grayscale conversion (applied AFTER saturation, per-image)
        if apply_grayscale:
            # Load per-image grayscale decision (bool tensor)
            do_grayscale = tl.load(grayscale_mask_ptr + batch_idx, mask=spatial_mask, other=False)
            gray = 0.2989 * r + 0.587 * g + 0.114 * b
            r = tl.where(do_grayscale, gray, r)
            g = tl.where(do_grayscale, gray, g)
            b = tl.where(do_grayscale, gray, b)
        
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
        
        # === GEOMETRIC TIER: Apply fused transform (Flip -> Crop -> Affine) ===
        # Load per-image geometric parameters
        top_offset = tl.load(top_offsets_ptr + n, mask=mask, other=0.0)
        left_offset = tl.load(left_offsets_ptr + n, mask=mask, other=0.0)
        do_flip = tl.load(flip_mask_ptr + n, mask=mask, other=False)
        
        # Apply fused geometric transform: Flip -> Crop offset -> Affine
        x_in, y_in = apply_fused_geometric_transform(
            w_out, h_out,
            top_offset, left_offset,
            output_width,
            do_flip,
            matrix_ptr,
            n,
            input_width, input_height,
            has_affine,
            mask
        )
        
        # Calculate base offset for this channel
        input_spatial_size = input_height * input_width
        base_offset = n * channels * input_spatial_size + c * input_spatial_size
        
        if has_affine:
            # Sample pixel with interpolation (affine may produce non-integer coords)
            if interpolation_mode == 0:  # Nearest
                pixel = sample_nearest(input_ptr, x_in, y_in, base_offset, input_width, input_height, fill_value, mask)
            else:  # Bilinear
                pixel = sample_bilinear(input_ptr, x_in, y_in, base_offset, input_width, input_height, fill_value, mask)
        else:
            # Direct load (crop/flip produces integer coords)
            x_in_int = x_in.to(tl.int32)
            y_in_int = y_in.to(tl.int32)
            input_offset = base_offset + y_in_int * input_width + x_in_int
            pixel = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # === PIXEL TIER: Apply all color operations ===
        
        # Load per-image factors
        brightness_factor = tl.load(brightness_factors_ptr + n, mask=mask, other=1.0)
        contrast_factor = tl.load(contrast_factors_ptr + n, mask=mask, other=1.0)
        
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

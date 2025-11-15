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
def ultimate_fused_kernel(
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
    top_offsets_ptr,      # Per-image top offsets [N] (int32)
    left_offsets_ptr,     # Per-image left offsets [N] (int32)
    flip_mask_ptr,        # Per-image flip decisions [N] (uint8: 0 or 1)
    # Pixel operation parameters (per-image)
    norm_mean_ptr,  # Normalization mean [C]
    norm_std_ptr,   # Normalization std [C]
    brightness_factors_ptr,  # Per-image brightness factors [N]
    contrast_factors_ptr,    # Per-image contrast factors [N]
    saturation_factors_ptr,  # Per-image saturation factors [N]
    grayscale_mask_ptr,      # Per-image grayscale decisions [N] (uint8: 0 or 1)
    # Flags
    apply_brightness: tl.constexpr,
    apply_contrast: tl.constexpr,
    apply_saturation: tl.constexpr,
    apply_grayscale: tl.constexpr,  # Whether to check grayscale mask at all
    apply_normalize: tl.constexpr,
    # Block size (auto-tuned)
    BLOCK_SIZE: tl.constexpr,
):
    """
    THE ULTIMATE FUSED KERNEL: All augmentations in ONE pass.
    
    This kernel combines:
    - **Geometric Tier**: RandomCrop + RandomHorizontalFlip (index transformations)
    - **Pixel Tier**: Brightness + Contrast + Saturation + Random Grayscale + Normalize (value operations)
    
    **OPTIMIZATION**: Uses TWO processing paths based on saturation/grayscale:
    - LINEAR path (no saturation/grayscale): Process N*C*H*W pixels individually → FASTEST
    - SPATIAL path (with saturation/grayscale): Process N*H*W locations with RGB triplets → Required for grayscale
    
    **Processing Flow**:
    1. Calculate output position (n, c, h, w)
    2. Apply geometric transforms (crop + optional flip) → get input position
    3. Single memory read from transformed input position
    4. Apply all pixel operations in registers:
       - Brightness (multiplicative)
       - Contrast (fast centered scaling)
       - Saturation (blend with grayscale) [if enabled]
       - Random Grayscale (per-image conversion) [if enabled]
       - Normalize (per-channel)
    5. Single memory write to output
    
    **Zero intermediate global memory access!**
    
    **Auto-Tuning**:
    - When ENABLE_AUTOTUNE=True: Tests 4 BLOCK_SIZE configs (256, 1024×4w×2s, 1024×8w×3s, 1024×4w×4s)
    - When ENABLE_AUTOTUNE=False: Uses single default config (1024×4w×3s)
    - Auto-tune key: N (total output elements)
    
    **Usage for Different Fused Operations**:
    - Crop+Flip: Set crop/flip params, disable all pixel ops
    - Color+Normalize: Set input=output size, no crop/flip, enable pixel ops
    - Ultimate (All): Enable everything
    
    Args:
        input_ptr: Input tensor [N, C, input_H, input_W]
        output_ptr: Output tensor [N, C, output_H, output_W]
        N: Total output elements (batch_size * channels * output_H * output_W)
           Used as auto-tuning key, not directly in computation
        batch_size, channels: Batch and channel dimensions
        input_height, input_width: Input image dimensions
        output_height, output_width: Output (cropped) dimensions
        top_offsets_ptr: Per-image crop top offsets [N] (int32)
        left_offsets_ptr: Per-image crop left offsets [N] (int32)
        flip_mask_ptr: Per-image flip decisions [N] (uint8: 0=no flip, 1=flip)
        norm_mean_ptr, norm_std_ptr: Normalization parameters [C]
        brightness_factors_ptr: Per-image brightness factors [N]
        contrast_factors_ptr: Per-image contrast factors [N]
        saturation_factors_ptr: Per-image saturation factors [N]
        grayscale_mask_ptr: Per-image grayscale decisions [N] (uint8: 0=color, 1=gray)
        apply_* flags: Compile-time constants to enable/disable operations
        BLOCK_SIZE: Elements to process per thread block (auto-tuned)
    
    Performance:
        - Up to 12x faster on large images (8.1x average on Tesla T4, scales dramatically with image size)
        - No intermediate memory buffers
        - Single kernel launch for entire pipeline
        - Auto-tuning provides optimal configuration per GPU architecture
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
        
        # === GEOMETRIC TIER: Apply flip and crop (per-image) ===
        # Load per-image geometric parameters
        top_offset = tl.load(top_offsets_ptr + batch_idx, mask=spatial_mask, other=0)
        left_offset = tl.load(left_offsets_ptr + batch_idx, mask=spatial_mask, other=0)
        do_flip = tl.load(flip_mask_ptr + batch_idx, mask=spatial_mask, other=False)
        
        # Apply flip conditionally
        w_out_transformed = tl.where(do_flip, output_width - 1 - w_out, w_out)
        
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
        
        # === GEOMETRIC TIER: Apply flip and crop (per-image) ===
        # Load per-image geometric parameters
        top_offset = tl.load(top_offsets_ptr + n, mask=mask, other=0)
        left_offset = tl.load(left_offsets_ptr + n, mask=mask, other=0)
        do_flip = tl.load(flip_mask_ptr + n, mask=mask, other=False)
        
        # Apply flip conditionally
        w_out_transformed = tl.where(do_flip, output_width - 1 - w_out, w_out)
        
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


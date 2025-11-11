"""
Functional API for Triton-accelerated image transformations.

This module provides PyTorch-style functional interfaces that wrap the raw
Triton kernels for ease of use and input validation.

Implementation matches torchvision.transforms.v2.functional exactly.
Reference: torchvision/transforms/v2/functional/_color.py

Author: yuhezhang-ai
"""

import torch
import triton
import sys
from .kernels.color_normalize_kernel import (
    fused_color_normalize_kernel,
    brightness_kernel,
    contrast_kernel,
    contrast_fast_kernel,
    saturation_kernel,
    normalize_kernel,
)
from .kernels.geometric_kernel import (
    crop_kernel,
    horizontal_flip_kernel,
    fused_crop_flip_kernel,
    ultimate_fused_kernel,
)
from .utils import should_show_autotune_message
from .config import ENABLE_AUTOTUNE


def _validate_image_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Validate that the input is a valid image tensor.
    
    Args:
        tensor: Input tensor to validate
        name: Name of the tensor for error messages
        
    Raises:
        TypeError: If tensor is not a torch.Tensor
        ValueError: If tensor is not on CUDA or has invalid shape
    
    Note:
        Currently only supports 4D tensors (N, C, H, W).
        Future versions will support:
        - 3D tensors (C, H, W) for single images
        - 5D tensors (N, T, C, H, W) for video batches
        This matches torchvision's broader dimension support.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA device")
    
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must be a 4D tensor with shape (N, C, H, W), "
            f"got shape {tensor.shape}"
        )
    
    c = tensor.shape[1]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")


def _rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to grayscale using torchvision's formula.
    
    Reference: torchvision/transforms/v2/functional/_color.py line 42
    Formula: 0.2989*R + 0.587*G + 0.114*B
    """
    r, g, b = image.unbind(dim=-3)
    # Exact formula from torchvision: r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    l_img = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    return l_img.unsqueeze(dim=-3)


def rgb_to_grayscale(
    image: torch.Tensor,
    num_output_channels: int = 1,
) -> torch.Tensor:
    """
    Convert RGB image to grayscale.
    
    Matches torchvision.transforms.v2.functional.rgb_to_grayscale exactly.
    Uses weights: 0.2989*R + 0.587*G + 0.114*B
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
               where C must be 3 (RGB)
        num_output_channels: Number of output channels (1 or 3)
                            If 3, grayscale is replicated across channels
                            
    Returns:
        Grayscale tensor of shape (N, num_output_channels, H, W)
        
    Raises:
        ValueError: If num_output_channels not in {1, 3} or if input not RGB
        
    Example:
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> gray_1ch = rgb_to_grayscale(img, num_output_channels=1)   # Shape: (1, 1, 224, 224)
        >>> gray_3ch = rgb_to_grayscale(img, num_output_channels=3)   # Shape: (1, 3, 224, 224)
    """
    _validate_image_tensor(image, "image")
    
    if num_output_channels not in (1, 3):
        raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}")
    
    if image.shape[1] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {image.shape[1]}")
    
    # Compute grayscale using helper
    grayscale = _rgb_to_grayscale(image)  # Shape: (N, 1, H, W)
    
    # Replicate to 3 channels if requested
    if num_output_channels == 3:
        grayscale = grayscale.expand(-1, 3, -1, -1).contiguous()
    
    return grayscale


def random_grayscale(
    image: torch.Tensor,
    p: float = 0.1,
    num_output_channels: int = 3,
) -> torch.Tensor:
    """
    Randomly convert image to grayscale with probability p.
    
    Matches torchvision.transforms.v2.RandomGrayscale behavior.
    
    Args:
        image: Input image tensor of shape (N, 3, H, W) on CUDA
        p: Probability of converting to grayscale (default: 0.1)
        num_output_channels: Number of output channels (1 or 3, default: 3)
                            Usually 3 to maintain compatibility with RGB pipelines
                            
    Returns:
        Image tensor, either original or grayscale based on probability
        
    Raises:
        ValueError: If p not in [0, 1]
        
    Example:
        >>> img = torch.rand(4, 3, 224, 224, device='cuda')
        >>> result = random_grayscale(img, p=0.5)  # 50% chance of grayscale
    """
    _validate_image_tensor(image, "image")
    
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p ({p}) must be in [0, 1]")
    
    # Decide randomly whether to apply grayscale
    if p > 0 and torch.rand(1).item() < p:
        return rgb_to_grayscale(image, num_output_channels=num_output_channels)
    else:
        return image


def adjust_brightness(
    image: torch.Tensor,
    brightness_factor: float,
) -> torch.Tensor:
    """
    Adjust brightness of an image (MULTIPLICATIVE operation).
    
    Matches torchvision.transforms.v2.functional.adjust_brightness exactly.
    Reference: torchvision/transforms/v2/functional/_color.py line 114-125
    
    Formula: output = input * brightness_factor
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        brightness_factor: How much to adjust the brightness. Must be non-negative.
                          0 gives a black image, 1 gives the original image,
                          2 increases brightness by 2x.
                          
    Returns:
        Brightness-adjusted tensor of the same shape and dtype
        
    Raises:
        ValueError: If brightness_factor is negative
        
    Example:
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> bright_img = adjust_brightness(img, brightness_factor=1.2)  # 20% brighter
        >>> dark_img = adjust_brightness(img, brightness_factor=0.8)   # 20% darker
    """
    _validate_image_tensor(image, "image")
    
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")
    
    # Allocate output tensor
    output_tensor = torch.empty_like(image)
    
    # Calculate grid size
    n_elements = image.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    brightness_kernel[grid](
        image,
        output_tensor,
        n_elements,
        brightness_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor


def adjust_contrast(
    image: torch.Tensor,
    contrast_factor: float,
) -> torch.Tensor:
    """
    Adjust contrast of an image.
    
    Matches torchvision.transforms.v2.functional.adjust_contrast exactly.
    Reference: torchvision/transforms/v2/functional/_color.py line 190-206
    
    Formula: output = blend(image, grayscale_mean, contrast_factor)
           = image * contrast_factor + grayscale_mean * (1 - contrast_factor)
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        contrast_factor: How much to adjust the contrast. Must be non-negative.
                        0 gives a gray image, 1 gives the original image,
                        values > 1 increase contrast.
                          
    Returns:
        Contrast-adjusted tensor of the same shape and dtype
        
    Raises:
        ValueError: If contrast_factor is negative
        
    Example:
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> high_contrast = adjust_contrast(img, contrast_factor=1.5)
        >>> low_contrast = adjust_contrast(img, contrast_factor=0.5)
    """
    _validate_image_tensor(image, "image")
    
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
    
    # Compute grayscale mean for contrast (per-image)
    c = image.shape[1]
    if c == 3:
        grayscale_image = _rgb_to_grayscale(image)
    else:
        grayscale_image = image
    
    # Compute mean per image
    mean = torch.mean(grayscale_image, dim=(-3, -2, -1), keepdim=False)  # Shape: [N]
    
    # Allocate output tensor
    output_tensor = torch.empty_like(image)
    
    # Get tensor dimensions
    batch_size, channels, height, width = image.shape
    spatial_size = height * width
    total_elements = batch_size * channels * spatial_size
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    contrast_kernel[grid](
        image,
        output_tensor,
        mean,
        batch_size,
        channels,
        height,
        width,
        contrast_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor


def adjust_contrast_fast(
    image: torch.Tensor,
    contrast_factor: float,
) -> torch.Tensor:
    """
    Adjust contrast of an image using FAST centered scaling.
    
    This is faster than adjust_contrast() because it doesn't require computing
    the grayscale mean. Uses formula: output = (input - 0.5) * contrast_factor + 0.5
    
    NOTE: This is NOT equivalent to torchvision's adjust_contrast, but provides
    similar perceptual results and is fully fusible with other operations.
    
    Use this when:
    - You want maximum performance with fusion
    - Exact torchvision reproduction is not critical
    
    Use adjust_contrast() when:
    - You need exact torchvision compatibility
    - Reproducibility with torchvision is required
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        contrast_factor: How much to adjust the contrast. Must be non-negative.
                        0.5 decreases contrast, 1.0 gives original, >1.0 increases.
                          
    Returns:
        Contrast-adjusted tensor of the same shape and dtype
        
    Raises:
        ValueError: If contrast_factor is negative
        
    Example:
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> high_contrast = adjust_contrast_fast(img, contrast_factor=1.5)
        >>> # Use in fused operation for maximum speed
        >>> result = fused_color_normalize(img, contrast_factor=1.5, ...)
    """
    _validate_image_tensor(image, "image")
    
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
    
    # Allocate output tensor
    output_tensor = torch.empty_like(image)
    
    # Get total number of elements
    n_elements = image.numel()
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    contrast_fast_kernel[grid](
        image,
        output_tensor,
        n_elements,
        contrast_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor


def adjust_saturation(
    image: torch.Tensor,
    saturation_factor: float,
) -> torch.Tensor:
    """
    Adjust color saturation of an image.
    
    Matches torchvision.transforms.v2.functional.adjust_saturation exactly.
    Reference: torchvision/transforms/v2/functional/_color.py line 151-166
    
    Formula: output = blend(image, grayscale, saturation_factor)
           = image * saturation_factor + grayscale * (1 - saturation_factor)
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        saturation_factor: How much to adjust the saturation. Must be non-negative.
                          0 will give a grayscale image,
                          1 will give the original image,
                          values > 1 increase saturation.
                          
    Returns:
        Saturation-adjusted tensor of the same shape and dtype
        
    Raises:
        ValueError: If saturation_factor is negative
        
    Example:
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> grayscale = adjust_saturation(img, saturation_factor=0.0)
        >>> saturated = adjust_saturation(img, saturation_factor=2.0)
    """
    _validate_image_tensor(image, "image")
    
    if saturation_factor < 0:
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")
    
    c = image.shape[1]
    if c == 1:  # Match PIL behavior - grayscale images are returned as-is
        return image.clone()
    
    # Allocate output tensor
    output_tensor = torch.empty_like(image)
    
    # Get tensor dimensions
    batch_size, channels, height, width = image.shape
    spatial_size = height * width
    total_spatial_elements = batch_size * spatial_size
    
    # Calculate grid size  
    BLOCK_SIZE = 1024  # Fixed block size for simple operations
    grid = lambda meta: (triton.cdiv(total_spatial_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    saturation_kernel[grid](
        image,
        output_tensor,
        batch_size,
        channels,
        height,
        width,
        saturation_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor


def normalize(
    image: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """
    Normalize a tensor image with mean and standard deviation.
    
    This function normalizes each channel:
        output[c] = (input[c] - mean[c]) / std[c]
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        mean: Tuple of mean values for each channel (R, G, B)
        std: Tuple of standard deviation values for each channel (R, G, B)
                          
    Returns:
        Normalized tensor of the same shape and dtype
        
    Example:
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> normalized = normalize(img, 
        ...                        mean=(0.485, 0.456, 0.406),
        ...                        std=(0.229, 0.224, 0.225))
    """
    _validate_image_tensor(image, "image")
    
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must have 3 values for RGB channels")
    
    # Allocate output tensor
    output_tensor = torch.empty_like(image)
    
    # Convert mean and std to CUDA tensors
    mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype)
    
    # Calculate grid size
    n_elements = image.numel()
    batch_size, n_channels, height, width = image.shape
    spatial_size = height * width
    BLOCK_SIZE = 1024  # Fixed block size for simple operations
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    normalize_kernel[grid](
        image,
        output_tensor,
        n_elements,
        spatial_size,
        n_channels,
        mean_tensor,
        std_tensor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor


def fused_color_normalize(
    image: torch.Tensor,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
    saturation_factor: float = 1.0,
    random_grayscale_p: float = 0.0,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """
    Apply fused color jitter and normalization in a single kernel.
    
    This function fuses multiple operations:
    1. Brightness adjustment: pixel = pixel * brightness_factor [torchvision-exact]
    2. Contrast adjustment (FAST): pixel = (pixel - 0.5) * contrast + 0.5
    3. Saturation adjustment: blend with grayscale [torchvision-exact]
    4. Random grayscale: with probability p, convert to grayscale (saturation=0)
    5. Per-channel normalization [torchvision-exact]
    
    The fusion eliminates intermediate memory reads/writes, providing
    2-5x performance improvements over sequential operations.
    
    **IMPORTANT**: Contrast uses FAST centered scaling, NOT torchvision's blend-with-mean.
    For exact torchvision reproduction, use individual functions:
        adjust_brightness() + adjust_contrast() + adjust_saturation() + normalize()
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        brightness_factor: Brightness multiplier (default: 1.0 = no change)
                          Must be non-negative. 0=black, 1=original, >1=brighter
        contrast_factor: Contrast multiplier (default: 1.0 = no change) [FAST mode]
                        Must be non-negative. 0.5=low, 1=original, >1=high contrast
        saturation_factor: Saturation multiplier (default: 1.0 = no change)
                          Must be non-negative. 0=grayscale, 1=original, >1=more saturated
        random_grayscale_p: Probability of converting to grayscale (default: 0.0 = no grayscale)
                           Must be in [0, 1]. When triggered, overrides saturation to 0.
        mean: Tuple of mean values for normalization (R, G, B)
              If None, normalization is skipped
        std: Tuple of standard deviation values for normalization (R, G, B)
             If None, normalization is skipped
                          
    Returns:
        Transformed tensor of the same shape and dtype
        
    Raises:
        ValueError: If any factor is negative
        
    Example:
        >>> img = torch.rand(2, 3, 224, 224, device='cuda')
        >>> # Fast fused pipeline
        >>> augmented = fused_color_normalize(
        ...     img,
        ...     brightness_factor=1.2,
        ...     contrast_factor=1.1,      # Uses FAST contrast
        ...     saturation_factor=0.9,
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225)
        ... )
        
    Notes:
        - Brightness and saturation match torchvision exactly
        - Contrast uses FAST method for fusion efficiency
        - Operations are performed in a single GPU kernel
        - For torchvision-exact contrast, use adjust_contrast() separately
    """
    _validate_image_tensor(image, "image")
    
    # Validate factors
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
    if saturation_factor < 0:
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")
    if not (0.0 <= random_grayscale_p <= 1.0):
        raise ValueError(f"random_grayscale_p ({random_grayscale_p}) must be in [0, 1].")
    
    # Handle random grayscale by overriding saturation to 0
    # This converts to grayscale using the existing saturation kernel
    if random_grayscale_p > 0 and torch.rand(1).item() < random_grayscale_p:
        saturation_factor = 0.0
    
    # Determine which operations to apply
    apply_brightness = (brightness_factor != 1.0)
    apply_contrast = (contrast_factor != 1.0)
    apply_saturation = (saturation_factor != 1.0)
    apply_normalize = (mean is not None and std is not None)
    
    # If no operations, return copy
    if not (apply_brightness or apply_contrast or apply_saturation or apply_normalize):
        return image.clone()
    
    # Validate normalization parameters
    if apply_normalize:
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must have 3 values for RGB channels")
        norm_mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype)
        norm_std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype)
    else:
        # Create dummy tensors (won't be used due to compile-time flag)
        norm_mean_tensor = torch.zeros(3, device=image.device, dtype=image.dtype)
        norm_std_tensor = torch.ones(3, device=image.device, dtype=image.dtype)
    
    # Allocate output tensor
    output_tensor = torch.empty_like(image)
    
    # Get tensor dimensions
    batch_size, channels, height, width = image.shape
    spatial_size = height * width
    total_spatial_elements = batch_size * spatial_size
    total_elements = batch_size * channels * spatial_size
    
    # Calculate N for auto-tuning key
    N = total_elements
    
    # Calculate grid size based on processing path
    # - If apply_saturation=True: SPATIAL path processes N*H*W locations
    # - If apply_saturation=False: LINEAR path processes N*C*H*W elements
    if apply_saturation:
        # Spatial processing: one thread block per BLOCK_SIZE spatial locations
        grid = lambda meta: (triton.cdiv(total_spatial_elements, meta['BLOCK_SIZE']),)
    else:
        # Linear processing: one thread block per BLOCK_SIZE pixels
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Show auto-tuning message if this kernel hasn't been tuned yet (only when enabled)
    if ENABLE_AUTOTUNE and should_show_autotune_message('fused_color_normalize_kernel', (N,)):
        print(f"[Triton-Augment] Auto-tuning fused_color_normalize_kernel for batch={batch_size}, size={height}×{width}... (~2-5 sec)", 
              file=sys.stderr, flush=True)
    
    # Launch kernel (auto-tuned with 4 configs or fixed with 1 config based on ENABLE_AUTOTUNE)
    fused_color_normalize_kernel[grid](
        image,
        output_tensor,
        N,
        batch_size,
        channels,
        height,
        width,
        norm_mean_tensor,
        norm_std_tensor,
        brightness_factor,
        contrast_factor,
        saturation_factor,
        apply_brightness,
        apply_contrast,
        apply_saturation,
        apply_normalize,
    )
    
    return output_tensor


# ============================================================================
# Geometric Transformations
# ============================================================================


def crop(
    image: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Crop a rectangular region from the input image.
    
    Matches torchvision.transforms.v2.functional.crop exactly.
    Reference: torchvision/transforms/v2/functional/_geometry.py line 1787
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        top: Top pixel coordinate for cropping (vertical offset)
        left: Left pixel coordinate for cropping (horizontal offset)
        height: Height of the cropped image
        width: Width of the cropped image
        
    Returns:
        Cropped tensor of shape (N, C, height, width)
        
    Raises:
        ValueError: If crop region is invalid (out of bounds)
        
    Example:
        >>> img = torch.rand(2, 3, 224, 224, device='cuda')
        >>> # Crop center 112x112 region
        >>> cropped = crop(img, top=56, left=56, height=112, width=112)
        >>> cropped.shape
        torch.Size([2, 3, 112, 112])
    
    Note:
        For MVP, this requires valid crop coordinates (no padding).
        Future versions will support padding for out-of-bounds crops.
    """
    _validate_image_tensor(image, "image")
    
    batch_size, channels, img_height, img_width = image.shape
    
    # Validate crop region
    if top < 0 or left < 0:
        raise ValueError(f"Crop coordinates must be non-negative, got top={top}, left={left}")
    if top + height > img_height or left + width > img_width:
        raise ValueError(
            f"Crop region ({top}:{top+height}, {left}:{left+width}) "
            f"exceeds image dimensions ({img_height}, {img_width})"
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"Crop size must be positive, got height={height}, width={width}")
    
    # Allocate output tensor
    output = torch.empty(
        batch_size, channels, height, width,
        dtype=image.dtype,
        device=image.device
    )
    
    # Calculate total elements and grid size
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    crop_kernel[grid](
        image,
        output,
        batch_size,
        channels,
        img_height,
        img_width,
        height,
        width,
        top,
        left,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def center_crop(
    image: torch.Tensor,
    output_size: tuple[int, int] | int,
) -> torch.Tensor:
    """
    Crop the center of the image to the given size.
    
    Matches torchvision.transforms.v2.functional.center_crop exactly.
    Reference: torchvision/transforms/v2/functional/_geometry.py line 2545
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        output_size: Desired output size (height, width) or int for square crop
        
    Returns:
        Center-cropped tensor of shape (N, C, output_size[0], output_size[1])
        
    Raises:
        ValueError: If output_size is larger than image size
        
    Example:
        >>> img = torch.rand(2, 3, 224, 224, device='cuda')
        >>> # Center crop to 112x112
        >>> cropped = center_crop(img, (112, 112))
        >>> # or for square crop
        >>> cropped = center_crop(img, 112)
    """
    _validate_image_tensor(image, "image")
    
    # Parse output_size
    if isinstance(output_size, int):
        crop_height = crop_width = output_size
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        crop_height = crop_width = output_size[0]
    else:
        crop_height, crop_width = output_size
    
    _, _, image_height, image_width = image.shape
    
    # Validate crop size
    if crop_height > image_height or crop_width > image_width:
        raise ValueError(
            f"Crop size ({crop_height}, {crop_width}) larger than "
            f"image size ({image_height}, {image_width}). "
            f"Padding not yet supported in MVP."
        )
    
    # Calculate center crop coordinates
    crop_top = (image_height - crop_height) // 2
    crop_left = (image_width - crop_width) // 2
    
    # Use crop() functional
    return crop(image, crop_top, crop_left, crop_height, crop_width)


def horizontal_flip(image: torch.Tensor) -> torch.Tensor:
    """
    Flip the image horizontally (left to right).
    
    Matches torchvision.transforms.v2.functional.horizontal_flip exactly.
    Reference: torchvision/transforms/v2/functional/_geometry.py line 56
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        
    Returns:
        Horizontally flipped tensor of the same shape
        
    Example:
        >>> img = torch.rand(2, 3, 224, 224, device='cuda')
        >>> flipped = horizontal_flip(img)
        >>> # Verify flip: img[:, :, :, 0] == flipped[:, :, :, -1]
    
    Note:
        This uses a custom Triton kernel. For standalone flip operations,
        PyTorch's tensor.flip(-1) is highly optimized and may be comparable.
        The main benefit is when fusing with crop (see fused_crop_flip).
    """
    _validate_image_tensor(image, "image")
    
    batch_size, channels, height, width = image.shape
    
    # Allocate output tensor
    output = torch.empty_like(image)
    
    # Calculate total elements and grid size
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    horizontal_flip_kernel[grid](
        image,
        output,
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def fused_crop_flip(
    image: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    flip_horizontal: bool = False,
) -> torch.Tensor:
    """
    Fused crop + horizontal flip in a single kernel.
    
    This function combines crop and horizontal flip operations, eliminating
    intermediate memory transfers. Provides ~1.5-2x speedup over sequential operations.
    
    **OPTIMIZATION**: Uses compile-time branching (tl.constexpr):
    - When flip_horizontal=False: Only crop code is compiled (no flip overhead)
    - When flip_horizontal=True: Both crop and flip are fused
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        top: Top pixel coordinate for cropping
        left: Left pixel coordinate for cropping
        height: Height of the cropped image
        width: Width of the cropped image
        flip_horizontal: Whether to flip horizontally after cropping
        
    Returns:
        Cropped (and optionally flipped) tensor of shape (N, C, height, width)
        
    Raises:
        ValueError: If crop region is invalid
        
    Example:
        >>> img = torch.rand(2, 3, 224, 224, device='cuda')
        >>> # Crop and flip in one kernel
        >>> result = fused_crop_flip(img, 56, 56, 112, 112, flip_horizontal=True)
        >>> result.shape
        torch.Size([2, 3, 112, 112])
        
        >>> # Equivalent sequential operations (slower):
        >>> cropped = crop(img, 56, 56, 112, 112)
        >>> result_seq = horizontal_flip(cropped)
    
    Performance:
        - ~1.5-2x faster than crop() + horizontal_flip()
        - No intermediate memory buffer needed
        - Zero overhead when flip_horizontal=False (compile-time optimization)
    """
    _validate_image_tensor(image, "image")
    
    batch_size, channels, img_height, img_width = image.shape
    
    # Validate crop region
    if top < 0 or left < 0:
        raise ValueError(f"Crop coordinates must be non-negative, got top={top}, left={left}")
    if top + height > img_height or left + width > img_width:
        raise ValueError(
            f"Crop region ({top}:{top+height}, {left}:{left+width}) "
            f"exceeds image dimensions ({img_height}, {img_width})"
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"Crop size must be positive, got height={height}, width={width}")
    
    # Allocate output tensor
    output = torch.empty(
        batch_size, channels, height, width,
        dtype=image.dtype,
        device=image.device
    )
    
    # Calculate total elements and grid size
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Launch fused kernel
    fused_crop_flip_kernel[grid](
        image,
        output,
        batch_size,
        channels,
        img_height,
        img_width,
        height,
        width,
        top,
        left,
        int(flip_horizontal),  # Convert bool to int for tl.constexpr
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def ultimate_fused_augment(
    image: torch.Tensor,
    # Geometric parameters
    top: int,
    left: int,
    height: int,
    width: int,
    flip_horizontal: bool = False,
    # Pixel operation parameters
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
    saturation_factor: float = 1.0,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """
    THE ULTIMATE FUSED AUGMENTATION: ALL operations in ONE kernel.
    
    Combines geometric (crop + flip) and pixel (color + normalize) operations
    in a single GPU kernel, providing maximum performance.
    
    **Performance**: ~3-5x faster than torchvision Compose (6 sequential operations)
    
    Operations applied in sequence:
    1. **Geometric**: Crop + Optional Horizontal Flip (index transformations)
    2. **Pixel**: Brightness + Contrast (fast) + Saturation + Normalize (value operations)
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        top, left: Crop offsets
        height, width: Crop dimensions
        flip_horizontal: Whether to flip horizontally after crop
        brightness_factor: Brightness multiplier (1.0 = no change)
        contrast_factor: Contrast multiplier (1.0 = no change) [FAST mode]
        saturation_factor: Saturation multiplier (1.0 = no change)
        mean, std: Normalization parameters (None = skip normalization)
        
    Returns:
        Transformed tensor of shape (N, C, height, width)
        
    Example:
        >>> img = torch.rand(4, 3, 224, 224, device='cuda')
        >>> # Single kernel launch for ALL operations!
        >>> result = ultimate_fused_augment(
        ...     img,
        ...     top=20, left=30, height=112, width=112,
        ...     flip_horizontal=True,
        ...     brightness_factor=1.2,
        ...     contrast_factor=1.1,
        ...     saturation_factor=0.9,
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225)
        ... )
        >>> 
        >>> # Equivalent sequential operations (MUCH slower - 6 kernel launches):
        >>> result_seq = crop(img, 20, 30, 112, 112)
        >>> result_seq = horizontal_flip(result_seq)
        >>> result_seq = adjust_brightness(result_seq, 1.2)
        >>> result_seq = adjust_contrast_fast(result_seq, 1.1)
        >>> result_seq = adjust_saturation(result_seq, 0.9)
        >>> result_seq = normalize(result_seq, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    Note:
        - Uses FAST contrast (centered scaling), not torchvision's blend-with-mean
        - For torchvision-exact contrast, use sequential operations
        - Zero intermediate memory allocations!
    """
    _validate_image_tensor(image, "image")
    
    batch_size, channels, img_height, img_width = image.shape
    
    # Validate crop region
    if top < 0 or left < 0:
        raise ValueError(f"Crop coordinates must be non-negative, got top={top}, left={left}")
    if top + height > img_height or left + width > img_width:
        raise ValueError(
            f"Crop region ({top}:{top+height}, {left}:{left+width}) "
            f"exceeds image dimensions ({img_height}, {img_width})"
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"Crop size must be positive, got height={height}, width={width}")
    
    # Determine which pixel operations to apply
    apply_brightness = (brightness_factor != 1.0)
    apply_contrast = (contrast_factor != 1.0)
    apply_saturation = (saturation_factor != 1.0)
    apply_normalize = (mean is not None and std is not None)
    
    # Validate and prepare normalization parameters
    if apply_normalize:
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must have 3 values for RGB channels")
        norm_mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype)
        norm_std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype)
    else:
        # Create dummy tensors (won't be used due to compile-time flag)
        norm_mean_tensor = torch.zeros(3, device=image.device, dtype=image.dtype)
        norm_std_tensor = torch.ones(3, device=image.device, dtype=image.dtype)
    
    # Allocate output tensor
    output = torch.empty(
        batch_size, channels, height, width,
        dtype=image.dtype,
        device=image.device
    )
    
    # Calculate total elements and grid size based on processing path
    output_spatial_size = height * width
    if apply_saturation:
        # Spatial processing
        total_spatial = batch_size * output_spatial_size
        grid = lambda meta: (triton.cdiv(total_spatial, meta['BLOCK_SIZE']),)
    else:
        # Linear processing (faster)
        total_elements = batch_size * channels * output_spatial_size
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Launch the ultimate fused kernel!
    BLOCK_SIZE = 1024
    ultimate_fused_kernel[grid](
        image,
        output,
        batch_size,
        channels,
        img_height,
        img_width,
        height,
        width,
        top,
        left,
        int(flip_horizontal),
        norm_mean_tensor,
        norm_std_tensor,
        brightness_factor,
        contrast_factor,
        saturation_factor,
        apply_brightness,
        apply_contrast,
        apply_saturation,
        apply_normalize,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# ============================================================================
# Aliases for convenience
# ============================================================================

apply_brightness = adjust_brightness
apply_contrast = adjust_contrast
apply_saturation = adjust_saturation
apply_normalize = normalize


__all__ = [
    # Color operations
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
    # Grayscale operations
    'rgb_to_grayscale',
    'random_grayscale',
    # Geometric operations
    'crop',
    'center_crop',
    'horizontal_flip',
    'fused_crop_flip',
    # Ultimate fusion
    'ultimate_fused_augment',
]

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
    brightness_kernel,
    contrast_kernel,
    contrast_fast_kernel,
    saturation_kernel,
    normalize_kernel,
    rgb_to_grayscale_kernel,
)
from .kernels.geometric_kernel import (
    crop_kernel,
    horizontal_flip_kernel,
)
from .kernels.fused_kernel import ultimate_fused_kernel
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
        This functional API expects 4D tensors (N, C, H, W).
        Transform classes (e.g., TritonFusedAugment) handle 3D (C, H, W) and 5D (N, T, C, H, W) inputs
        by normalizing them to 4D before calling the functional API.
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


# ============================================================================
# Parameter conversion helpers for per-image randomness
# ============================================================================

def _convert_to_tensor(
    value: int | float | bool | torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str = "parameter"
) -> torch.Tensor:
    """
    Convert scalar or tensor to a tensor of shape (batch_size,) with proper dtype and device.
    
    This helper is used to support both scalar parameters (applied to all images in batch)
    and tensor parameters (per-image values).
    
    Args:
        value: Scalar value or tensor to convert
        batch_size: Expected batch size
        dtype: Target dtype for the output tensor
        device: Target device for the output tensor
        name: Parameter name for error messages
    
    Returns:
        Tensor of shape (batch_size,) with the given dtype and device
    
    Raises:
        ValueError: If tensor shape doesn't match (batch_size,)
    
    Example:
        ```python
        # Scalar -> broadcast to all images
        t = _convert_to_tensor(1.5, batch_size=4, dtype=torch.float32, device='cuda')
        # t = tensor([1.5, 1.5, 1.5, 1.5], device='cuda')
        # Tensor -> validate and convert
        per_image = torch.tensor([1.0, 1.2, 0.8, 1.5], device='cuda')
        t = _convert_to_tensor(per_image, batch_size=4, dtype=torch.float32, device='cuda')
        # t = tensor([1.0, 1.2, 0.8, 1.5], device='cuda')
        ```
    """
    if isinstance(value, torch.Tensor):
        # Validate shape
        if value.shape != (batch_size,):
            raise ValueError(
                f"{name} tensor must have shape ({batch_size},), got {value.shape}"
            )
        # Convert dtype and device
        return value.to(device=device, dtype=dtype)
    else:
        # Convert scalar to tensor filled with same value
        return torch.full((batch_size,), value, dtype=dtype, device=device)


def _sample_uniform_tensor(
    size: int,
    min_val: float,
    max_val: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample uniform random values independently for each position.
    
    Args:
        size: Number of random values to generate
        min_val: Minimum value (inclusive)
        max_val: Maximum value (exclusive)
        device: Device to create tensor on
    
    Returns:
        Tensor of shape (size,) with uniform random values
    
    Example:
        ```python
        t = _sample_uniform_tensor(4, 0.8, 1.2, 'cuda')
        # t = tensor([0.95, 1.1, 0.82, 1.18], device='cuda')
        ```
    """
    return torch.empty(size, device=device).uniform_(min_val, max_val)


def _sample_bernoulli_tensor(
    size: int,
    p: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample Bernoulli random values as boolean independently for each position.
    
    Args:
        size: Number of random values to generate
        p: Probability of True
        device: Device to create tensor on
    
    Returns:
        Tensor of shape (size,) with bool dtype
    
    Example:
        ```python
        t = _sample_bernoulli_tensor(4, 0.5, 'cuda')
        # t = tensor([True, False, True, True], device='cuda', dtype=torch.bool)
        ```
    """
    return torch.rand(size, device=device) < p


def rgb_to_grayscale(
    image: torch.Tensor,
    num_output_channels: int = 1,
    grayscale_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert RGB image to grayscale with optional per-image masking.
    
    Matches torchvision.transforms.v2.functional.rgb_to_grayscale exactly.
    Uses weights: 0.2989*R + 0.587*G + 0.114*B
    
    Args:
        image: Input image tensor of shape (N, 3, H, W) on CUDA
        num_output_channels: Number of output channels (1 or 3)
                            If 3, grayscale is replicated across channels
        grayscale_mask: Optional per-image mask [N] (uint8: 0=keep original, 1=convert to gray)
                       If None, converts all images
                            
    Returns:
        Grayscale tensor of shape (N, num_output_channels, H, W)
        
    Raises:
        ValueError: If num_output_channels not in {1, 3} or if input not RGB
        
    Example:
        ```python
        img = torch.rand(4, 3, 224, 224, device='cuda')
        # Convert all images
        gray = rgb_to_grayscale(img, num_output_channels=3)
        # Convert only some images (per-image mask)
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool, device='cuda')
        gray = rgb_to_grayscale(img, num_output_channels=3, grayscale_mask=mask)
        ```
    """
    _validate_image_tensor(image, "image")
    
    if num_output_channels not in (1, 3):
        raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}")
    
    if image.shape[1] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {image.shape[1]}")
    
    batch_size, channels, height, width = image.shape
    
    # For 1-channel output, we still need to process all 3 channels internally
    # So we'll always use the kernel with 3-channel output, then select channel 0 if needed
    output = torch.empty_like(image)  # Always (N, 3, H, W)
    
    # Prepare grayscale mask
    if grayscale_mask is None:
        # Convert all images - create an all-ones mask
        grayscale_mask_tensor = torch.ones(batch_size, dtype=torch.bool, device=image.device)
        apply_per_image = False
    else:
        grayscale_mask_tensor = _convert_to_tensor(grayscale_mask, batch_size, torch.bool, image.device, "grayscale_mask")
        apply_per_image = True
    
    # Calculate grid size (spatial processing: N*H*W)
    spatial_size = height * width
    total_spatial = batch_size * spatial_size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total_spatial, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    rgb_to_grayscale_kernel[grid](
        image,
        output,
        grayscale_mask_tensor,
        batch_size,
        channels,
        height,
        width,
        apply_per_image,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return 1-channel if requested
    if num_output_channels == 1:
        return output[:, 0:1, :, :]  # Select first channel only
    else:
        return output


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
        ```python
        img = torch.rand(1, 3, 224, 224, device='cuda')
        bright_img = adjust_brightness(img, brightness_factor=1.2)  # 20% brighter
        dark_img = adjust_brightness(img, brightness_factor=0.8)   # 20% darker
        ```
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
        ```python
        img = torch.rand(1, 3, 224, 224, device='cuda')
        high_contrast = adjust_contrast(img, contrast_factor=1.5)
        low_contrast = adjust_contrast(img, contrast_factor=0.5)
        ```
    """
    _validate_image_tensor(image, "image")
    
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
    
    # Compute grayscale mean for contrast (per-image)
    c = image.shape[1]
    if c == 3:
        # Use rgb_to_grayscale with num_output_channels=1 to get (N, 1, H, W)
        grayscale_image = rgb_to_grayscale(image, num_output_channels=1)
    else:
        grayscale_image = image
    
    # Compute mean per image (ensure dtype matches input)
    mean = torch.mean(grayscale_image, dim=(-3, -2, -1), keepdim=False)  # Shape: [N]
    mean = mean.to(dtype=image.dtype)  # Match input dtype for Triton kernel
    
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
        ```python
        img = torch.rand(1, 3, 224, 224, device='cuda')
        high_contrast = adjust_contrast_fast(img, contrast_factor=1.5)
        # Use in fused operation for maximum speed
        result = fused_color_normalize(img, contrast_factor=1.5, ...)
        ```
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
    
    Formula: 
        ```python
        output = blend(image, grayscale, saturation_factor)
               = image * saturation_factor + grayscale * (1 - saturation_factor)
        ```
    
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
        ```python
        img = torch.rand(1, 3, 224, 224, device='cuda')
        grayscale = adjust_saturation(img, saturation_factor=0.0)
        saturated = adjust_saturation(img, saturation_factor=2.0)
        ```
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
        ```python
        img = torch.rand(1, 3, 224, 224, device='cuda')
        normalized = normalize(img,
                              mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225))
        ```
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


# ============================================================================
# Geometric Transformations
# ============================================================================


def crop(
    image: torch.Tensor,
    top: int | torch.Tensor,
    left: int | torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Crop a rectangular region from the input image, with optional per-image crop positions.
    
    Matches torchvision.transforms.v2.functional.crop exactly when using scalar top/left.
    Reference: torchvision/transforms/v2/functional/_geometry.py line 1787
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        top: Top pixel coordinate for cropping (int or int32 tensor of shape (N,) for per-image)
        left: Left pixel coordinate for cropping (int or int32 tensor of shape (N,) for per-image)
        height: Height of the cropped image
        width: Width of the cropped image
        
    Returns:
        Cropped tensor of shape (N, C, height, width)
        
    Example:
        ```python
        img = torch.rand(2, 3, 224, 224, device='cuda')
        # Crop all images at same position
        cropped = crop(img, top=56, left=56, height=112, width=112)
        # Crop each image at different position
        tops = torch.tensor([56, 100], device='cuda', dtype=torch.int32)
        lefts = torch.tensor([56, 80], device='cuda', dtype=torch.int32)
        cropped = crop(img, top=tops, left=lefts, height=112, width=112)
        ```
    
    Note:
        For MVP, this requires valid crop coordinates (no padding).
        Future versions will support padding for out-of-bounds crops.
    """
    _validate_image_tensor(image, "image")
    
    batch_size, channels, img_height, img_width = image.shape
    
    if height <= 0 or width <= 0:
        raise ValueError(f"Crop size must be positive, got height={height}, width={width}")
    
    # Convert crop offsets to tensors
    top_offsets = _convert_to_tensor(top, batch_size, torch.int32, image.device, "top")
    left_offsets = _convert_to_tensor(left, batch_size, torch.int32, image.device, "left")
    
    # Determine if per-image
    apply_per_image = isinstance(top, torch.Tensor) or isinstance(left, torch.Tensor)
    
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
        top_offsets,
        left_offsets,
        apply_per_image,
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
        ```python
        img = torch.rand(2, 3, 224, 224, device='cuda')
        # Center crop to 112x112
        cropped = center_crop(img, (112, 112))
        # or for square crop
        cropped = center_crop(img, 112)
        ```
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


def horizontal_flip(
    image: torch.Tensor,
    flip_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Flip the image horizontally (left to right), with optional per-image control.
    
    Matches torchvision.transforms.v2.functional.horizontal_flip exactly when flip_mask=None.
    Reference: torchvision/transforms/v2/functional/_geometry.py line 56
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        flip_mask: Optional uint8 tensor of shape (N,) indicating which images to flip (0=no flip, 1=flip).
                  If None, flips all images (default behavior).
        
    Returns:
        Horizontally flipped tensor of the same shape
        
    Example:
        ```python
        img = torch.rand(2, 3, 224, 224, device='cuda')
        # Flip all images
        flipped = horizontal_flip(img)
        # Flip only first image
        flip_mask = torch.tensor([1, 0], device='cuda', dtype=torch.bool)
        flipped = horizontal_flip(img, flip_mask)
        ```
    
    Note:
        This uses a custom Triton kernel. For standalone flip operations,
        PyTorch's tensor.flip(-1) is highly optimized and may be comparable.
        The main benefit is when fusing with crop (see fused_crop_flip).
    """
    _validate_image_tensor(image, "image")
    
    batch_size, channels, height, width = image.shape
    
    # Handle flip_mask
    if flip_mask is None:
        # Flip all images - use simple path
        apply_per_image = False
        flip_mask_tensor = torch.ones(batch_size, device=image.device, dtype=torch.bool)
    else:
        # Per-image flip control
        apply_per_image = True
        flip_mask_tensor = _convert_to_tensor(flip_mask, batch_size, torch.bool, image.device, "flip_mask")
        # Early exit if no images need flipping
        if not torch.any(flip_mask_tensor).item():
            return image.clone()
    
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
        flip_mask_tensor,
        apply_per_image,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def fused_augment(
    image: torch.Tensor,
    # Geometric parameters (can be scalars or tensors of shape (N,))
    top: int | torch.Tensor,
    left: int | torch.Tensor,
    height: int,
    width: int,
    flip_horizontal: bool | torch.Tensor = False,
    # Pixel operation parameters (can be scalars or tensors of shape (N,))
    brightness_factor: float | torch.Tensor = 1.0,
    contrast_factor: float | torch.Tensor = 1.0,
    saturation_factor: float | torch.Tensor = 1.0,
    grayscale: bool | torch.Tensor = False,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """
    Fused augmentation: ALL operations in ONE kernel.
    
    Combines geometric (crop + flip) and pixel (color + normalize) operations
    in a single GPU kernel, providing maximum performance.
    
    **Performance**: Up to 12x faster on large images (8.1x average on Tesla T4, scales dramatically with image size)
    
    Operations applied in sequence:
    1. **Geometric**: Crop + Optional Horizontal Flip (index transformations, per-image)
    2. **Pixel**: Brightness + Contrast (fast) + Saturation + Random Grayscale + Normalize (value operations, per-image)
    
    Args:
        image: Input image tensor of shape (N, C, H, W) on CUDA
        top: Crop top offset (int or int32 tensor of shape (N,) for per-image)
        left: Crop left offset (int or int32 tensor of shape (N,) for per-image)
        height: Crop height
        width: Crop width
        flip_horizontal: Whether to flip horizontally (bool or uint8 tensor of shape (N,) for per-image, default: False)
        brightness_factor: Brightness multiplier (float or tensor of shape (N,) for per-image, 1.0 = no change)
        contrast_factor: Contrast multiplier (float or tensor of shape (N,) for per-image, 1.0 = no change) [FAST mode]
        saturation_factor: Saturation multiplier (float or tensor of shape (N,) for per-image, 1.0 = no change)
        grayscale: Whether to convert to grayscale (bool or uint8 tensor of shape (N,) for per-image, default: False)
        mean: Normalization mean parameters (None = skip normalization)
        std: Normalization std parameters (None = skip normalization)
        
    Returns:
        Transformed tensor of shape (N, C, height, width)
        
    Example:
        ```python
        img = torch.rand(4, 3, 224, 224, device='cuda')
        # Single kernel launch for ALL operations!
        result = fused_augment(
            img,
            top=20, left=30, height=112, width=112,
            flip_horizontal=True,
            brightness_factor=1.2,
            contrast_factor=1.1,
            saturation_factor=0.9,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # Equivalent sequential operations (MUCH slower - 6 kernel launches):
        result_seq = crop(img, 20, 30, 112, 112)
        result_seq = horizontal_flip(result_seq)
        result_seq = adjust_brightness(result_seq, 1.2)
        result_seq = adjust_contrast_fast(result_seq, 1.1)
        result_seq = adjust_saturation(result_seq, 0.9)
        result_seq = normalize(result_seq, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ```
    
    Note:
        - Uses FAST contrast (centered scaling), not torchvision's blend-with-mean
        - For torchvision-exact contrast, use sequential operations
        - Zero intermediate memory allocations!
    """
    _validate_image_tensor(image, "image")
    
    batch_size, channels, img_height, img_width = image.shape
    
    if height <= 0 or width <= 0:
        raise ValueError(f"Crop size must be positive, got height={height}, width={width}")
    
    # Convert geometric parameters to tensors of shape (N,)
    top_offsets = _convert_to_tensor(top, batch_size, torch.int32, image.device, "top")
    left_offsets = _convert_to_tensor(left, batch_size, torch.int32, image.device, "left")
    flip_mask = _convert_to_tensor(flip_horizontal, batch_size, torch.bool, image.device, "flip_horizontal")
    
    # Convert color factors to tensors of shape (N,)
    brightness_factors = _convert_to_tensor(brightness_factor, batch_size, image.dtype, image.device, "brightness_factor")
    contrast_factors = _convert_to_tensor(contrast_factor, batch_size, image.dtype, image.device, "contrast_factor")
    saturation_factors = _convert_to_tensor(saturation_factor, batch_size, image.dtype, image.device, "saturation_factor")
    grayscale_mask = _convert_to_tensor(grayscale, batch_size, torch.bool, image.device, "grayscale")
    
    # Determine which pixel operations to apply
    # Check if ANY image needs the operation
    apply_brightness = not torch.all(brightness_factors == 1.0).item()
    apply_contrast = not torch.all(contrast_factors == 1.0).item()
    apply_saturation = not torch.all(saturation_factors == 1.0).item()
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
    
    # Check if any image needs grayscale
    apply_grayscale = torch.any(grayscale_mask).item()
    
    # Calculate total elements and grid size based on processing path
    output_spatial_size = height * width
    total_elements = batch_size * channels * output_spatial_size
    
    # N for auto-tuning key
    N = total_elements
    
    if apply_saturation or apply_grayscale:
        # Spatial processing (required for saturation/grayscale)
        total_spatial = batch_size * output_spatial_size
        grid = lambda meta: (triton.cdiv(total_spatial, meta['BLOCK_SIZE']),)
    else:
        # Linear processing (faster)
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Show auto-tuning message if this kernel hasn't been tuned yet (only when enabled)
    if ENABLE_AUTOTUNE and should_show_autotune_message('ultimate_fused_kernel', (N,)):
        print(f"[Triton-Augment] Auto-tuning ultimate_fused_kernel for batch={batch_size}, size={height}×{width}... (~2-5 sec)", 
              file=sys.stderr, flush=True)
    
    # Launch the ultimate fused kernel (auto-tuned with 4 configs or fixed with 1 config based on ENABLE_AUTOTUNE)
    ultimate_fused_kernel[grid](
        image,
        output,
        N,                     # Auto-tuning key
        batch_size,
        channels,
        img_height,
        img_width,
        height,
        width,
        top_offsets,           # Per-image tensors!
        left_offsets,
        flip_mask,
        norm_mean_tensor,
        norm_std_tensor,
        brightness_factors,
        contrast_factors,
        saturation_factors,
        grayscale_mask,        # Per-image tensor!
        apply_brightness,
        apply_contrast,
        apply_saturation,
        apply_grayscale,       # Whether to check grayscale mask at all
        apply_normalize,
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
    'rgb_to_grayscale',
    # Geometric operations
    'crop',
    'center_crop',
    'horizontal_flip',
    # Fused operations
    'fused_augment',
]

"""
Transform classes for Triton-accelerated image augmentation.

This module provides stateful transform classes similar to torchvision.transforms,
designed to work seamlessly in data augmentation pipelines.

Implementation matches torchvision.transforms.v2 exactly.
Reference: torchvision/transforms/v2/_color.py

Author: yuhezhang-ai
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Sequence
import collections.abc

from . import functional as F
from .functional import InterpolationMode


# ============================================================================
# Helper functions for video (5D) tensor support
# ============================================================================

def _normalize_video_shape(
    image: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[int], Optional[int], Tuple, bool]:
    """
    Normalize input shape to [N, C, H, W] for processing, handling 3D, 4D, and 5D inputs.
    
    Supported shapes:
    - [C, H, W] → [1, C, H, W], batch_size=None, num_frames=None
    - [N, C, H, W] → unchanged, batch_size=N, num_frames=None
    - [N, T, C, H, W] → [N*T, C, H, W], batch_size=N, num_frames=T
    
    Args:
        image: Input tensor of shape [C, H, W], [N, C, H, W], or [N, T, C, H, W]
        
    Returns:
        Tuple of:
        - normalized_image: Tensor of shape [N, C, H, W] ready for processing
        - batch_size: N (first dimension) or None if not video
        - num_frames: T (second dimension) or None if not video
        - original_shape: Original shape to reshape back
        - was_3d: Whether input was originally 3D
    """
    original_shape = image.shape
    was_3d = image.ndim == 3
    
    # Handle 3D: [C, H, W] → [1, C, H, W]
    if was_3d:
        image = image.unsqueeze(0)
        return image, None, None, original_shape, was_3d
    
    # Handle 4D: [N, C, H, W] - no change needed
    if image.ndim == 4:
        return image, image.shape[0], None, original_shape, was_3d
    
    # Handle 5D: [N, T, C, H, W] → [N*T, C, H, W]
    if image.ndim == 5:
        batch_size = image.shape[0]
        num_frames = image.shape[1]
        # Flatten: [N, T, C, H, W] → [N*T, C, H, W]
        normalized = image.reshape(-1, *image.shape[2:])
        return normalized, batch_size, num_frames, original_shape, was_3d
    
    raise ValueError(f"Expected 3D, 4D, or 5D tensor, got {image.ndim}D tensor with shape {image.shape}")


def _compute_param_count(
    batch_size: Optional[int],
    num_frames: Optional[int],
    same_on_batch: bool,
    same_on_frame: bool
) -> int:
    """
    Compute how many parameter sets to generate based on video dimensions and flags.
    
    For [N, T, C, H, W]:
    - same_on_batch=False, same_on_frame=False → N*T params (all independent)
    - same_on_batch=False, same_on_frame=True  → N params (per video)
    - same_on_batch=True,  same_on_frame=False → T params (per frame position)
    - same_on_batch=True,  same_on_frame=True  → 1 param (shared)
    
    For 3D inputs (num_frames=None), returns 1.
    For 4D inputs (num_frames=None), returns 1 or the batch size if same_on_batch is False.
    
    Args:
        batch_size: N dimension or None for 3D
        num_frames: T dimension or None for 3D/4D
        same_on_batch: Whether to share params across batch dimension
        same_on_frame: Whether to share params across frame dimension
        
    Returns:
        Number of parameter sets to generate
    """
    # 3D input: [C, H, W] → always 1 param
    if batch_size is None:
        return 1
    
    # 4D input: [N, C, H, W] → use same_on_batch for batch dimension
    if num_frames is None:
        return 1 if same_on_batch else batch_size

    # For 5D inputs
    if same_on_batch and same_on_frame:
        return 1
    elif same_on_batch and not same_on_frame:
        return num_frames
    elif not same_on_batch and same_on_frame:
        return batch_size
    else:  # not same_on_batch and not same_on_frame
        return batch_size * num_frames


def _broadcast_params_to_all_samples(
    params: torch.Tensor,
    batch_size: Optional[int],
    num_frames: Optional[int],
    total_samples: int,
    same_on_batch: bool,
    same_on_frame: bool
) -> torch.Tensor:
    """
    Broadcast parameters to shape [total_samples] based on same_on_batch and same_on_frame.
    
    For [N, T, C, H, W] with total_samples=N*T:
    - params shape [1] → broadcast to [N*T]
    - params shape [N] → each broadcasted to T frames → [N*T]
    - params shape [T] → repeat for each of N videos → [N*T]
    - params shape [N*T] → already correct shape
    
    For [N, C, H, W] with total_samples=N:
    - params shape [1] → broadcast to [N]
    - params shape [N] → already correct shape
    
    Args:
        params: Parameter tensor
        batch_size: N dimension (None for 3D only)
        num_frames: T dimension (None for 3D/4D)
        total_samples: Total flattened size (N*T for 5D, N for 4D, 1 for 3D)
        same_on_batch: Whether params are shared across batch
        same_on_frame: Whether params are shared across frames
        
    Returns:
        Parameter tensor of shape [total_samples]
    """
    # Already correct shape
    if params.shape[0] == total_samples:
        return params
    
    # 3D input: [C, H, W]
    if batch_size is None:
        if params.shape[0] == 1:
            return params.expand(total_samples).contiguous()
        return params
    
    # 4D input: [N, C, H, W]
    if num_frames is None:
        if params.shape[0] == 1:
            return params.expand(total_samples).contiguous()
        return params
    
    # 5D input: [N, T, C, H, W] - use flags to determine which case we're in
    N, T = batch_size, num_frames
    
    # Case 1: [1] → [N*T] (all same: same_on_batch=True, same_on_frame=True)
    if params.shape[0] == 1:
        return params.expand(total_samples).contiguous()
    
    # Case 2: [N] → [N*T] (same_on_frame=True, not same_on_batch)
    # Each video's param broadcast to its T frames
    if not same_on_batch and same_on_frame:
        if params.shape[0] != N:
            raise ValueError(
                f"Expected {N} params for same_on_frame=True, got {params.shape[0]}"
            )
        # [N] → [N, 1] → [N, T] → [N*T]
        return params.unsqueeze(1).expand(N, T).reshape(-1).contiguous()
    
    # Case 3: [T] → [N*T] (same_on_batch=True, not same_on_frame)
    # Frame params repeated for each video
    if same_on_batch and not same_on_frame:
        if params.shape[0] != T:
            raise ValueError(
                f"Expected {T} params for same_on_batch=True, got {params.shape[0]}"
            )
        # [T] → [1, T] → [N, T] → [N*T]
        return params.unsqueeze(0).expand(N, T).reshape(-1).contiguous()
    
    # Case 4: Should not reach here (all other cases handled by earlier checks)
    raise ValueError(
        f"Unexpected param shape {params.shape} for video with "
        f"batch_size={batch_size}, num_frames={num_frames}, total_samples={total_samples}, "
        f"same_on_batch={same_on_batch}, same_on_frame={same_on_frame}"
    )


def _broadcast_2d_params(
    params: torch.Tensor,
    batch_size: Optional[int],
    num_frames: Optional[int],
    total_samples: int,
    same_on_batch: bool,
    same_on_frame: bool
) -> torch.Tensor:
    """
    Broadcast 2D parameters [N, 2] to all samples [total_samples, 2].
    """
    if params.shape[0] == total_samples:
        return params
        
    # Broadcast each component separately
    x = _broadcast_params_to_all_samples(params[:, 0], batch_size, num_frames, total_samples, same_on_batch, same_on_frame)
    y = _broadcast_params_to_all_samples(params[:, 1], batch_size, num_frames, total_samples, same_on_batch, same_on_frame)
    
    return torch.stack([x, y], dim=1)



def _reshape_to_original(
    output: torch.Tensor,
    original_shape: Tuple,
    was_3d: bool
) -> torch.Tensor:
    """
    Reshape output back to original input shape.
    
    Args:
        output: Processed tensor of shape [N, C, H', W'] (spatial dims may differ)
        original_shape: Original input shape
        was_3d: Whether input was 3D
        
    Returns:
        Tensor reshaped to match original dimensions (3D→3D, 4D→4D, 5D→5D)
        Note: Spatial dimensions (H', W') may differ from original due to cropping
    """
    if was_3d:
        # 3D input: [C, H, W] → [1, C, H', W'] → [C, H', W']
        return output.squeeze(0)
    elif len(original_shape) == 5:
        # 5D input: [N, T, C, H, W] → [N*T, C, H', W'] → [N, T, C, H', W']
        N, T = original_shape[0], original_shape[1]
        return output.reshape(N, T, *output.shape[1:])
    else:
        # 4D input: no reshaping needed, spatial dims may have changed
        return output


class TritonColorJitter(nn.Module):
    """
    Randomly change the brightness, contrast, and saturation of an image.
    
    This is a GPU-accelerated version of torchvision.transforms.v2.ColorJitter
    that uses a fused kernel for maximum performance.
    
    **IMPORTANT**: Contrast uses FAST mode (centered scaling: `(pixel - 0.5) * factor + 0.5`),
    NOT torchvision's blend-with-mean approach. This is much faster and provides similar visual results.
    
    If you need exact torchvision behavior, use the individual functional APIs:
    - `F.adjust_brightness()` (exact)
    - `F.adjust_contrast()` (torchvision-exact, slower)
    - `F.adjust_saturation()` (exact)
    
    Args:
        brightness: How much to jitter brightness.
                   brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                   or the given [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast (uses FAST mode).
                 contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                 or the given [min, max]. Should be non-negative numbers.
        saturation: How much to jitter saturation.
                   saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                   or the given [min, max]. Should be non negative numbers.
        same_on_batch: If True, all images in batch (N dimension) share the same random parameters.
                      If False (default), each image gets different random parameters.
        same_on_frame: If True, all frames in a video (T dimension) share the same random parameters.
                      If False, each frame gets different random parameters.
                      Only applies to 5D input [N, T, C, H, W]. Default: True (consistent augmentation across frames).
        
    Example:
        ```python
        # Basic usage with per-image randomness
        transform = TritonColorJitter(
            brightness=0.2,  # Range: [0.8, 1.2]
            contrast=0.2,    # Range: [0.8, 1.2] (FAST contrast)
            saturation=0.2,  # Range: [0.8, 1.2]
            same_on_batch=False
        )
        img = torch.rand(4, 3, 224, 224, device='cuda')
        augmented = transform(img)  # Each image gets different augmentation
        
        # Custom ranges
        transform = TritonColorJitter(
            brightness=(0.5, 1.5),  # Custom range
            contrast=(0.7, 1.3),     # Custom range (FAST mode)
            saturation=(0.0, 2.0)    # Custom range
        )
        ```
    
    Performance:
        - Uses fused kernel for all operations in a single pass
        - Faster than sequential operations
        - For even more speed, combine with normalization using TritonColorJitterNormalize
    """
    
    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        same_on_batch: bool = False,
        same_on_frame: bool = True,
    ):
        super().__init__()
        
        # Process parameters exactly like torchvision
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.same_on_batch = same_on_batch
        self.same_on_frame = same_on_frame
    
    def _check_input(
        self,
        value: Optional[Union[float, Sequence[float]]],
        name: str,
        center: float = 1.0,
        bound: Tuple[float, float] = (0, float("inf")),
        clip_first_on_zero: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """
        Check and process input parameters exactly like torchvision.
        Reference: torchvision/transforms/v2/_color.py line 115-140
        """
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, collections.abc.Sequence) and len(value) == 2:
            value = [float(v) for v in value]
        else:
            raise TypeError(f"{name}={value} should be a single number or a sequence with length 2.")
        
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound} and increasing, but got {value}.")
        
        return None if value[0] == value[1] == center else (float(value[0]), float(value[1]))
    
    def _get_params(self, param_count: int, device: torch.device):
        """
        Randomly sample transformation parameters (per-image or batch-wide).
        
        Returns:
            Tuple of (brightness_factors, contrast_factors, saturation_factors) as tensors
        """
        # Generate brightness factors
        if self.brightness is not None:
            brightness_factors = F._sample_uniform_tensor(
                param_count, self.brightness[0], self.brightness[1], device
            )
        else:
            brightness_factors = torch.ones(param_count, device=device)
        
        # Generate contrast factors
        if self.contrast is not None:
            contrast_factors = F._sample_uniform_tensor(
                param_count, self.contrast[0], self.contrast[1], device
            )
        else:
            contrast_factors = torch.ones(param_count, device=device)
        
        # Generate saturation factors
        if self.saturation is not None:
            saturation_factors = F._sample_uniform_tensor(
                param_count, self.saturation[0], self.saturation[1], device
            )
        else:
            saturation_factors = torch.ones(param_count, device=device)
        
        return brightness_factors, contrast_factors, saturation_factors
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random color jitter to the input image tensor.
        
        Args:
            img: Input image tensor of shape (C, H, W), (N, C, H, W), or (N, T, C, H, W)
            
        Returns:
            Augmented image tensor of the same shape and dtype
        """
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(img)
        
        # Move to CUDA if needed
        if not normalized_img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton-Augment requires CUDA")
            normalized_img = normalized_img.cuda()
        
        total_samples = normalized_img.shape[0]
        
        # Compute how many parameter sets to generate
        param_count = _compute_param_count(batch_size, num_frames, self.same_on_batch, self.same_on_frame)
        
        # Sample random parameters
        brightness_factors, contrast_factors, saturation_factors = self._get_params(param_count, normalized_img.device)
        
        # Broadcast params to all samples
        brightness_factors = _broadcast_params_to_all_samples(
            brightness_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        contrast_factors = _broadcast_params_to_all_samples(
            contrast_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        saturation_factors = _broadcast_params_to_all_samples(
            saturation_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        # Apply transformations using fused kernel (FAST contrast mode)
        _, _, height, width = normalized_img.shape
        output = F.fused_augment(
            normalized_img,
            top=0,  # No crop
            left=0,  # No crop
            height=height,
            width=width,
            flip_horizontal=False,  # No flip
            brightness_factor=brightness_factors,
            contrast_factor=contrast_factors,
            saturation_factor=saturation_factors,
            grayscale=False,  # No grayscale
            mean=None,  # No normalization
            std=None,
        )
        
        # Reshape back to original shape
        return _reshape_to_original(output, original_shape, was_3d)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"same_on_batch={self.same_on_batch}, "
            f"same_on_frame={self.same_on_frame})"
        )


class TritonNormalize(nn.Module):
    """
    Normalize a tensor image with mean and standard deviation.
    
    This is a GPU-accelerated version of torchvision.transforms.Normalize
    that uses a Triton kernel for improved performance.
    
    Args:
        mean: Sequence of means for each channel (R, G, B)
        std: Sequence of standard deviations for each channel (R, G, B)
        
    Example:
        ```python
        # ImageNet normalization
        normalize = TritonNormalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        img = torch.rand(1, 3, 224, 224, device='cuda')
        normalized = normalize(img)
        ```
    """
    
    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ):
        super().__init__()
        self.mean = tuple(mean)
        self.std = tuple(std)
        
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must have 3 values for RGB channels")
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input image tensor.
        
        Args:
            img: Input tensor of shape (C, H, W), (N, C, H, W), or (N, T, C, H, W)
                 Can be on CPU or CUDA (will be moved to CUDA automatically)
            
        Returns:
            Normalized tensor of same shape and device as input
        """
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(img)
        
        # Move to CUDA if not already (Triton requires CUDA)
        if not normalized_img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Triton-Augment requires CUDA, but CUDA is not available. "
                    "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
                )
            normalized_img = normalized_img.cuda()
        
        result = F.normalize(normalized_img, mean=self.mean, std=self.std)
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mean={self.mean}, "
            f"std={self.std})"
        )


class TritonColorJitterNormalize(nn.Module):
    """
    Combined color jitter, random grayscale, and normalization in a single fused operation.
    
    This class combines TritonColorJitter, TritonRandomGrayscale, and TritonNormalize 
    into a single operation that uses a fused kernel for maximum performance. This is the
    recommended way to apply color augmentations and normalization.
    
    Args:
        brightness: How much to jitter brightness (same as TritonColorJitter)
        contrast: How much to jitter contrast (same as TritonColorJitter)
        saturation: How much to jitter saturation (same as TritonColorJitter)
        grayscale_p: Probability of converting to grayscale (default: 0.0)
        mean: Sequence of means for normalization (R, G, B). If None, normalization is skipped.
        std: Sequence of standard deviations for normalization (R, G, B). If None, normalization is skipped.
        same_on_batch: If True, all images in batch share the same random parameters
                             If False (default), each image in batch gets different random parameters
        
    Example:
        ```python
        # Full augmentation pipeline in one transform (per-image randomness)
        transform = TritonColorJitterNormalize(
            brightness=0.2,  # Range: [0.8, 1.2]
            contrast=0.2,    # Range: [0.8, 1.2]
            saturation=0.2,  # Range: [0.8, 1.2]
            grayscale_p=0.1,  # 10% chance of grayscale (per-image)
            mean=(0.485, 0.456, 0.406),  # ImageNet normalization (optional)
            std=(0.229, 0.224, 0.225),    # ImageNet normalization (optional)
            same_on_batch=False
        )
        img = torch.rand(4, 3, 224, 224, device='cuda')
        augmented = transform(img)  # Each image gets different augmentation
        
        # Without normalization (mean=None, std=None by default)
        transform_no_norm = TritonColorJitterNormalize(
            brightness=0.2, contrast=0.2, saturation=0.2
        )
        ```
    """
    
    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        grayscale_p: float = 0.0,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        same_on_batch: bool = False,
        same_on_frame: bool = True,
    ):
        super().__init__()
        
        # Process parameters exactly like torchvision
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.grayscale_p = grayscale_p
        self.same_on_batch = same_on_batch
        self.same_on_frame = same_on_frame
        
        if not (0.0 <= grayscale_p <= 1.0):
            raise ValueError(f"grayscale_p must be in [0, 1], got {grayscale_p}")
        
        # Store normalization parameters
        if (mean is None) != (std is None):
            raise ValueError("mean and std must both be provided or both be None")
        
        if mean is not None:
            mean = tuple(mean)
            if len(mean) != 3:
                raise ValueError("mean must have 3 values for RGB channels")
        if std is not None:
            std = tuple(std)
            if len(std) != 3:
                raise ValueError("std must have 3 values for RGB channels")
        
        self.mean = mean
        self.std = std
    
    def _check_input(
        self,
        value: Optional[Union[float, Sequence[float]]],
        name: str,
        center: float = 1.0,
        bound: Tuple[float, float] = (0, float("inf")),
        clip_first_on_zero: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Check and process input parameters exactly like torchvision."""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, collections.abc.Sequence) and len(value) == 2:
            value = [float(v) for v in value]
        else:
            raise TypeError(f"{name}={value} should be a single number or a sequence with length 2.")
        
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound} and increasing, but got {value}.")
        
        return None if value[0] == value[1] == center else (float(value[0]), float(value[1]))
    
    def _get_params(self, param_count: int, device: torch.device):
        """Randomly sample transformation parameters (per-image or batch-wide)."""
        # Generate brightness factors
        if self.brightness is not None:
            brightness_factors = F._sample_uniform_tensor(
                param_count, self.brightness[0], self.brightness[1], device
            )
        else:
            brightness_factors = torch.ones(param_count, device=device)
        
        # Generate contrast factors
        if self.contrast is not None:
            contrast_factors = F._sample_uniform_tensor(
                param_count, self.contrast[0], self.contrast[1], device
            )
        else:
            contrast_factors = torch.ones(param_count, device=device)
        
        # Generate saturation factors
        if self.saturation is not None:
            saturation_factors = F._sample_uniform_tensor(
                param_count, self.saturation[0], self.saturation[1], device
            )
        else:
            saturation_factors = torch.ones(param_count, device=device)
        
        # Generate grayscale mask
        grayscale_mask = F._sample_bernoulli_tensor(
            param_count, self.grayscale_p, device
        )
        
        return brightness_factors, contrast_factors, saturation_factors, grayscale_mask
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random color jitter and normalization in a single fused operation.
        
        Args:
            img: Input tensor of shape (C, H, W), (N, C, H, W), or (N, T, C, H, W)
                 Can be on CPU or CUDA (will be moved to CUDA automatically)
            
        Returns:
            Augmented and normalized tensor of same shape and device as input
        """
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(img)
        
        # Move to CUDA if not already (Triton requires CUDA)
        if not normalized_img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Triton-Augment requires CUDA, but CUDA is not available. "
                    "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
                )
            normalized_img = normalized_img.cuda()
        
        total_samples = normalized_img.shape[0]
        
        # Compute how many parameter sets to generate
        param_count = _compute_param_count(batch_size, num_frames, self.same_on_batch, self.same_on_frame)

        # Sample random parameters
        brightness_factors, contrast_factors, saturation_factors, grayscale_mask = self._get_params(param_count, normalized_img.device)
        
        # Broadcast params to all samples
        brightness_factors = _broadcast_params_to_all_samples(
            brightness_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        contrast_factors = _broadcast_params_to_all_samples(
            contrast_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        saturation_factors = _broadcast_params_to_all_samples(
            saturation_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        grayscale_mask = _broadcast_params_to_all_samples(
            grayscale_mask, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        # Apply the fully fused transformation
        _, _, height, width = normalized_img.shape
        result = F.fused_augment(
            normalized_img,
            top=0,  # No crop
            left=0,  # No crop
            height=height,
            width=width,
            flip_horizontal=False,  # No flip
            brightness_factor=brightness_factors,
            contrast_factor=contrast_factors,
            saturation_factor=saturation_factors,
            grayscale=grayscale_mask,
            mean=self.mean,
            std=self.std,
        )
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"grayscale_p={self.grayscale_p}, "
            f"mean={self.mean}, "
            f"std={self.std}, "
            f"same_on_batch={self.same_on_batch}, "
            f"same_on_frame={self.same_on_frame})"
        )


class TritonGrayscale(nn.Module):
    """
    Convert image to grayscale.
    
    Matches torchvision.transforms.v2.Grayscale behavior.
    Uses weights: 0.2989*R + 0.587*G + 0.114*B
    
    Args:
        num_output_channels: Number of output channels (1 or 3).
                            If 1, output is single-channel grayscale.
                            If 3, grayscale is replicated to 3 channels.
                            
    Example:
        ```python
        transform = TritonGrayscale(num_output_channels=3)
        img = torch.rand(1, 3, 224, 224, device='cuda')
        gray = transform(img)  # Shape: (1, 3, 224, 224), all channels identical
        ```
    """
    
    def __init__(self, num_output_channels: int = 1):
        super().__init__()
        if num_output_channels not in (1, 3):
            raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}")
        self.num_output_channels = num_output_channels
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (C, H, W), (N, 3, H, W), or (N, T, 3, H, W)
            
        Returns:
            Grayscale tensor of shape matching input structure
        """
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(image)
        
        # Apply grayscale conversion
        result = F.rgb_to_grayscale(normalized_img, num_output_channels=self.num_output_channels)
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class TritonRandomGrayscale(nn.Module):
    """
    Randomly convert image to grayscale with probability p.
    
    Matches torchvision.transforms.v2.RandomGrayscale behavior with optional per-image randomness.
    
    Args:
        p: Probability of converting to grayscale (default: 0.1)
        num_output_channels: Number of output channels (1 or 3, default: 3)
                            Usually 3 to maintain compatibility with RGB pipelines
        same_on_batch: If True, all images in batch (N dimension) make the same grayscale decision.
                      If False (default), each image independently decides grayscale conversion.
        same_on_frame: If True, all frames in a video (T dimension) make the same grayscale decision.
                      If False, each frame independently decides.
                      Only applies to 5D input [N, T, C, H, W]. Default: True.
                            
    Example:
        ```python
        # Per-image randomness (each image independently converted)
        transform = TritonRandomGrayscale(p=0.5, num_output_channels=3, same_on_batch=False)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        result = transform(img)  # Each image has 50% chance of being grayscale
        
        # Batch-wide (all images converted or none)
        transform = TritonRandomGrayscale(p=0.5, num_output_channels=3, same_on_batch=True)
        result = transform(img)  # Either all 4 images are grayscale or none are
        ```
    """
    
    def __init__(
        self,
        p: float = 0.1,
        num_output_channels: int = 3,
        same_on_batch: bool = False,
        same_on_frame: bool = True,
    ):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p ({p}) must be in [0, 1]")
        if num_output_channels not in (1, 3):
            raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}")
        self.p = p
        self.num_output_channels = num_output_channels
        self.same_on_batch = same_on_batch
        self.same_on_frame = same_on_frame
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (C, H, W), (N, 3, H, W), or (N, T, 3, H, W)
            
        Returns:
            Image tensor, either original or grayscale based on probability
        """
        if self.p == 0:
            return image
        
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(image)
        total_samples = normalized_img.shape[0]
        
        # Compute how many parameter sets to generate
        param_count = _compute_param_count(batch_size, num_frames, self.same_on_batch, self.same_on_frame)
        
        # Generate grayscale mask
        grayscale_mask = F._sample_bernoulli_tensor(
            param_count, self.p, normalized_img.device
        )
        
        # Broadcast mask to all samples
        grayscale_mask = _broadcast_params_to_all_samples(
            grayscale_mask, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        # Early exit if no images need grayscale
        if not torch.any(grayscale_mask).item():
            return image
        
        # Use deterministic functional API with mask
        result = F.rgb_to_grayscale(
            normalized_img,
            num_output_channels=self.num_output_channels,
            grayscale_mask=grayscale_mask
        )
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}, "
            f"num_output_channels={self.num_output_channels}, "
            f"same_on_batch={self.same_on_batch}, "
            f"same_on_frame={self.same_on_frame})"
        )


# ============================================================================
# Geometric Transforms
# ============================================================================


class TritonRandomCrop(nn.Module):
    """
    Crop a random portion of the image.
    
    Matches torchvision.transforms.v2.RandomCrop behavior (simplified MVP version).
    
    Args:
        size: Desired output size (height, width) or int for square crop
        same_on_batch: If True, all images in batch (N dimension) crop at the same position.
                      If False (default), each image gets different random crop position.
        same_on_frame: If True, all frames in a video (T dimension) crop at the same position.
                      If False, each frame gets different random crop.
                      Only applies to 5D input [N, T, C, H, W]. Default: True.
        
    Example:
        ```python
        transform = TritonRandomCrop(112)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        cropped = transform(img)
        cropped.shape
        ```
        torch.Size([4, 3, 112, 112])
    
    Note:
        For MVP, padding is not supported. Image must be larger than crop size.
        Future versions will support padding, pad_if_needed, fill, padding_mode.
    """
    
    def __init__(self, size: Union[int, Sequence[int]], same_on_batch: bool = False, same_on_frame: bool = True):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            self.size = tuple(size)
        
        if len(self.size) != 2:
            raise ValueError(f"size must have 2 elements, got {len(self.size)}")
        
        self.same_on_batch = same_on_batch
        self.same_on_frame = same_on_frame
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (C, H, W), (N, C, H, W), or (N, T, C, H, W)
            
        Returns:
            Randomly cropped tensor of shape matching input
        """
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(image)
        
        # Move to CUDA if needed
        if not normalized_img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton-Augment requires CUDA")
            normalized_img = normalized_img.cuda()
        
        total_samples, _, h, w = normalized_img.shape
        th, tw = self.size
        
        if h < th or w < tw:
            raise ValueError(
                f"Image size ({h}, {w}) is smaller than crop size ({th}, {tw}). "
                f"Padding not yet supported in MVP."
            )
        
        # Compute how many parameter sets to generate
        param_count = _compute_param_count(batch_size, num_frames, self.same_on_batch, self.same_on_frame)

        # Generate crop positions
        top_offsets = torch.randint(0, h - th + 1, (param_count,), device=normalized_img.device, dtype=torch.int32)
        left_offsets = torch.randint(0, w - tw + 1, (param_count,), device=normalized_img.device, dtype=torch.int32)
        
        # Broadcast params to all samples
        top_offsets = _broadcast_params_to_all_samples(
            top_offsets, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        left_offsets = _broadcast_params_to_all_samples(
            left_offsets, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        result = F.crop(normalized_img, top_offsets, left_offsets, th, tw)
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, same_on_batch={self.same_on_batch}, same_on_frame={self.same_on_frame})"


class TritonCenterCrop(nn.Module):
    """
    Crop the center of the image.
    
    Matches torchvision.transforms.v2.CenterCrop behavior.
    
    Args:
        size: Desired output size (height, width) or int for square crop
        
    Example:
        ```python
        transform = TritonCenterCrop(112)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        cropped = transform(img)
        cropped.shape
        ```
        torch.Size([4, 3, 112, 112])
    """
    
    def __init__(self, size: Union[int, Sequence[int]]):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            self.size = tuple(size)
        
        if len(self.size) != 2:
            raise ValueError(f"size must have 2 elements, got {len(self.size)}")
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (C, H, W), (N, C, H, W), or (N, T, C, H, W)
            
        Returns:
            Center-cropped tensor of shape matching input
        """
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(image)
        
        # Apply center crop
        result = F.center_crop(normalized_img, self.size)
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class TritonRandomHorizontalFlip(nn.Module):
    """
    Horizontally flip the image randomly with probability p.
    
    Matches torchvision.transforms.v2.RandomHorizontalFlip behavior.
    
    Args:
        p: Probability of flipping (default: 0.5)
        same_on_batch: If True, all images in batch (N dimension) share the same flip decision.
                      If False (default), each image gets different random decision.
        same_on_frame: If True, all frames in a video (T dimension) share the same flip decision.
                      If False, each frame gets different random decision.
                      Only applies to 5D input [N, T, C, H, W]. Default: True.
        
    Example:
        ```python
        transform = TritonRandomHorizontalFlip(p=0.5)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        flipped = transform(img)  # Each image has 50% chance of being flipped
        ```
    """
    
    def __init__(self, p: float = 0.5, same_on_batch: bool = False, same_on_frame: bool = True):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p ({p}) must be in [0, 1]")
        self.p = p
        self.same_on_batch = same_on_batch
        self.same_on_frame = same_on_frame
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (C, H, W), (N, C, H, W), or (N, T, C, H, W)
            
        Returns:
            Image tensor, either original or horizontally flipped
        """
        if self.p == 0:
            return image
        
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(image)
        
        # Move to CUDA if needed
        if not normalized_img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton-Augment requires CUDA")
            normalized_img = normalized_img.cuda()
        
        total_samples = normalized_img.shape[0]
        
        # Compute how many parameter sets to generate
        param_count = _compute_param_count(batch_size, num_frames, self.same_on_batch, self.same_on_frame)

        # Generate flip decisions
        flip_mask = F._sample_bernoulli_tensor(param_count, self.p, normalized_img.device)
        
        # Broadcast mask to all samples
        flip_mask = _broadcast_params_to_all_samples(
            flip_mask, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        # Use kernel-based per-image flipping
        result = F.horizontal_flip(normalized_img, flip_mask=flip_mask)
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, same_on_batch={self.same_on_batch}, same_on_frame={self.same_on_frame})"


class TritonRandomCropFlip(nn.Module):
    """
    Fused random crop + random horizontal flip.
    
    This class combines random crop and random horizontal flip in a SINGLE
    kernel launch, eliminating intermediate memory transfers.
    
    **Performance**: ~1.5-2x faster than applying TritonRandomCrop + TritonRandomHorizontalFlip sequentially.
    
    Args:
        size: Desired output size (height, width) or int for square crop
        horizontal_flip_p: Probability of horizontal flip (default: 0.5)
        same_on_batch: If True, all images in batch (N dimension) share the same random parameters.
                      If False (default), each image gets different random parameters.
        same_on_frame: If True, all frames in a video (T dimension) share the same random parameters.
                      If False, each frame gets different random parameters.
                      Only applies to 5D input [N, T, C, H, W]. Default: True.
        
    Example:
        ```python
        # Fused version (FAST - single kernel, per-image randomness)
        transform_fused = TritonRandomCropFlip(112, horizontal_flip_p=0.5, same_on_batch=False)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        result = transform_fused(img)  # Each image gets different crop & flip
        
        # Equivalent sequential version (SLOWER - 2 kernels)
        transform_seq = nn.Sequential(
            TritonRandomCrop(112, same_on_batch=False),
            TritonRandomHorizontalFlip(p=0.5, same_on_batch=False)
        )
        result_seq = transform_seq(img)
        ```
    
    Note:
        The fused version uses compile-time branching (tl.constexpr), so there's
        zero overhead when flip is not triggered.
    """
    
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        horizontal_flip_p: float = 0.5,
        same_on_batch: bool = False,
        same_on_frame: bool = True,
    ):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            self.size = tuple(size)
        
        if len(self.size) != 2:
            raise ValueError(f"size must have 2 elements, got {len(self.size)}")
        
        if not (0.0 <= horizontal_flip_p <= 1.0):
            raise ValueError(f"horizontal_flip_p ({horizontal_flip_p}) must be in [0, 1]")
        
        self.horizontal_flip_p = horizontal_flip_p
        self.same_on_batch = same_on_batch
        self.same_on_frame = same_on_frame
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (C, H, W), (N, C, H, W), or (N, T, C, H, W)
            
        Returns:
            Randomly cropped (and optionally flipped) tensor of shape matching input
        """
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(image)
        
        # Move to CUDA if needed
        if not normalized_img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton-Augment requires CUDA")
            normalized_img = normalized_img.cuda()
        
        total_samples, _, img_height, img_width = normalized_img.shape
        th, tw = self.size
        
        # Validate crop size
        if th > img_height or tw > img_width:
            raise ValueError(
                f"Crop size ({th}, {tw}) is larger than image size ({img_height}, {img_width})"
            )
        
        # Compute how many parameter sets to generate
        param_count = _compute_param_count(batch_size, num_frames, self.same_on_batch, self.same_on_frame)
        
        # Generate crop offsets
        top_offsets = torch.randint(0, img_height - th + 1, (param_count,), device=normalized_img.device, dtype=torch.int32)
        left_offsets = torch.randint(0, img_width - tw + 1, (param_count,), device=normalized_img.device, dtype=torch.int32)
        
        # Generate flip decisions
        flip_mask = F._sample_bernoulli_tensor(
            param_count, self.horizontal_flip_p, normalized_img.device
        )
        
        # Broadcast params to all samples
        top_offsets = _broadcast_params_to_all_samples(
            top_offsets, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        left_offsets = _broadcast_params_to_all_samples(
            left_offsets, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        flip_mask = _broadcast_params_to_all_samples(
            flip_mask, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        # Single fused kernel launch (using fused_augment with no-op color operations)
        result = F.fused_augment(
            normalized_img,
            top=top_offsets,
            left=left_offsets,
            height=th,
            width=tw,
            flip_horizontal=flip_mask,
            brightness_factor=1.0,  # No-op
            contrast_factor=1.0,    # No-op
            saturation_factor=1.0,  # No-op
            grayscale=False,        # No-op
            mean=None,              # No-op
            std=None                # No-op
        )
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, horizontal_flip_p={self.horizontal_flip_p}, same_on_batch={self.same_on_batch}, same_on_frame={self.same_on_frame})"


class TritonFusedAugment(nn.Module):
    """
    Fused augmentation: All operations in ONE kernel.
    
    This transform combines ALL augmentations in a single GPU kernel launch.
    **Unified Fusion:**
    Combines Affine (Rotation/Translation/Scale/Shear) + Crop + Flip + Color Jitter + Grayscale + Normalize
    in order into a single kernel launch.
    
    **Performance**: up to 14x faster than torchvision.transforms.Compose!
    
    Args:
        crop_size: Desired output size (int or tuple). If int, output is square (crop_size, crop_size).
                  If None, output size equals input size (no cropping). Default: None.
        horizontal_flip_p: Probability of horizontal flip (default: 0.0, no flip)
        
        # Affine parameters (optional - setting any enables affine mode)
        degrees: Rotation degrees range. If float, range is (-degrees, +degrees).
                If tuple, range is (degrees[0], degrees[1]). Default: 0 (no rotation).
        translate: Translation range as fraction of image size (tx, ty).
                  E.g., (0.1, 0.1) means translate up to 10% of width/height.
                  Default: None (no translation).
        scale: Scale range (min, max). E.g., (0.8, 1.2) scales between 80% and 120%.
              Default: None (no scaling).
        shear: Shear range in degrees. If float, range is (-shear, +shear) for x-axis.
              If tuple of 2, (shear[0], shear[1]) for x-axis.
              If tuple of 4, (shear[0], shear[1]) for x-axis and (shear[2], shear[3]) for y-axis.
              Default: None (no shearing).
        interpolation: Interpolation mode for affine (InterpolationMode.NEAREST or BILINEAR).
                      Only used in affine mode. Default: NEAREST.
        fill: Fill value for out-of-bounds pixels in affine mode. Default: 0.0.
        
        # Color jitter parameters (used in both modes)
        brightness: How much to jitter brightness. If float, chosen uniformly from [max(0, 1-brightness), 1+brightness].
                   If tuple, chosen uniformly from [brightness[0], brightness[1]].
        contrast: How much to jitter contrast (same format as brightness)
        saturation: How much to jitter saturation (same format as brightness)
        grayscale_p: Probability of converting to grayscale (default: 0.0, no grayscale)
        mean: Sequence of means for R, G, B channels. If None, normalization is skipped.
        std: Sequence of stds for R, G, B channels. If None, normalization is skipped.
        same_on_batch: If True, all images in batch (N dimension) share the same random parameters.
                      If False (default), each image gets different random parameters.
        same_on_frame: If True, all frames in a video (T dimension) share the same random parameters.
                      If False, each frame gets different random parameters.
                      Only applies to 5D input [N, T, C, H, W]. Default: True (consistent augmentation across frames).
        
    Example:
        ```python
        # Simple mode (crop + flip + color)
        transform = ta.TritonFusedAugment(
            crop_size=112,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # Affine mode (rotation + scale + color)
        transform = ta.TritonFusedAugment(
            crop_size=224,
            degrees=30,           # Enables affine mode
            scale=(0.8, 1.2),
            horizontal_flip_p=0.5,
            brightness=0.2,
            interpolation=InterpolationMode.BILINEAR
        )
        
        img = torch.rand(4, 3, 224, 224, device='cuda')
        result = transform(img)  # Single kernel launch!
        ```
    
    Note:
        - Uses FAST contrast (centered scaling), not torchvision's blend-with-mean
        - Input must be (C, H, W), (N, C, H, W), or (N, T, C, H, W) float tensor on CUDA in [0, 1] range
    """
    
    def __init__(
        self,
        crop_size: int | tuple[int, int] | None = None,
        horizontal_flip_p: float = 0.0,
        # Affine parameters (optional - enables affine mode if any are set)
        degrees: float | tuple[float, float] = 0,
        translate: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        shear: float | tuple[float, float] | None = None,
        interpolation = InterpolationMode.NEAREST,
        fill: float = 0.0,
        # Color jitter parameters
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        grayscale_p: float = 0.0,
        mean: Optional[tuple[float, float, float]] = None,
        std: Optional[tuple[float, float, float]] = None,
        same_on_batch: bool = False,
        same_on_frame: bool = True,
    ):
        super().__init__()
        
        # Parse crop size
        if crop_size is None:
            self.crop_height = self.crop_width = None
        elif isinstance(crop_size, int):
            self.crop_height = self.crop_width = crop_size
        else:
            self.crop_height, self.crop_width = crop_size
        
        self.horizontal_flip_p = horizontal_flip_p
        
        # Auto-detect affine mode
        self.has_affine = any([
            degrees != 0,
            translate is not None,
            scale is not None,
            shear is not None,
        ])
        
        # Store affine params
        if self.has_affine:
            self.affine_helper = TritonRandomAffine(
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=interpolation,
                fill=fill,
                same_on_batch=same_on_batch,
                same_on_frame=same_on_frame
            )
            self.interpolation = interpolation
            self.fill = fill

        # Store color params
        self.color_helper = TritonColorJitterNormalize(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            grayscale_p=grayscale_p,
            mean=mean,
            std=std,
            same_on_batch=same_on_batch,
            same_on_frame=same_on_frame
        )
        
        self.same_on_batch = same_on_batch
        self.same_on_frame = same_on_frame
    
    def _check_input(
        self,
        value: Optional[Union[float, Sequence[float]]],
        name: str,
        center: float = 1.0,
        bound: Tuple[float, float] = (0, float("inf")),
        clip_first_on_zero: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Check and process input parameters exactly like torchvision."""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, collections.abc.Sequence) and len(value) == 2:
            value = [float(v) for v in value]
        else:
            raise TypeError(f"{name}={value} should be a single number or a sequence with length 2.")
        
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound} and increasing, but got {value}.")
        
        return None if value[0] == value[1] == center else (float(value[0]), float(value[1]))
    
    
    
    def _get_params(self, param_count: int, device: torch.device, img_size: Tuple[int, int]):
        """
        Generate all parameters for the fused transformation.
        
        Args:
            param_count: Number of parameter sets to generate
            device: Device to generate parameters on
            img_size: (height, width) of the input image
            
        Returns:
            Tuple of all parameters required by fused_augment
        """
        img_height, img_width = img_size
        
        # Initialize params
        angle = 0.0
        translate = [0.0, 0.0]
        scale = 1.0
        shear = [0.0, 0.0]
        
        if self.has_affine:
            # ===== AFFINE MODE =====
            # Sample affine params
            angle, translate, scale, shear = self.affine_helper._get_params(
                param_count, device, (img_height, img_width)
            )
            
        # Sample crop params
        if self.crop_height is None:
            # No crop -> offsets are 0
            top_offsets = torch.zeros(param_count, device=device, dtype=torch.float32)
            left_offsets = torch.zeros(param_count, device=device, dtype=torch.float32)
        else:
            top_offsets = torch.randint(0, img_height - self.crop_height + 1, (param_count,), device=device, dtype=torch.float32)
            left_offsets = torch.randint(0, img_width - self.crop_width + 1, (param_count,), device=device, dtype=torch.float32)
        
        # Sample flip params
        do_flip = (
            F._sample_bernoulli_tensor(param_count, self.horizontal_flip_p, device)
            if self.horizontal_flip_p > 0
            else torch.zeros(param_count, device=device, dtype=torch.bool)
        )
        
        # Sample color params using helper
        brightness_factors, contrast_factors, saturation_factors, grayscale_mask = self.color_helper._get_params(param_count, device)
        
        return (
            angle, translate, scale, shear,
            top_offsets, left_offsets, do_flip,
            brightness_factors, contrast_factors, saturation_factors, grayscale_mask
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply all augmentations in a single fused kernel.
        
        Args:
            image: Input tensor of shape (C, H, W), (N, C, H, W), or (N, T, C, H, W)
                   Can be on CPU or CUDA (will be moved to CUDA automatically)
            
        Returns:
            Augmented tensor of same shape and device as input
        """
        # Normalize shape: 3D/4D/5D → 4D
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(image)
        
        # Move to CUDA if not already (Triton requires CUDA)
        if not normalized_img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Triton-Augment requires CUDA, but CUDA is not available. "
                    "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
                )
            normalized_img = normalized_img.cuda()
        
        total_samples, _, img_height, img_width = normalized_img.shape
        
        # Compute how many parameter sets to generate
        param_count = _compute_param_count(batch_size, num_frames, self.same_on_batch, self.same_on_frame)
        
        # Get all parameters
        (
            angle, translate, scale, shear,
            top_offsets, left_offsets, do_flip,
            brightness_factors, contrast_factors, saturation_factors, grayscale_mask
        ) = self._get_params(param_count, normalized_img.device, (img_height, img_width))
        
        # Broadcast params
        if self.has_affine:
            angle = _broadcast_params_to_all_samples(angle, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
            translate = _broadcast_2d_params(translate, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
            shear = _broadcast_2d_params(shear, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
            scale = _broadcast_params_to_all_samples(scale, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)

        top_offsets = _broadcast_params_to_all_samples(top_offsets, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
        left_offsets = _broadcast_params_to_all_samples(left_offsets, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
        do_flip = _broadcast_params_to_all_samples(do_flip, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
        brightness_factors = _broadcast_params_to_all_samples(brightness_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
        contrast_factors = _broadcast_params_to_all_samples(contrast_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
        saturation_factors = _broadcast_params_to_all_samples(saturation_factors, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
        grayscale_mask = _broadcast_params_to_all_samples(grayscale_mask, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame)
        
        # Call unified fused_augment
        result = F.fused_augment(
            normalized_img,
            top=top_offsets,
            left=left_offsets,
            height=self.crop_height if self.crop_height is not None else img_height,
            width=self.crop_width if self.crop_width is not None else img_width,
            flip_horizontal=do_flip,
            # Affine params
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=self.interpolation if self.has_affine else "nearest",
            fill=self.fill if self.has_affine else 0.0,
            # Color params
            brightness_factor=brightness_factors,
            contrast_factor=contrast_factors,
            saturation_factor=saturation_factors,
            grayscale=grayscale_mask,
            mean=self.color_helper.mean,
            std=self.color_helper.std,
        )
        
        # Reshape back to original shape
        return _reshape_to_original(result, original_shape, was_3d)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.crop_height is not None:
            format_string += f'crop_size=({self.crop_height}, {self.crop_width})'
        else:
            format_string += 'crop_size=None'
        format_string += f', horizontal_flip_p={self.horizontal_flip_p}'
        if self.brightness:
            format_string += f', brightness={self.brightness}'
        if self.contrast:
            format_string += f', contrast={self.contrast}'
        if self.saturation:
            format_string += f', saturation={self.saturation}'
        if self.grayscale_p > 0:
            format_string += f', grayscale_p={self.grayscale_p}'
        if self.color_helper.mean is not None:
            format_string += f', mean={self.color_helper.mean}'
        if self.color_helper.std is not None:
            format_string += f', std={self.color_helper.std}'
        format_string += ')'
        return format_string

class TritonRandomAffine(nn.Module):
    """
    Random affine transformation of the image keeping center invariant.
    
    GPU-accelerated implementation using Triton kernels. Matches the API of
    torchvision.transforms.v2.RandomAffine.
    
    Supports:
        - 3D input: (C, H, W) - single image
        - 4D input: (N, C, H, W) - batch of images  
        - 5D input: (N, T, C, H, W) - batch of videos
    
    Args:
        degrees: Range of degrees to select from. If degrees is a number instead of
            sequence like (min, max), the range of degrees will be (-degrees, +degrees).
            Set to 0 to deactivate rotations.
        translate: Tuple of maximum absolute fraction for horizontal and vertical
            translations. For example translate=(a, b), then horizontal shift is
            randomly sampled in the range -img_width * a < dx < img_width * a and
            vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b.
            Will not translate by default.
        scale: Scaling factor interval, e.g (a, b), then scale is randomly sampled
            from the range a <= scale <= b. Will keep original scale by default.
        shear: Range of degrees to select from. If shear is a number, a shear parallel
            to the x axis in the range (-shear, +shear) will be applied. Else if shear
            is a tuple of 2 values, a x-axis shear in (shear[0], shear[1]) will be applied.
            Else if shear is a tuple of 4 values, a x-axis shear in (shear[0], shear[1])
            and y-axis shear in (shear[2], shear[3]) will be applied. Default: None.
        interpolation: Interpolation mode for sampling. Either:
            - InterpolationMode.NEAREST (default): Nearest neighbor, faster.
            - InterpolationMode.BILINEAR: Bilinear interpolation, smoother.
        fill: Constant fill value for areas outside the transformed image. Default: 0.
        center: Optional center of rotation (x, y) in pixel coordinates. Origin is the
            upper left corner. Default is the center of the image.
        same_on_batch: If True, all images in batch share the same random parameters.
            Default: False.
        same_on_frame: If True, all frames in a video (5D input) share the same random
            parameters. Default: True.
            
    Note:
        For nearest neighbor interpolation, there may be minor differences compared
        to torchvision at exact pixel boundaries due to floating-point rounding.
        Bilinear interpolation does not have this limitation.
        
    Example:
        ```python
        transform = TritonRandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10,
            interpolation=InterpolationMode.BILINEAR
        )
        
        # Apply to batch of images
        img = torch.rand(4, 3, 224, 224, device='cuda')
        result = transform(img)
        
        # Apply to video with same transform per frame
        video = torch.rand(2, 8, 3, 112, 112, device='cuda')
        result = transform(video)
        ```
    """
    
    def __init__(
        self,
        degrees: Union[float, Sequence[float]],
        translate: Optional[Tuple[float, float]] = None,
        scale: Optional[Tuple[float, float]] = None,
        shear: Optional[Union[float, Sequence[float]]] = None,
        interpolation = InterpolationMode.NEAREST,
        fill: float = 0.0,
        center: Optional[Tuple[int, int]] = None,
        same_on_batch: bool = False,
        same_on_frame: bool = True,
    ):
        super().__init__()
        
        # Process degrees
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-float(degrees), float(degrees))
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of length 2.")
            self.degrees = (float(degrees[0]), float(degrees[1]))
            
        # Process translate
        if translate is not None:
            if len(translate) != 2:
                raise ValueError("translate should be a sequence of length 2.")
            if not (0.0 <= translate[0] <= 1.0 and 0.0 <= translate[1] <= 1.0):
                raise ValueError("translation values should be between 0 and 1")
            self.translate = (float(translate[0]), float(translate[1]))
        else:
            self.translate = None
            
        # Process scale
        if scale is not None:
            if len(scale) != 2:
                raise ValueError("scale should be a sequence of length 2.")
            if scale[0] > scale[1]:
                raise ValueError("scale should be (min, max)")
            self.scale = (float(scale[0]), float(scale[1]))
        else:
            self.scale = None
            
        # Process shear
        if shear is not None:
            if isinstance(shear, (int, float)):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-float(shear), float(shear), 0.0, 0.0)
            else:
                if len(shear) == 2:
                    self.shear = (float(shear[0]), float(shear[1]), 0.0, 0.0)
                elif len(shear) == 4:
                    self.shear = (float(shear[0]), float(shear[1]), float(shear[2]), float(shear[3]))
                else:
                    raise ValueError("shear should be a single number or a sequence of length 2 or 4.")
        else:
            self.shear = None
            
        self.interpolation = interpolation
        self.fill = float(fill)
        self.center = center
        self.same_on_batch = same_on_batch
        self.same_on_frame = same_on_frame

    def _get_params(self, param_count: int, device: torch.device, img_size: Tuple[int, int]):
        """Get parameters for affine transformation."""
        height, width = img_size
        
        # Angle
        angle = F._sample_uniform_tensor(
            param_count, self.degrees[0], self.degrees[1], device
        )
        
        # Translate
        if self.translate is not None:
            max_dx = float(self.translate[0] * width)
            max_dy = float(self.translate[1] * height)
            tx = F._sample_uniform_tensor(param_count, -max_dx, max_dx, device)
            ty = F._sample_uniform_tensor(param_count, -max_dy, max_dy, device)
            translate = torch.stack([tx, ty], dim=1)
        else:
            translate = torch.zeros((param_count, 2), device=device)
            
        # Scale
        if self.scale is not None:
            scale = F._sample_uniform_tensor(
                param_count, self.scale[0], self.scale[1], device
            )
        else:
            scale = torch.ones(param_count, device=device)
            
        # Shear
        if self.shear is not None:
            sx = F._sample_uniform_tensor(
                param_count, self.shear[0], self.shear[1], device
            )
            sy = F._sample_uniform_tensor(
                param_count, self.shear[2], self.shear[3], device
            )
            shear = torch.stack([sx, sy], dim=1)
        else:
            shear = torch.zeros((param_count, 2), device=device)
            
        return angle, translate, scale, shear

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation.
        """
        # Normalize shape
        normalized_img, batch_size, num_frames, original_shape, was_3d = _normalize_video_shape(img)
        
        if not normalized_img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton-Augment requires CUDA")
            normalized_img = normalized_img.cuda()
            
        total_samples = normalized_img.shape[0]
        _, _, height, width = normalized_img.shape
        
        # Compute param count
        param_count = _compute_param_count(batch_size, num_frames, self.same_on_batch, self.same_on_frame)
        
        # Sample params
        angle, translate, scale, shear = self._get_params(param_count, normalized_img.device, (height, width))
        
        # Broadcast params
        angle = _broadcast_params_to_all_samples(
            angle, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        translate = _broadcast_2d_params(
            translate, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        shear = _broadcast_2d_params(
            shear, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
            
        scale = _broadcast_params_to_all_samples(
            scale, batch_size, num_frames, total_samples, self.same_on_batch, self.same_on_frame
        )
        
        # Apply affine transform using F.affine
        output = F.affine(
            normalized_img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=self.interpolation,
            fill=self.fill,
            center=self.center
        )
        
        return _reshape_to_original(output, original_shape, was_3d)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"degrees={self.degrees}, "
            f"translate={self.translate}, "
            f"scale={self.scale}, "
            f"shear={self.shear}, "
            f"interpolation={self.interpolation}, "
            f"fill={self.fill}, "
            f"center={self.center}, "
            f"same_on_batch={self.same_on_batch}, "
            f"same_on_frame={self.same_on_frame})"
        )




class TritonRandomRotation(TritonRandomAffine):
    """
    Random rotation of the image.
    
    GPU-accelerated implementation using Triton kernels. Matches the API of
    torchvision.transforms.v2.RandomRotation.
    
    Supports:
        - 3D input: (C, H, W) - single image
        - 4D input: (N, C, H, W) - batch of images
        - 5D input: (N, T, C, H, W) - batch of videos
    
    Args:
        degrees: Range of degrees to select from. If degrees is a number instead of
            sequence like (min, max), the range of degrees will be (-degrees, +degrees).
        interpolation: Interpolation mode for sampling. Either:
            - InterpolationMode.NEAREST (default): Nearest neighbor, faster.
            - InterpolationMode.BILINEAR: Bilinear interpolation, smoother.
        expand: If True, expands the output to hold the entire rotated image.
            Currently not supported (raises NotImplementedError).
        center: Optional center of rotation (x, y) in pixel coordinates. Origin is the
            upper left corner. Default is the center of the image.
        fill: Constant fill value for areas outside the rotated image. Default: 0.
        same_on_batch: If True, all images in batch share the same random rotation angle.
            Default: False.
        same_on_frame: If True, all frames in a video (5D input) share the same random
            rotation angle. Default: True.
            
    Note:
        For nearest neighbor interpolation, there may be minor differences compared
        to torchvision at exact pixel boundaries due to floating-point rounding.
        Bilinear interpolation does not have this limitation.
        
    Example:
        ```python
        transform = TritonRandomRotation(
            degrees=30,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.5
        )
        
        # Apply to batch of images
        img = torch.rand(4, 3, 224, 224, device='cuda')
        result = transform(img)
        ```
    """
    
    def __init__(
        self,
        degrees: Union[float, Sequence[float]],
        interpolation = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[Tuple[int, int]] = None,
        fill: float = 0.0,
        same_on_batch: bool = False,
        same_on_frame: bool = True,
    ):
        if expand:
            raise NotImplementedError("expand=True is not yet supported in Triton-Augment")
            
        super().__init__(
            degrees=degrees,
            translate=None,
            scale=None,
            shear=None,
            interpolation=interpolation,
            fill=fill,
            center=center,
            same_on_batch=same_on_batch,
            same_on_frame=same_on_frame,
        )
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"degrees={self.degrees}, "
            f"interpolation={self.interpolation}, "
            f"fill={self.fill}, "
            f"same_on_batch={self.same_on_batch}, "
            f"same_on_frame={self.same_on_frame})"
        )


__all__ = [
    # Color transforms
    'TritonColorJitter',
    'TritonNormalize',
    'TritonColorJitterNormalize',
    'TritonGrayscale',
    'TritonRandomGrayscale',
    # Geometric transforms
    'TritonRandomCrop',
    'TritonCenterCrop',
    'TritonRandomHorizontalFlip',
    'TritonRandomCropFlip',
    'TritonRandomRotation',
    'TritonRandomAffine',
    # Ultimate fusion
    'TritonFusedAugment',
]
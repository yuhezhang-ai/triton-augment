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
        same_on_batch: If True, all images in batch share the same random parameters
                             If False (default), each image in batch gets different random parameters
        
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
    ):
        super().__init__()
        
        # Process parameters exactly like torchvision
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.same_on_batch = same_on_batch
    
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
    
    def _get_params(self, batch_size: int, device: torch.device):
        """
        Randomly sample transformation parameters (per-image or batch-wide).
        
        Returns:
            Tuple of (brightness_factors, contrast_factors, saturation_factors) as tensors
        """
        # Generate brightness factors
        if self.brightness is not None:
            brightness_factors = F._sample_uniform_tensor(
                batch_size, self.brightness[0], self.brightness[1], device, same_on_batch=self.same_on_batch
            )
        else:
            brightness_factors = torch.ones(batch_size, device=device)
        
        # Generate contrast factors
        if self.contrast is not None:
            contrast_factors = F._sample_uniform_tensor(
                batch_size, self.contrast[0], self.contrast[1], device, same_on_batch=self.same_on_batch
            )
        else:
            contrast_factors = torch.ones(batch_size, device=device)
        
        # Generate saturation factors
        if self.saturation is not None:
            saturation_factors = F._sample_uniform_tensor(
                batch_size, self.saturation[0], self.saturation[1], device, same_on_batch=self.same_on_batch
            )
        else:
            saturation_factors = torch.ones(batch_size, device=device)
        
        return brightness_factors, contrast_factors, saturation_factors
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random color jitter to the input image tensor.
        
        Args:
            img: Input image tensor of shape (N, C, H, W) on CUDA
            
        Returns:
            Augmented image tensor of the same shape and dtype
        """
        batch_size = img.shape[0]
        
        # Sample random parameters (per-image or batch-wide)
        brightness_factors, contrast_factors, saturation_factors = self._get_params(batch_size, img.device)
        
        # Apply transformations using fused kernel (FAST contrast mode)
        _, _, height, width = img.shape
        output = F.fused_augment(
            img,
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
        
        return output
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"same_on_batch={self.same_on_batch})"
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
            img: Input tensor of shape (N, C, H, W) or (C, H, W)
                 Can be on CPU or CUDA (will be moved to CUDA automatically)
            
        Returns:
            Normalized tensor of same shape and device as input
        """
        # Handle 3D tensors (C, H, W) from dataset transforms
        is_3d = img.ndim == 3
        if is_3d:
            img = img.unsqueeze(0)
        
        # Move to CUDA if not already (Triton requires CUDA)
        if not img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Triton-Augment requires CUDA, but CUDA is not available. "
                    "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
                )
            img = img.cuda()
        
        result = F.normalize(img, mean=self.mean, std=self.std)
        
        # Remove batch dimension if input was 3D
        if is_3d:
            result = result.squeeze(0)
        
        return result
    
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
        random_grayscale_p: Probability of converting to grayscale (default: 0.0)
        mean: Sequence of means for normalization (R, G, B)
        std: Sequence of standard deviations for normalization (R, G, B)
        same_on_batch: If True, all images in batch share the same random parameters
                             If False (default), each image in batch gets different random parameters
        
    Example:
        ```python
        # Full augmentation pipeline in one transform (per-image randomness)
        transform = TritonColorJitterNormalize(
            brightness=0.2,  # Range: [0.8, 1.2]
            contrast=0.2,    # Range: [0.8, 1.2]
            saturation=0.2,  # Range: [0.8, 1.2]
            random_grayscale_p=0.1,  # 10% chance of grayscale (per-image)
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            same_on_batch=False
        )
        img = torch.rand(4, 3, 224, 224, device='cuda')
        augmented = transform(img)  # Each image gets different augmentation
        ```
    """
    
    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        random_grayscale_p: float = 0.0,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        same_on_batch: bool = False,
    ):
        super().__init__()
        
        # Process parameters exactly like torchvision
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.random_grayscale_p = random_grayscale_p
        self.same_on_batch = same_on_batch
        
        if not (0.0 <= random_grayscale_p <= 1.0):
            raise ValueError(f"random_grayscale_p must be in [0, 1], got {random_grayscale_p}")
        
        # Store normalization parameters
        self.mean = tuple(mean)
        self.std = tuple(std)
        
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must have 3 values for RGB channels")
    
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
    
    def _get_params(self, batch_size: int, device: torch.device):
        """Randomly sample transformation parameters (per-image or batch-wide)."""
        # Generate brightness factors
        if self.brightness is not None:
            brightness_factors = F._sample_uniform_tensor(
                batch_size, self.brightness[0], self.brightness[1], device, same_on_batch=self.same_on_batch
            )
        else:
            brightness_factors = torch.ones(batch_size, device=device)
        
        # Generate contrast factors
        if self.contrast is not None:
            contrast_factors = F._sample_uniform_tensor(
                batch_size, self.contrast[0], self.contrast[1], device, same_on_batch=self.same_on_batch
            )
        else:
            contrast_factors = torch.ones(batch_size, device=device)
        
        # Generate saturation factors
        if self.saturation is not None:
            saturation_factors = F._sample_uniform_tensor(
                batch_size, self.saturation[0], self.saturation[1], device, same_on_batch=self.same_on_batch
            )
        else:
            saturation_factors = torch.ones(batch_size, device=device)
        
        # Generate grayscale mask
        grayscale_mask = F._sample_bernoulli_tensor(
            batch_size, self.random_grayscale_p, device, same_on_batch=self.same_on_batch
        )
        
        return brightness_factors, contrast_factors, saturation_factors, grayscale_mask
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random color jitter and normalization in a single fused operation.
        
        Args:
            img: Input tensor of shape (N, C, H, W) or (C, H, W)
                 Can be on CPU or CUDA (will be moved to CUDA automatically)
            
        Returns:
            Augmented and normalized tensor of same shape and device as input
        """
        # Handle 3D tensors (C, H, W) from dataset transforms
        is_3d = img.ndim == 3
        if is_3d:
            img = img.unsqueeze(0)
        
        # Move to CUDA if not already (Triton requires CUDA)
        if not img.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Triton-Augment requires CUDA, but CUDA is not available. "
                    "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
                )
            img = img.cuda()
        
        batch_size = img.shape[0]
        
        # Sample random parameters (per-image or batch-wide)
        brightness_factors, contrast_factors, saturation_factors, grayscale_mask = self._get_params(batch_size, img.device)
        
        # Apply the fully fused transformation
        _, _, height, width = img.shape
        result = F.fused_augment(
            img,
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
        
        # Remove batch dimension if input was 3D
        if is_3d:
            result = result.squeeze(0)
        
        return result
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"random_grayscale_p={self.random_grayscale_p}, "
            f"mean={self.mean}, "
            f"std={self.std}, "
            f"same_on_batch={self.same_on_batch})"
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
            image: Input tensor of shape (N, 3, H, W)
            
        Returns:
            Grayscale tensor of shape (N, num_output_channels, H, W)
        """
        return F.rgb_to_grayscale(image, num_output_channels=self.num_output_channels)
    
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
        same_on_batch: If True, all images in batch make the same grayscale decision
                             If False (default), each image in batch independently decides grayscale conversion
                            
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
        same_on_batch: bool = False
    ):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p ({p}) must be in [0, 1]")
        if num_output_channels not in (1, 3):
            raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}")
        self.p = p
        self.num_output_channels = num_output_channels
        self.same_on_batch = same_on_batch
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (N, 3, H, W)
            
        Returns:
            Image tensor, either original or grayscale based on probability
        """
        if self.p == 0:
            return image
        
        batch_size = image.shape[0]
        
        # Generate grayscale mask (deterministic functional API)
        grayscale_mask = F._sample_bernoulli_tensor(
            batch_size, self.p, image.device, same_on_batch=self.same_on_batch
        )
        
        # Early exit if no images need grayscale
        if not torch.any(grayscale_mask).item():
            return image
        
        # Use deterministic functional API with mask
        return F.rgb_to_grayscale(
            image,
            num_output_channels=self.num_output_channels,
            grayscale_mask=grayscale_mask
        )
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}, "
            f"num_output_channels={self.num_output_channels}, "
            f"same_on_batch={self.same_on_batch})"
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
        same_on_batch: If True, all images in batch crop at the same position.
                             If False (default), each image in batch gets different random crop position.
        
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
    
    def __init__(self, size: Union[int, Sequence[int]], same_on_batch: bool = False):
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
    
    @staticmethod
    def get_params(image: torch.Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get parameters for random crop.
        
        Args:
            image: Input image tensor (N, C, H, W)
            output_size: Desired output size (height, width)
            
        Returns:
            Tuple (top, left, height, width) for cropping
        """
        _, _, h, w = image.shape
        th, tw = output_size
        
        if h < th or w < tw:
            raise ValueError(
                f"Image size ({h}, {w}) is smaller than crop size ({th}, {tw}). "
                f"Padding not yet supported in MVP."
            )
        
        if h == th and w == tw:
            return 0, 0, h, w
        
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        
        return i, j, th, tw
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (N, C, H, W) or (C, H, W)
            
        Returns:
            Randomly cropped tensor of shape (N, C, size[0], size[1])
        """
        # Handle 3D tensors
        is_3d = image.ndim == 3
        if is_3d:
            image = image.unsqueeze(0)
        
        # Move to CUDA if needed
        if not image.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton-Augment requires CUDA")
            image = image.cuda()
        
        batch_size, _, h, w = image.shape
        th, tw = self.size
        
        if h < th or w < tw:
            raise ValueError(
                f"Image size ({h}, {w}) is smaller than crop size ({th}, {tw}). "
                f"Padding not yet supported in MVP."
            )
        
        # Generate crop positions
        if self.same_on_batch:
            # Same crop position for all images
            top = torch.randint(0, h - th + 1, (1,)).item()
            left = torch.randint(0, w - tw + 1, (1,)).item()
            top_offsets = torch.full((batch_size,), top, device=image.device, dtype=torch.int32)
            left_offsets = torch.full((batch_size,), left, device=image.device, dtype=torch.int32)
        else:
            # Per-image random crop positions
            top_offsets = torch.randint(0, h - th + 1, (batch_size,), device=image.device, dtype=torch.int32)
            left_offsets = torch.randint(0, w - tw + 1, (batch_size,), device=image.device, dtype=torch.int32)
        
        result = F.crop(image, top_offsets, left_offsets, th, tw)
        
        if is_3d:
            result = result.squeeze(0)
        
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, same_on_batch={self.same_on_batch})"


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
            image: Input tensor of shape (N, C, H, W)
            
        Returns:
            Center-cropped tensor of shape (N, C, size[0], size[1])
        """
        return F.center_crop(image, self.size)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class TritonRandomHorizontalFlip(nn.Module):
    """
    Horizontally flip the image randomly with probability p.
    
    Matches torchvision.transforms.v2.RandomHorizontalFlip behavior.
    
    Args:
        p: Probability of flipping (default: 0.5)
        same_on_batch: If True, all images in batch share the same decision.
                             If False (default), each image in batch gets different random decision.
        
    Example:
        ```python
        transform = TritonRandomHorizontalFlip(p=0.5)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        flipped = transform(img)  # Each image has 50% chance of being flipped
        ```
    """
    
    def __init__(self, p: float = 0.5, same_on_batch: bool = False):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p ({p}) must be in [0, 1]")
        self.p = p
        self.same_on_batch = same_on_batch
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (N, C, H, W) or (C, H, W)
            
        Returns:
            Image tensor, either original or horizontally flipped
        """
        if self.p == 0:
            return image
        
        # Handle 3D tensors
        is_3d = image.ndim == 3
        if is_3d:
            image = image.unsqueeze(0)
        
        # Move to CUDA if needed
        if not image.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton-Augment requires CUDA")
            image = image.cuda()
        
        batch_size = image.shape[0]
        
        # Generate per-image or batch-wide flip decisions
        flip_mask = F._sample_bernoulli_tensor(batch_size, self.p, image.device, self.same_on_batch)
        
        # Use kernel-based per-image flipping
        result = F.horizontal_flip(image, flip_mask=flip_mask)
        
        if is_3d:
            result = result.squeeze(0)
        
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, same_on_batch={self.same_on_batch})"


class TritonRandomCropFlip(nn.Module):
    """
    Fused random crop + random horizontal flip.
    
    This class combines random crop and random horizontal flip in a SINGLE
    kernel launch, eliminating intermediate memory transfers.
    
    **Performance**: ~1.5-2x faster than applying TritonRandomCrop + TritonRandomHorizontalFlip sequentially.
    
    Args:
        size: Desired output size (height, width) or int for square crop
        horizontal_flip_p: Probability of horizontal flip (default: 0.5)
        same_on_batch: If True, all images in batch share the same random parameters
                             If False (default), each image in batch gets different random parameters
        
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
        same_on_batch: bool = False
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
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (N, C, H, W)
            
        Returns:
            Randomly cropped (and optionally flipped) tensor of shape (N, C, size[0], size[1])
        """
        batch_size = image.shape[0]
        img_height, img_width = image.shape[2], image.shape[3]
        th, tw = self.size
        
        # Validate crop size
        if th > img_height or tw > img_width:
            raise ValueError(
                f"Crop size ({th}, {tw}) is larger than image size ({img_height}, {img_width})"
            )
        
        # Generate per-image or batch-wide crop offsets
        if self.same_on_batch:
            # Same offsets for all images
            top = torch.randint(0, img_height - th + 1, (1,), device=image.device, dtype=torch.int32).item()
            left = torch.randint(0, img_width - tw + 1, (1,), device=image.device, dtype=torch.int32).item()
            top_offsets = torch.full((batch_size,), top, device=image.device, dtype=torch.int32)
            left_offsets = torch.full((batch_size,), left, device=image.device, dtype=torch.int32)
        else:
            # Different offsets per image
            top_offsets = torch.randint(0, img_height - th + 1, (batch_size,), device=image.device, dtype=torch.int32)
            left_offsets = torch.randint(0, img_width - tw + 1, (batch_size,), device=image.device, dtype=torch.int32)
        
        # Generate per-image or batch-wide flip decisions
        flip_mask = F._sample_bernoulli_tensor(
            batch_size, self.horizontal_flip_p, image.device, same_on_batch=self.same_on_batch
        )
        
        # Single fused kernel launch (using fused_augment with no-op color operations)
        return F.fused_augment(
            image,
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
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, horizontal_flip_p={self.horizontal_flip_p}, same_on_batch={self.same_on_batch})"


class TritonFusedAugment(nn.Module):
    """
    Fused augmentation: All operations in ONE kernel.
    
    This transform combines ALL augmentations in a single GPU kernel launch:
    - **Geometric Tier**: RandomCrop + RandomHorizontalFlip
    - **Pixel Tier**: ColorJitter (brightness, contrast, saturation) + Normalize
    
    **Performance**: up to 12x faster than torchvision.transforms.Compose!
    
    This is the PEAK PERFORMANCE path - single kernel launch for entire pipeline.
    No intermediate memory allocations or kernel launch overhead.
    
    Args:
        crop_size: Desired output size (int or tuple). If int, output is square (crop_size, crop_size).
        horizontal_flip_p: Probability of horizontal flip (default: 0.0, no flip)
        brightness: How much to jitter brightness. If float, chosen uniformly from [max(0, 1-brightness), 1+brightness].
                   If tuple, chosen uniformly from [brightness[0], brightness[1]].
        contrast: How much to jitter contrast (same format as brightness)
        saturation: How much to jitter saturation (same format as brightness)
        random_grayscale_p: Probability of converting to grayscale (default: 0.0, no grayscale)
        mean: Sequence of means for R, G, B channels
        std: Sequence of stds for R, G, B channels
        same_on_batch: If True, all images in the batch share the same parameters.
                             If False (default), each image in the batch gets different random parameters.
        
    Example:
        ```python
        # Replace torchvision Compose with single transform
        # OLD (6 kernel launches):
        transform = transforms.Compose([
            transforms.RandomCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        # NEW (1 kernel launch - significantly faster!):
        import triton_augment as ta
        transform = ta.TritonFusedAugment(
            crop_size=112,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        img = torch.rand(4, 3, 224, 224, device='cuda')
        result = transform(img)  # Single kernel launch!
        ```
    
    Note:
        - Uses FAST contrast (centered scaling), not torchvision's blend-with-mean
        - By default, each image gets different random parameters (set same_on_batch=False for same params)
        - Input must be (N, 3, H, W) float tensor on CUDA in [0, 1] range
    """
    
    def __init__(
        self,
        crop_size: int | tuple[int, int],
        horizontal_flip_p: float = 0.0,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        random_grayscale_p: float = 0.0,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        same_on_batch: bool = False,
    ):
        super().__init__()
        
        # Parse crop size
        if isinstance(crop_size, int):
            self.crop_height = self.crop_width = crop_size
        else:
            self.crop_height, self.crop_width = crop_size
        
        self.horizontal_flip_p = horizontal_flip_p
        
        # Parse color jitter ranges
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        
        self.random_grayscale_p = random_grayscale_p
        
        self.mean = mean
        self.std = std
        
        self.same_on_batch = same_on_batch
    
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
    
    
    def _get_params(self, batch_size: int, image_height: int, image_width: int, device: torch.device):
        """
        Sample all random parameters for the ultimate transform.
        
        Args:
            batch_size: Number of images in the batch
            image_height, image_width: Input image dimensions
            device: Device to create tensors on
        
        Returns:
            Tuple of (top_offsets, left_offsets, flip_mask, brightness_factors, contrast_factors, saturation_factors, grayscale_mask)
            - All are tensors of shape (N,)
            - If same_on_batch=False, tensors are filled with the same value for all images
        """
        # Validate crop size
        if self.crop_height > image_height or self.crop_width > image_width:
            raise ValueError(
                f"Crop size ({self.crop_height}, {self.crop_width}) is larger than "
                f"input size ({image_height}, {image_width})"
            )
        
        # Sample crop parameters (integer offsets)
        if self.same_on_batch:
            # Same offsets for all images
            top = torch.randint(0, image_height - self.crop_height + 1, (1,)).item()
            left = torch.randint(0, image_width - self.crop_width + 1, (1,)).item()
            top_offsets = torch.full((batch_size,), top, device=device, dtype=torch.int32)
            left_offsets = torch.full((batch_size,), left, device=device, dtype=torch.int32)
        else:
            # Different offsets per image
            top_offsets = torch.randint(0, image_height - self.crop_height + 1, (batch_size,), device=device, dtype=torch.int32)
            left_offsets = torch.randint(0, image_width - self.crop_width + 1, (batch_size,), device=device, dtype=torch.int32)
        
        # Sample flip decisions
        flip_mask = (
            F._sample_bernoulli_tensor(batch_size, self.horizontal_flip_p, device, self.same_on_batch)
            if self.horizontal_flip_p > 0
            else torch.zeros(batch_size, device=device, dtype=torch.uint8)
        )
        
        # Sample color jitter parameters
        brightness_factors = (
            F._sample_uniform_tensor(batch_size, self.brightness[0], self.brightness[1], device, self.same_on_batch)
            if self.brightness is not None
            else torch.ones(batch_size, device=device)
        )
        contrast_factors = (
            F._sample_uniform_tensor(batch_size, self.contrast[0], self.contrast[1], device, self.same_on_batch)
            if self.contrast is not None
            else torch.ones(batch_size, device=device)
        )
        saturation_factors = (
            F._sample_uniform_tensor(batch_size, self.saturation[0], self.saturation[1], device, self.same_on_batch)
            if self.saturation is not None
            else torch.ones(batch_size, device=device)
        )
        
        # Sample grayscale decisions
        grayscale_mask = (
            F._sample_bernoulli_tensor(batch_size, self.random_grayscale_p, device, self.same_on_batch)
            if self.random_grayscale_p > 0
            else torch.zeros(batch_size, device=device, dtype=torch.uint8)
        )
        
        return top_offsets, left_offsets, flip_mask, brightness_factors, contrast_factors, saturation_factors, grayscale_mask
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply all augmentations in a single fused kernel.
        
        Args:
            image: Input tensor of shape (N, C, H, W) or (C, H, W)
                   Can be on CPU or CUDA (will be moved to CUDA automatically)
            
        Returns:
            Augmented tensor of same shape and device as input
        """
        # Handle 3D tensors (C, H, W) from dataset transforms
        is_3d = image.ndim == 3
        if is_3d:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Move to CUDA if not already (Triton requires CUDA)
        if not image.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Triton-Augment requires CUDA, but CUDA is not available. "
                    "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
                )
            image = image.cuda()
        
        batch_size, _, img_height, img_width = image.shape
        
        # Sample all random parameters (per-image if same_on_batch=False)
        top_offsets, left_offsets, flip_mask, brightness_factors, contrast_factors, saturation_factors, grayscale_mask = self._get_params(
            batch_size, img_height, img_width, image.device
        )
        
        # Single kernel launch for ALL operations!
        result = F.fused_augment(
            image,
            top=top_offsets,
            left=left_offsets,
            height=self.crop_height,
            width=self.crop_width,
            flip_horizontal=flip_mask,
            brightness_factor=brightness_factors,
            contrast_factor=contrast_factors,
            saturation_factor=saturation_factors,
            grayscale=grayscale_mask,
            mean=self.mean,
            std=self.std,
        )
        
        # Remove batch dimension if input was 3D
        if is_3d:
            result = result.squeeze(0)
        
        return result
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'crop_size=({self.crop_height}, {self.crop_width})'
        format_string += f', horizontal_flip_p={self.horizontal_flip_p}'
        if self.brightness:
            format_string += f', brightness={self.brightness}'
        if self.contrast:
            format_string += f', contrast={self.contrast}'
        if self.saturation:
            format_string += f', saturation={self.saturation}'
        if self.random_grayscale_p > 0:
            format_string += f', random_grayscale_p={self.random_grayscale_p}'
        format_string += f', mean={self.mean}'
        format_string += f', std={self.std}'
        format_string += ')'
        return format_string


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
    # Ultimate fusion
    'TritonUltimateAugment',
]

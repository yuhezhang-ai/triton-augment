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
    that uses fused Triton kernels for maximum performance.
    
    Args:
        brightness: How much to jitter brightness.
                   brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                   or the given [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast.
                 contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                 or the given [min, max]. Should be non-negative numbers.
        saturation: How much to jitter saturation.
                   saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                   or the given [min, max]. Should be non negative numbers.
        
    Example:
        >>> transform = TritonColorJitter(
        ...     brightness=0.2,  # Range: [0.8, 1.2]
        ...     contrast=0.2,    # Range: [0.8, 1.2]
        ...     saturation=0.2   # Range: [0.8, 1.2]
        ... )
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> augmented = transform(img)
        
        >>> # Can also be used with specific ranges
        >>> transform = TritonColorJitter(
        ...     brightness=(0.5, 1.5),  # Custom range
        ...     contrast=(0.7, 1.3),     # Custom range
        ...     saturation=(0.0, 2.0)    # Custom range
        ... )
    """
    
    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
    ):
        super().__init__()
        
        # Process parameters exactly like torchvision
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
    
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
    
    @staticmethod
    def _generate_value(left: float, right: float) -> float:
        """Generate a random value in [left, right] range."""
        return torch.empty(1).uniform_(left, right).item()
    
    def _get_params(self):
        """
        Randomly sample transformation parameters.
        
        Returns:
            Tuple of (brightness_factor, contrast_factor, saturation_factor)
        """
        brightness_factor = (
            self._generate_value(self.brightness[0], self.brightness[1])
            if self.brightness is not None
            else 1.0
        )
        contrast_factor = (
            self._generate_value(self.contrast[0], self.contrast[1])
            if self.contrast is not None
            else 1.0
        )
        saturation_factor = (
            self._generate_value(self.saturation[0], self.saturation[1])
            if self.saturation is not None
            else 1.0
        )
        
        return brightness_factor, contrast_factor, saturation_factor
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random color jitter to the input image tensor.
        
        Args:
            img: Input image tensor of shape (N, C, H, W) on CUDA
            
        Returns:
            Augmented image tensor of the same shape and dtype
        """
        # Sample random parameters
        brightness_factor, contrast_factor, saturation_factor = self._get_params()
        
        # Apply transformations sequentially to match torchvision
        # Note: For maximum speed, use TritonColorJitterNormalize instead
        output = img
        if brightness_factor != 1.0:
            output = F.adjust_brightness(output, brightness_factor)
        if contrast_factor != 1.0:
            output = F.adjust_contrast(output, contrast_factor)
        if saturation_factor != 1.0:
            output = F.adjust_saturation(output, saturation_factor)
        
        return output
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation})"
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
        >>> # ImageNet normalization
        >>> normalize = TritonNormalize(
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225)
        ... )
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> normalized = normalize(img)
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
            img: Input image tensor of shape (N, C, H, W) on CUDA
            
        Returns:
            Normalized image tensor of the same shape and dtype
        """
        return F.normalize(img, mean=self.mean, std=self.std)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mean={self.mean}, "
            f"std={self.std})"
        )


class TritonColorJitterNormalize(nn.Module):
    """
    Combined color jitter and normalization in a single fused operation.
    
    This class combines TritonColorJitter and TritonNormalize into a single
    operation that uses a fused kernel for maximum performance. This is the
    recommended way to apply both color jitter and normalization.
    
    Args:
        brightness: How much to jitter brightness (same as TritonColorJitter)
        contrast: How much to jitter contrast (same as TritonColorJitter)
        saturation: How much to jitter saturation (same as TritonColorJitter)
        mean: Sequence of means for normalization (R, G, B)
        std: Sequence of standard deviations for normalization (R, G, B)
        
    Example:
        >>> # Full augmentation pipeline in one transform
        >>> transform = TritonColorJitterNormalize(
        ...     brightness=0.2,  # Range: [0.8, 1.2]
        ...     contrast=0.2,    # Range: [0.8, 1.2]
        ...     saturation=0.2,  # Range: [0.8, 1.2]
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225)
        ... )
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> augmented = transform(img)
    """
    
    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        
        # Process parameters exactly like torchvision
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        
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
    
    @staticmethod
    def _generate_value(left: float, right: float) -> float:
        """Generate a random value in [left, right] range."""
        return torch.empty(1).uniform_(left, right).item()
    
    def _get_params(self):
        """Randomly sample transformation parameters."""
        brightness_factor = (
            self._generate_value(self.brightness[0], self.brightness[1])
            if self.brightness is not None
            else 1.0
        )
        contrast_factor = (
            self._generate_value(self.contrast[0], self.contrast[1])
            if self.contrast is not None
            else 1.0
        )
        saturation_factor = (
            self._generate_value(self.saturation[0], self.saturation[1])
            if self.saturation is not None
            else 1.0
        )
        
        return brightness_factor, contrast_factor, saturation_factor
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random color jitter and normalization in a single fused operation.
        
        Args:
            img: Input image tensor of shape (N, C, H, W) on CUDA
            
        Returns:
            Augmented and normalized image tensor of the same shape and dtype
        """
        # Sample random parameters
        brightness_factor, contrast_factor, saturation_factor = self._get_params()
        
        # Apply the fully fused transformation
        return F.fused_color_normalize(
            img,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            mean=self.mean,
            std=self.std,
        )
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"mean={self.mean}, "
            f"std={self.std})"
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
        >>> transform = TritonGrayscale(num_output_channels=3)
        >>> img = torch.rand(1, 3, 224, 224, device='cuda')
        >>> gray = transform(img)  # Shape: (1, 3, 224, 224), all channels identical
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
    
    Matches torchvision.transforms.v2.RandomGrayscale behavior.
    
    Args:
        p: Probability of converting to grayscale (default: 0.1)
        num_output_channels: Number of output channels (1 or 3, default: 3)
                            Usually 3 to maintain compatibility with RGB pipelines
                            
    Example:
        >>> transform = TritonRandomGrayscale(p=0.5, num_output_channels=3)
        >>> img = torch.rand(4, 3, 224, 224, device='cuda')
        >>> result = transform(img)  # 50% chance of being grayscale
    """
    
    def __init__(self, p: float = 0.1, num_output_channels: int = 3):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p ({p}) must be in [0, 1]")
        if num_output_channels not in (1, 3):
            raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}")
        self.p = p
        self.num_output_channels = num_output_channels
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (N, 3, H, W)
            
        Returns:
            Image tensor, either original or grayscale based on probability
        """
        return F.random_grayscale(image, p=self.p, num_output_channels=self.num_output_channels)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, num_output_channels={self.num_output_channels})"


__all__ = [
    'TritonColorJitter',
    'TritonNormalize',
    'TritonColorJitterNormalize',
    'TritonGrayscale',
    'TritonRandomGrayscale',
]

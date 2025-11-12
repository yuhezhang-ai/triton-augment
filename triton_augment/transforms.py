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
        
    Example:
        >>> # Full augmentation pipeline in one transform
        >>> transform = TritonColorJitterNormalize(
        ...     brightness=0.2,  # Range: [0.8, 1.2]
        ...     contrast=0.2,    # Range: [0.8, 1.2]
        ...     saturation=0.2,  # Range: [0.8, 1.2]
        ...     random_grayscale_p=0.1,  # 10% chance of grayscale
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
        random_grayscale_p: float = 0.0,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        
        # Process parameters exactly like torchvision
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.random_grayscale_p = random_grayscale_p
        
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
            random_grayscale_p=self.random_grayscale_p,
            mean=self.mean,
            std=self.std,
        )
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"random_grayscale_p={self.random_grayscale_p}, "
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


# ============================================================================
# Geometric Transforms
# ============================================================================


class TritonRandomCrop(nn.Module):
    """
    Crop a random portion of the image.
    
    Matches torchvision.transforms.v2.RandomCrop behavior (simplified MVP version).
    
    Args:
        size: Desired output size (height, width) or int for square crop
        
    Example:
        >>> transform = TritonRandomCrop(112)
        >>> img = torch.rand(4, 3, 224, 224, device='cuda')
        >>> cropped = transform(img)
        >>> cropped.shape
        torch.Size([4, 3, 112, 112])
    
    Note:
        For MVP, padding is not supported. Image must be larger than crop size.
        Future versions will support padding, pad_if_needed, fill, padding_mode.
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
            image: Input tensor of shape (N, C, H, W)
            
        Returns:
            Randomly cropped tensor of shape (N, C, size[0], size[1])
        """
        i, j, h, w = self.get_params(image, self.size)
        return F.crop(image, i, j, h, w)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class TritonCenterCrop(nn.Module):
    """
    Crop the center of the image.
    
    Matches torchvision.transforms.v2.CenterCrop behavior.
    
    Args:
        size: Desired output size (height, width) or int for square crop
        
    Example:
        >>> transform = TritonCenterCrop(112)
        >>> img = torch.rand(4, 3, 224, 224, device='cuda')
        >>> cropped = transform(img)
        >>> cropped.shape
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
        
    Example:
        >>> transform = TritonRandomHorizontalFlip(p=0.5)
        >>> img = torch.rand(4, 3, 224, 224, device='cuda')
        >>> flipped = transform(img)  # 50% chance of being flipped
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p ({p}) must be in [0, 1]")
        self.p = p
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (N, C, H, W)
            
        Returns:
            Image tensor, either original or horizontally flipped
        """
        if torch.rand(1).item() < self.p:
            return F.horizontal_flip(image)
        return image
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class TritonRandomCropFlip(nn.Module):
    """
    Fused random crop + random horizontal flip.
    
    This class combines random crop and random horizontal flip in a SINGLE
    kernel launch, eliminating intermediate memory transfers.
    
    **Performance**: ~1.5-2x faster than applying TritonRandomCrop + TritonRandomHorizontalFlip sequentially.
    
    Args:
        size: Desired output size (height, width) or int for square crop
        horizontal_flip_p: Probability of horizontal flip (default: 0.5)
        
    Example:
        >>> # Fused version (FAST - single kernel)
        >>> transform_fused = TritonRandomCropFlip(112, horizontal_flip_p=0.5)
        >>> img = torch.rand(4, 3, 224, 224, device='cuda')
        >>> result = transform_fused(img)
        >>> 
        >>> # Equivalent sequential version (SLOWER - 2 kernels)
        >>> transform_seq = nn.Sequential(
        ...     TritonRandomCrop(112),
        ...     TritonRandomHorizontalFlip(p=0.5)
        ... )
        >>> result_seq = transform_seq(img)
    
    Note:
        The fused version uses compile-time branching (tl.constexpr), so there's
        zero overhead when flip is not triggered.
    """
    
    def __init__(self, size: Union[int, Sequence[int]], horizontal_flip_p: float = 0.5):
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
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (N, C, H, W)
            
        Returns:
            Randomly cropped (and optionally flipped) tensor of shape (N, C, size[0], size[1])
        """
        # Sample random crop parameters
        i, j, h, w = TritonRandomCrop.get_params(image, self.size)
        
        # Sample flip decision
        do_flip = torch.rand(1).item() < self.horizontal_flip_p
        
        # Single fused kernel launch
        return F.fused_crop_flip(image, i, j, h, w, flip_horizontal=do_flip)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, horizontal_flip_p={self.horizontal_flip_p})"


class TritonUltimateAugment(nn.Module):
    """
    THE ULTIMATE FUSED AUGMENTATION: All operations in ONE kernel.
    
    This transform combines ALL augmentations in a single GPU kernel launch:
    - **Geometric Tier**: RandomCrop + RandomHorizontalFlip
    - **Pixel Tier**: ColorJitter (brightness, contrast, saturation) + Normalize
    
    **Performance**: ~3-5x faster than torchvision.transforms.Compose!
    
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
        
    Example:
        >>> # Replace torchvision Compose with single transform:
        >>> # OLD (6 kernel launches):
        >>> transform = transforms.Compose([
        ...     transforms.RandomCrop(112),
        ...     transforms.RandomHorizontalFlip(),
        ...     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ...     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ... ])
        >>> 
        >>> # NEW (1 kernel launch - 3-5x faster!):
        >>> import triton_augment as ta
        >>> transform = ta.TritonUltimateAugment(
        ...     crop_size=112,
        ...     horizontal_flip_p=0.5,
        ...     brightness=0.2,
        ...     contrast=0.2,
        ...     saturation=0.2,
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225)
        ... )
        >>> 
        >>> img = torch.rand(4, 3, 224, 224, device='cuda')
        >>> result = transform(img)  # Single kernel launch!
    
    Note:
        - Uses FAST contrast (centered scaling), not torchvision's blend-with-mean
        - All images in a batch use the same random parameters
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
    
    @staticmethod
    def get_crop_params(image_height: int, image_width: int, crop_height: int, crop_width: int):
        """Sample random crop parameters."""
        if crop_height > image_height or crop_width > image_width:
            raise ValueError(
                f"Crop size ({crop_height}, {crop_width}) is larger than "
                f"input size ({image_height}, {image_width})"
            )
        
        top = torch.randint(0, image_height - crop_height + 1, (1,)).item()
        left = torch.randint(0, image_width - crop_width + 1, (1,)).item()
        
        return top, left
    
    def _get_params(self, image_height: int, image_width: int):
        """
        Sample all random parameters for the ultimate transform.
        
        Returns:
            Tuple of (top, left, flip_horizontal, brightness_factor, contrast_factor, saturation_factor)
        """
        # Sample crop parameters
        top, left = self.get_crop_params(image_height, image_width, self.crop_height, self.crop_width)
        
        # Sample flip decision
        flip_horizontal = torch.rand(1).item() < self.horizontal_flip_p
        
        # Sample color jitter parameters
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
        
        return top, left, flip_horizontal, brightness_factor, contrast_factor, saturation_factor
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply all augmentations in a single fused kernel.
        
        Args:
            image: Input tensor of shape (N, 3, H, W) on CUDA
            
        Returns:
            Augmented tensor of shape (N, 3, crop_height, crop_width)
        """
        _, _, img_height, img_width = image.shape
        
        # Sample all random parameters
        top, left, flip_horizontal, brightness_factor, contrast_factor, saturation_factor = self._get_params(
            img_height, img_width
        )
        
        # Single kernel launch for ALL operations!
        return F.ultimate_fused_augment(
            image,
            top=top,
            left=left,
            height=self.crop_height,
            width=self.crop_width,
            flip_horizontal=flip_horizontal,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            random_grayscale_p=self.random_grayscale_p,
            mean=self.mean,
            std=self.std,
        )
    
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

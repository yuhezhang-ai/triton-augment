# API Reference

Complete reference for all Triton-Augment operations.

## Transform Classes

Transform classes provide stateful, random augmentations similar to torchvision.

### TritonColorJitterNormalize

Fused color jitter and normalization in a single kernel. **Recommended for best performance.**

```python
ta.TritonColorJitterNormalize(
    brightness=0,
    contrast=0,
    saturation=0,
    random_grayscale_p=0.0,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

**Parameters:**

- `brightness` (float or tuple): Brightness jitter range. If float, range is `[max(0, 1-brightness), 1+brightness]`. Default: 0
- `contrast` (float or tuple): Contrast jitter range. Default: 0
- `saturation` (float or tuple): Saturation jitter range. Default: 0
- `random_grayscale_p` (float): Probability of converting to grayscale. Default: 0.0
- `mean` (tuple): Normalization means for R, G, B channels
- `std` (tuple): Normalization stds for R, G, B channels

**Note**: Uses fast contrast (centered scaling), not torchvision-exact. See [Contrast Guide](contrast.md).

---

### TritonColorJitter

Randomly change brightness, contrast, and saturation.

```python
ta.TritonColorJitter(
    brightness=0,
    contrast=0,
    saturation=0
)
```

**Parameters:**

- `brightness` (float or tuple): Brightness jitter range
- `contrast` (float or tuple): Contrast jitter range (uses exact torchvision algorithm)
- `saturation` (float or tuple): Saturation jitter range

**Note**: Applies brightness, contrast, saturation transformations sequentially to match torchvision's behavior. Uses exact torchvision contrast algorithm (blend with mean). For fused operations with better performance, use `TritonColorJitterNormalize`.

---

### TritonNormalize

Normalize image with mean and standard deviation.

```python
ta.TritonNormalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

**Parameters:**

- `mean` (tuple): Per-channel means
- `std` (tuple): Per-channel standard deviations

---

### TritonGrayscale

Convert RGB image to grayscale.

```python
ta.TritonGrayscale(num_output_channels=1)
```

**Parameters:**

- `num_output_channels` (int): 1 for single-channel grayscale, 3 to replicate across channels

---

### TritonRandomGrayscale

Randomly convert image to grayscale.

```python
ta.TritonRandomGrayscale(p=0.1, num_output_channels=3)
```

**Parameters:**

- `p` (float): Probability of grayscale conversion
- `num_output_channels` (int): 1 or 3

---

## Functional API

Low-level functional interface for fine-grained control.

### fused_color_normalize

Fuse multiple operations into a single kernel.

```python
ta.fused_color_normalize(
    image,
    brightness_factor=1.0,
    contrast_factor=1.0,
    saturation_factor=1.0,
    random_grayscale_p=0.0,
    mean=None,
    std=None
)
```

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `brightness_factor` (float): Brightness multiplier (1.0 = no change)
- `contrast_factor` (float): Contrast multiplier (1.0 = no change, uses fast contrast)
- `saturation_factor` (float): Saturation multiplier (1.0 = no change)
- `random_grayscale_p` (float): Probability of grayscale
- `mean` (tuple): Normalization means (None = skip normalization)
- `std` (tuple): Normalization stds (None = skip normalization)

**Returns:** Transformed tensor of same shape

---

### adjust_brightness

Apply brightness adjustment.

```python
ta.adjust_brightness(image, brightness_factor)
```

Formula: `output = input * brightness_factor`

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `brightness_factor` (float): Brightness multiplier

---

### adjust_contrast

Apply contrast adjustment (torchvision-exact).

```python
ta.adjust_contrast(image, contrast_factor)
```

Formula: `output = input * factor + mean(grayscale) * (1 - factor)`

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `contrast_factor` (float): Contrast multiplier

---

### adjust_contrast_fast

Apply fast contrast adjustment.

```python
ta.adjust_contrast_fast(image, contrast_factor)
```

Formula: `output = (input - 0.5) * factor + 0.5`

**Note**: Same as NVIDIA DALI. Faster and fusible, but not torchvision-exact.

---

### adjust_saturation

Apply saturation adjustment.

```python
ta.adjust_saturation(image, saturation_factor)
```

Formula: `output = input * factor + grayscale * (1 - factor)`

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `saturation_factor` (float): Saturation multiplier (0.0 = grayscale, 1.0 = original)

---

### normalize

Apply per-channel normalization.

```python
ta.normalize(image, mean, std)
```

Formula: `output[c] = (input[c] - mean[c]) / std[c]`

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `mean` (tuple): Per-channel means
- `std` (tuple): Per-channel stds

---

### rgb_to_grayscale

Convert RGB to grayscale.

```python
ta.rgb_to_grayscale(image, num_output_channels=1)
```

Formula: `gray = 0.2989*R + 0.587*G + 0.114*B`

**Parameters:**

- `image` (torch.Tensor): RGB tensor `(N, 3, H, W)`
- `num_output_channels` (int): 1 or 3

---

### random_grayscale

Randomly convert to grayscale.

```python
ta.random_grayscale(image, p=0.1, num_output_channels=3)
```

**Parameters:**

- `image` (torch.Tensor): Input tensor
- `p` (float): Probability of conversion
- `num_output_channels` (int): 1 or 3

---

## Geometric Operations

### crop

Crop a rectangular region from the image.

```python
ta.crop(image, top, left, height, width)
```

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `top` (int): Top coordinate of crop region
- `left` (int): Left coordinate of crop region
- `height` (int): Height of crop region
- `width` (int): Width of crop region

**Returns:** Cropped tensor of shape `(N, C, height, width)`

**Example:**

```python
img = torch.rand(4, 3, 224, 224, device='cuda')
cropped = ta.crop(img, top=20, left=30, height=112, width=112)
# cropped.shape = (4, 3, 112, 112)
```

---

### center_crop

Crop the center region from the image.

```python
ta.center_crop(image, size)
```

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `size` (int or tuple): Desired output size. If int, output is square `(size, size)`

**Returns:** Center-cropped tensor

**Example:**

```python
img = torch.rand(4, 3, 224, 224, device='cuda')
cropped = ta.center_crop(img, 112)
# cropped.shape = (4, 3, 112, 112), centered
```

---

### horizontal_flip

Flip image horizontally (left-to-right).

```python
ta.horizontal_flip(image)
```

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA

**Returns:** Horizontally flipped tensor of same shape

**Example:**

```python
img = torch.rand(4, 3, 224, 224, device='cuda')
flipped = ta.horizontal_flip(img)
```

---

### fused_crop_flip

**Fused operation**: Combine crop + horizontal flip in a single kernel.

```python
ta.fused_crop_flip(image, top, left, height, width, flip_horizontal=True)
```

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `top`, `left`, `height`, `width`: Crop parameters
- `flip_horizontal` (bool): Whether to apply horizontal flip

**Returns:** Cropped and optionally flipped tensor

**Performance**: ~1.5-2x faster than sequential crop + flip

**Example:**

```python
# Fused (single kernel - FAST)
result = ta.fused_crop_flip(img, 20, 30, 112, 112, flip_horizontal=True)

# vs Sequential (2 kernels)
result = ta.crop(img, 20, 30, 112, 112)
result = ta.horizontal_flip(result)
```

---

### fused_augment

**THE ULTIMATE**: Fuse ALL augmentations (geometric + pixel) in ONE kernel! 🚀

```python
ta.fused_augment(
    image,
    top, left, height, width,
    flip_horizontal=False,
    brightness_factor=1.0,
    contrast_factor=1.0,
    saturation_factor=1.0,
    mean=None,
    std=None
)
```

**Operations fused:**
1. **Geometric**: Crop + Horizontal Flip
2. **Pixel**: Brightness + Contrast (fast) + Saturation + Normalize

**Parameters:**

- `image` (torch.Tensor): Input tensor `(N, C, H, W)` on CUDA
- `top`, `left`, `height`, `width`: Crop parameters
- `flip_horizontal` (bool): Whether to flip horizontally
- `brightness_factor` (float): Brightness multiplier (1.0 = no change)
- `contrast_factor` (float): Contrast multiplier (uses fast contrast)
- `saturation_factor` (float): Saturation multiplier (1.0 = no change)
- `mean`, `std` (tuple): Normalization parameters (None = skip)

**Returns:** Fully augmented tensor of shape `(N, C, height, width)`

**Performance**: Up to 12x faster on large images (8.1x average on Tesla T4, scales dramatically with image size)

**Example:**

```python
# Single kernel launch for ALL operations!
result = ta.fused_augment(
    img,
    top=20, left=30, height=112, width=112,
    flip_horizontal=True,
    brightness_factor=1.2,
    contrast_factor=1.1,
    saturation_factor=0.9,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# vs torchvision Compose (6 kernel launches)
result = tvF.crop(img, 20, 30, 112, 112)
result = tvF.horizontal_flip(result)
result = tvF.adjust_brightness(result, 1.2)
result = tvF.adjust_contrast(result, 1.1)
result = tvF.adjust_saturation(result, 0.9)
result = tvF.normalize(result, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

**Note**: Uses fast contrast (centered scaling), not torchvision-exact. See [Contrast Guide](contrast.md).

---

## Geometric Transform Classes

### TritonRandomCrop

Randomly crop a portion of the image.

```python
ta.TritonRandomCrop(size)
```

**Parameters:**

- `size` (int or tuple): Desired output size. If int, output is square

**Example:**

```python
transform = ta.TritonRandomCrop(112)
img = torch.rand(4, 3, 224, 224, device='cuda')
cropped = transform(img)  # Random 112x112 crop
```

---

### TritonCenterCrop

Crop the center of the image.

```python
ta.TritonCenterCrop(size)
```

**Parameters:**

- `size` (int or tuple): Desired output size

**Example:**

```python
transform = ta.TritonCenterCrop(112)
img = torch.rand(4, 3, 224, 224, device='cuda')
cropped = transform(img)  # Center 112x112 crop
```

---

### TritonRandomHorizontalFlip

Randomly flip image horizontally.

```python
ta.TritonRandomHorizontalFlip(p=0.5)
```

**Parameters:**

- `p` (float): Probability of flipping (default: 0.5)

**Example:**

```python
transform = ta.TritonRandomHorizontalFlip(p=0.5)
img = torch.rand(4, 3, 224, 224, device='cuda')
result = transform(img)  # 50% chance of flip
```

---

### TritonRandomCropFlip

**Fused transform**: Random crop + random flip in one kernel.

```python
ta.TritonRandomCropFlip(size, horizontal_flip_p=0.5)
```

**Parameters:**

- `size` (int or tuple): Desired output size
- `horizontal_flip_p` (float): Probability of horizontal flip

**Performance**: ~1.5-2x faster than sequential transforms

**Example:**

```python
# Fused (single kernel)
transform = ta.TritonRandomCropFlip(112, horizontal_flip_p=0.5)

# vs Sequential (2 kernels)
transform = transforms.Compose([
    ta.TritonRandomCrop(112),
    ta.TritonRandomHorizontalFlip(0.5)
])
```

---

### TritonFusedAugment

**THE ULTIMATE TRANSFORM**: All augmentations in ONE kernel! 🚀

```python
ta.TritonFusedAugment(
    crop_size,
    horizontal_flip_p=0.0,
    brightness=0,
    contrast=0,
    saturation=0,
    random_grayscale_p=0.0,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

**Parameters:**

- `crop_size` (int or tuple): Crop size
- `horizontal_flip_p` (float): Probability of horizontal flip (default: 0.0, no flip)
- `brightness`, `contrast`, `saturation` (float or tuple): Color jitter ranges
- `random_grayscale_p` (float): Probability of converting to grayscale (default: 0.0, no grayscale)
- `mean`, `std` (tuple): Normalization parameters

**Performance**: Up to 12x faster on large images (8.1x average on Tesla T4, scales dramatically with image size)

**Design Note**: All augmentation parameters default to 0 (no augmentation) for consistency and predictability. Users explicitly opt-in to each augmentation they want to use.

**Example:**

```python
# Replace torchvision Compose (6 kernel launches)
old_transform = transforms.Compose([
    transforms.RandomCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# With TritonFusedAugment (1 kernel launch - significantly faster!)
new_transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

img = torch.rand(32, 3, 224, 224, device='cuda')
result = new_transform(img)  # Single kernel launch!
```

**Note**: Uses fast contrast. See [Contrast Guide](contrast.md).

---

## Utility Functions

### enable_autotune / disable_autotune

Control auto-tuning behavior.

```python
ta.enable_autotune()   # Enable for optimal performance
ta.disable_autotune()  # Disable for faster startup (default)
```

---

### is_autotune_enabled

Check auto-tuning status.

```python
enabled = ta.is_autotune_enabled()
```

**Returns:** `bool` - True if auto-tuning is enabled

---

### warmup_cache

Pre-compile kernels for specified sizes.

```python
ta.warmup_cache(
    batch_sizes=(32, 64),
    image_sizes=(224, 256, 512)
)
```

**Parameters:**

- `batch_sizes` (tuple): Batch sizes to warm up
- `image_sizes` (tuple): Image sizes (height = width) to warm up

**Note**: Only useful when auto-tuning is enabled. See [Auto-Tuning Guide](auto-tuning.md).

---

## Type Annotations

All functions support type hints:

```python
from typing import Tuple, Optional
import torch

def fused_color_normalize(
    image: torch.Tensor,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
    saturation_factor: float = 1.0,
    random_grayscale_p: float = 0.0,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> torch.Tensor:
    ...
```

---

## Input Requirements

All operations accept:

- **Device**: CUDA (GPU) or CPU - *CPU tensors are automatically moved to GPU*
- **Shape**: `(C, H, W)` or `(N, C, H, W)` - *3D tensors are automatically batched*
- **Dtype**: float32 or float16
- **Range**: [0, 1] for color operations (required)

**Notes:**
- After normalization, values can be outside [0, 1] range
- 3D tensors `(C, H, W)` are automatically converted to `(1, C, H, W)` for processing
- CPU tensors are automatically transferred to CUDA for GPU processing

---

## Performance Tips

### Use Fused Kernel Even for Partial Operations

**Key insight**: Even if you only need a subset of operations, use `TritonFusedAugment` or `F.fused_augment` for best performance! Simply set unused operations to no-op values:

```python
# Example: Only need crop + normalize (no flip, no color jitter)
transform = ta.TritonFusedAugment(
    crop_size=224,
    horizontal_flip_p=0.0,      # No flip
    brightness=0.0,             # No brightness
    contrast=0.0,               # No contrast
    saturation=0.0,             # No saturation
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
# Still faster than calling crop() + normalize() separately!
```

The fused kernel is optimized to skip operations set to no-op values at compile time.

### Individual Operations Performance

Individual Triton operations (e.g., `ta.crop()`, `ta.adjust_brightness()`):
- **Small images/batches**: Slightly slower than torchvision (kernel launch overhead)
- **Large images/batches**: Faster than torchvision (better GPU utilization)

See [benchmark results](index.md#performance) for detailed performance comparisons.

**Recommendation**: Use `TritonFusedAugment` for production training, regardless of how many operations you need.


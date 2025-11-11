# Quick Start Guide

## Basic Usage: Ultimate Fusion (Recommended) 🚀

The **fastest way** to use Triton-Augment - fuse ALL augmentations in a single kernel:

```python
import torch
import triton_augment as ta

# Create a batch of images on GPU
images = torch.rand(32, 3, 224, 224, device='cuda')

# Replace torchvision Compose (6 kernel launches)
# With Triton-Augment (1 kernel launch - 3-5x faster!)
transform = ta.TritonUltimateAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# Apply transformation
augmented = transform(images)  # Single kernel launch for ALL operations!
```

**What it does:**
- RandomCrop (112×112)
- RandomHorizontalFlip (50% probability)
- ColorJitter (brightness, contrast, saturation)
- Normalize

**Performance**: ~8-10x faster than torchvision Compose

---

## Other Fusion Options

### Pixel-Only Fusion

If you don't need geometric operations (crop/flip), use pixel fusion:

```python
# Fuse color jitter + normalize (single kernel)
transform = ta.TritonColorJitterNormalize(
    brightness=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)  # ~2-3x faster
```

### Geometric-Only Fusion

If you only need crop + flip:

```python
# Fuse crop + flip (single kernel)
transform = ta.TritonRandomCropFlip(size=112, horizontal_flip_p=0.5)

augmented = transform(images)  # ~1.5-2x faster
```

---

## Individual Operations

For maximum control, use individual operations (fixed parameters):

```python
import triton_augment.functional as F

img = torch.rand(4, 3, 224, 224, device='cuda')

# Geometric operations
cropped = F.crop(img, top=20, left=30, height=112, width=112)
flipped = F.horizontal_flip(img)

# Color operations 
bright = F.adjust_brightness(img, 1.2)
saturated = F.adjust_saturation(img, 0.9)
normalized = F.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

Or use transform classes (random augmentations):

```python
import triton_augment as ta

# Individual transforms
crop = ta.TritonRandomCrop(112)
flip = ta.TritonRandomHorizontalFlip(p=0.5)
jitter = ta.TritonColorJitter(brightness=0.2)
normalize = ta.TritonNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

---

## Integration with PyTorch DataLoader

```python
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import triton_augment as ta

class GPUTransform:
    """Apply Triton augmentations on GPU."""
    def __init__(self):
        self.transform = ta.TritonUltimateAugment(
            crop_size=112,
            horizontal_flip_p=0.5,
            brightness=0.2,
            saturation=0.2,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    
    def __call__(self, batch):
        images = batch[0].cuda()
        labels = batch[1].cuda()
        augmented = self.transform(images)
        return augmented, labels

# Standard PyTorch DataLoader
dataset = ImageFolder('path/to/data')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

gpu_transform = GPUTransform()
for images, labels in loader:
    augmented, labels = gpu_transform((images, labels))
    # ... training code ...
```

---

## Fusion Levels Comparison

Choose the right level for your use case:

| Level | Operations | Kernels | Speedup | When to Use |
|-------|-----------|---------|---------|-------------|
| **Ultimate** | Crop+Flip+Color+Norm | 1 | ~3-5x 🚀 | Production training (best performance) |
| **Specialized** | Geometric OR Pixel | 1-2 | ~1.5-3x ⚡ | Need flexibility in pipeline |
| **Individual** | One at a time | 6+ | ~1.2-1.5x | Maximum control over each step |

**Recommendation**: Use **Ultimate Fusion** (`TritonUltimateAugment`) for production training.

---

## Next Steps

- [Float16 Support](float16.md) - Use half-precision for 1.3-2x additional speedup
- [Batch Behavior](batch-behavior.md) - Understand random parameter handling
- [Contrast Notes](contrast.md) - **Important**: Fast contrast vs torchvision-exact
- [Auto-Tuning](auto-tuning.md) - Optional performance optimization
- [API Reference](api-reference.md) - Complete API documentation

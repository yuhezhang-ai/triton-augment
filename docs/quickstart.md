# Quick Start Guide

## Basic Usage: Ultimate Fusion (Recommended) 🚀

The **fastest way** to use Triton-Augment - fuse ALL augmentations in a single kernel:

```python
import torch
import triton_augment as ta

# Create a batch of images on GPU
images = torch.rand(32, 3, 224, 224, device='cuda')

# Replace torchvision Compose (7 kernel launches)
# With Triton-Augment (1 kernel launch - significantly faster!)
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    grayscale_p=0.1,
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

**Performance**: Up to 12x faster on large images (8.1x average on Tesla T4, scales dramatically with image size)

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

augmented = transform(images)  # Faster, single fused kernel
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

## Training Integration

**Recommended Pattern**: Load data on CPU (fast async I/O), augment on GPU (fast batch processing)

```python
import torch
import triton_augment as ta
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: CPU data loading with workers
train_dataset = datasets.CIFAR10(
    './data', train=True,
    transform=transforms.ToTensor()  # Only ToTensor on CPU
)
train_loader = DataLoader(
    train_dataset, batch_size=128,
    num_workers=4, pin_memory=True  # Fast async loading!
)

# Step 2: GPU augmentation transform
augment = ta.TritonFusedAugment(
    crop_size=28, horizontal_flip_p=0.5,
    brightness=0.2, saturation=0.2,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616)
)

# Step 3: Apply in training loop
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    images = augment(images)  # All ops in 1 kernel! 🚀
    # ... rest of training ...
```

**Why This Pattern:**

- ✅ **Fast async data loading**: `num_workers > 0` for CPU parallelism
- ✅ **Fast GPU batch processing**: All augmentations in 1 fused kernel
- ✅ **Different parameters per sample**: Each image gets different random parameters (default)
- ✅ **Best of both worlds**: CPU for I/O, GPU for compute
- ✅ **Kernel fusion**: No intermediate memory allocations
- ✅ **Large batch advantage**: Speedup increases with batch size

**Note**: Set `same_on_batch=True` if you want all images to share the same random parameters.

💡 **Pro Tip**: Apply Triton-Augment transforms AFTER moving tensors to GPU for maximum performance!

**Full Examples**: See [`examples/train_mnist.py`](https://github.com/yuhezhang-ai/triton-augment/blob/main/examples/train_mnist.py) and [`examples/train_cifar10.py`](https://github.com/yuhezhang-ai/triton-augment/blob/main/examples/train_cifar10.py) for complete training scripts with neural networks.

---

## Next Steps

- [Batch Behavior](batch-behavior.md) - Understand random parameter handling
- [Contrast Notes](contrast.md) - Fast contrast vs torchvision-exact
- [Auto-Tuning](auto-tuning.md) - Optional performance optimization
- [Float16 Support](float16.md) - Use half-precision for 1.3-2x additional speedup
- [API Reference](api-reference.md) - Complete API documentation

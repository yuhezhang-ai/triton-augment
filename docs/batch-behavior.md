# Batch Behavior & Different Parameters Per Sample

!!! success "Different Parameters Per Sample by Default"
    Triton-Augment applies **different random parameters to each image in a batch** by default!

## Default Behavior: Different Parameters Per Sample

```python
import torch
import triton_augment as ta

# Each image gets DIFFERENT random augmentation
batch = torch.rand(32, 3, 224, 224, device='cuda')

transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    saturation=0.2
)

result = transform(batch)  # 32 different random augmentations! ✅
```

**How it works:**
- Random parameters are sampled **per-image** (32 different crop positions, flip decisions, color factors)
- All processed in **ONE kernel launch** on GPU
- Fast batch processing + individual randomness = best of both worlds! 🚀

---

## Controlling Randomness: `same_on_batch` Flag

All transform classes support the `same_on_batch` parameter:

### Different Parameters Per Sample (Default)

```python
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    same_on_batch=False  # Default
)

batch = torch.rand(32, 3, 224, 224, device='cuda')
result = transform(batch)  # Each image: different crop, flip, brightness
```

### Batch-Wide Parameters (Same for All)

```python
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    same_on_batch=True  # Same params for all images
)

batch = torch.rand(32, 3, 224, 224, device='cuda')
result = transform(batch)  # All images: same crop position, flip, brightness
```

---

## When to Use Each Mode

### Different Parameters Per Sample (Recommended for Training)

✅ **Use when:**
- Training neural networks (standard augmentation)
- You want maximum data diversity
- Each image should be augmented independently

```python
# Standard training setup
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    same_on_batch=False  # ✅ Default, each image different
)

for images, labels in train_loader:
    images = images.cuda()
    images = transform(images)  # Unique augmentation per image
    # ... training ...
```

**Performance:** Still fast! One kernel launch processes entire batch with per-image params.

### Batch-Wide Parameters (Specialized Use Cases)

✅ **Use when:**
- Processing video frames (consistent augmentation across frames)
- Debugging (easier to see effect of specific parameters)
- Specific research requirements

```python
# Video frame processing
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    same_on_batch=True  # Same for all frames
)

video_frames = torch.rand(8, 3, 224, 224, device='cuda')  # 8 frames
result = transform(video_frames)  # Consistent augmentation across frames ✅
```

---

## All Transforms Support Different Parameters Per Sample

The following transforms all support `same_on_batch`:

- `TritonFusedAugment` - Complete pipeline (crop, flip, color, normalize)
- `TritonRandomCropFlip` - Geometric operations only
- `TritonColorJitterNormalize` - ColorJitter + Normalize
- `TritonColorJitter` - ColorJitter only
- `TritonRandomCrop` - Random cropping
- `TritonRandomHorizontalFlip` - Random flipping
- `TritonRandomGrayscale` - Random grayscale conversion

Example:
```python
# Individual transforms also support same_on_batch
crop = ta.TritonRandomCrop(112, same_on_batch=False)
flip = ta.TritonRandomHorizontalFlip(p=0.5, same_on_batch=False)
jitter = ta.TritonColorJitter(brightness=0.2, same_on_batch=False)
```

---

## Functional API: Fixed Parameters

The functional API (`triton_augment.functional`) is for **deterministic augmentations**:

```python
import triton_augment.functional as F

batch = torch.rand(32, 3, 224, 224, device='cuda')

# Fixed parameters - same for all images
result = F.fused_augment(
    batch,
    top=20, left=30,          # Fixed crop position
    height=112, width=112,
    flip_horizontal=True,      # Fixed flip decision
    brightness_factor=1.2,     # Fixed brightness
    saturation_factor=0.9,     # Fixed saturation
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

If you need per-image fixed parameters, pass tensors:

```python
# Per-image fixed parameters (different but deterministic)
top_offsets = torch.tensor([10, 20, 30, ...], device='cuda')  # [32]
brightness = torch.tensor([1.1, 1.2, 1.3, ...], device='cuda')  # [32]

result = F.fused_augment(
    batch,
    top=top_offsets,           # Tensor: per-image positions
    left=30,                   # Scalar: same for all
    height=112, width=112,
    flip_horizontal=True,
    brightness_factor=brightness,  # Tensor: per-image factors
    saturation_factor=0.9,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

---

## Comparison with torchvision

**torchvision** doesn't support different parameters per sample on batched tensors:

```python
import torchvision.transforms.v2 as tv_transforms

batch = torch.rand(32, 3, 224, 224, device='cuda')
transform = tv_transforms.ColorJitter(brightness=0.2)

# All 32 images get the SAME brightness factor
result = transform(batch)
```

To get different parameters per sample in torchvision, you must apply transforms before batching (in DataLoader), which processes images sequentially.

**Triton-Augment advantage:** Different parameters per sample with GPU batch processing - best of both worlds! 🚀

---

## Performance Impact

**Different Parameters Per Sample:**
- ✅ Same kernel launch time as batch-wide
- ✅ One kernel launch for entire batch
- ✅ Minimal overhead (kernel uses `tl.load` to fetch per-image params)

**Batch-Wide Parameters:**
- ✅ Slightly faster (no per-image parameter indexing)
- ⚠️ Less data diversity for training

**Verdict:** Use different parameters per sample (default) for training. The performance difference is negligible (~1-2%), but data diversity is crucial!

---

## Example: Real Training Pipeline

```python
import torch
import triton_augment as ta
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Load data on CPU with workers (fast async I/O)
train_dataset = datasets.CIFAR10(
    './data', train=True,
    transform=transforms.ToTensor()  # Only ToTensor on CPU
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    num_workers=4,      # ✅ Async data loading
    pin_memory=True
)

# Step 2: GPU augmentation with different parameters per sample
augment = ta.TritonFusedAugment(
    crop_size=28,
    horizontal_flip_p=0.5,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
    same_on_batch=False  # ✅ Each image gets unique augmentation
)

# Step 3: Training loop
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    images = augment(images)  # 🚀 One kernel, all augmentations, per-image random!
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    # ... backprop ...
```

**Result:**
- ✅ Fast async CPU data loading (`num_workers > 0`)
- ✅ Fast GPU batch processing (one kernel)
- ✅ Different parameters per sample (maximum diversity)
- ✅ Best of all worlds! 🚀

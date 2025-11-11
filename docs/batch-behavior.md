# Batch Behavior & Random Augmentation

!!! warning "Important"
    Random augmentations apply the **same parameters to all images in a batch**.

## Default Behavior

```python
import torch
import triton_augment as ta

# This applies SAME random augmentation to all 32 images
batch = torch.rand(32, 3, 224, 224, device='cuda')

transform = ta.TritonColorJitterNormalize(brightness=0.2)
result = transform(batch)  # Same brightness factor for all 32 images
```

This matches torchvision's behavior when calling transforms on batched tensors.

## For Different Augmentations Per Image

### Standard Practice (Recommended)

Use PyTorch's `DataLoader` which applies transforms **before batching**:

```python
from torch.utils.data import Dataset, DataLoader
import triton_augment as ta

class MyDataset(Dataset):
    def __init__(self):
        self.transform = ta.TritonColorJitterNormalize(brightness=0.2)
    
    def __getitem__(self, idx):
        img = load_image(idx)  # Load single image
        img = self.transform(img.unsqueeze(0).cuda())  # Apply transform
        return img.squeeze(0)

# DataLoader batches AFTER transforms
# Each image gets different random augmentations ✅
loader = DataLoader(dataset, batch_size=32)
```

### Alternative: Loop Over Batch

If you have pre-batched tensors:

```python
import torch
import triton_augment as ta

batch = torch.rand(32, 3, 224, 224, device='cuda')
transform = ta.TritonColorJitterNormalize(brightness=0.2)

# Apply different random parameters to each image
results = torch.stack([
    transform(img.unsqueeze(0)).squeeze(0)
    for img in batch
])
```

!!! note
    This is less efficient (32 kernel launches instead of 1) but provides per-image randomness.

## When Same-Batch Parameters Are Fine

Many use cases don't need per-image randomness:

### 1. Pre-batched datasets

```python
# Batch created during data loading
# All images go through the same augmentation pipeline anyway
batch = load_batch_from_disk()
augmented = transform(batch)  # Same params OK
```

### 2. Inference/testing

```python
# No randomness needed during evaluation
transform = ta.TritonNormalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
batch = load_test_batch()
normalized = transform(batch)  # Deterministic
```

### 3. Video frames

```python
# Frames from the same video should have consistent augmentation
video_batch = torch.rand(8, 3, 224, 224, device='cuda')  # 8 frames
augmented = transform(video_batch)  # Same params across frames ✅
```

## Future: Per-Image Randomness

If you need per-image random parameters on pre-batched tensors (like Kornia's `same_on_batch=False`), please open an issue!

Implementation is straightforward - just need to:
1. Sample per-image parameters
2. Pass them to the kernel
3. Index by batch dimension

Example API (not yet implemented):

```python
# Future API
transform = ta.TritonColorJitterNormalize(
    brightness=0.2,
    same_on_batch=False  # ← Not yet implemented
)
```

## Comparison with Other Libraries

| Library | Default Behavior | Per-Image Option |
|---------|------------------|------------------|
| **torchvision** | Same on batch (when batched) | Use DataLoader |
| **Kornia** | Same on batch | `same_on_batch=False` |
| **Albumentations** | Per-image (CPU-based) | N/A |
| **NVIDIA DALI** | Per-image (pipeline-based) | Default |
| **Triton-Augment** | Same on batch | Use DataLoader |

## Best Practices

✅ **DO**: Use DataLoader for per-image randomness (standard PyTorch workflow)

```python
dataset = MyDataset(transforms=ta.TritonColorJitterNormalize(...))
loader = DataLoader(dataset, batch_size=32)
```

✅ **DO**: Use same-batch params for video frames or pre-batched data

```python
video_batch = load_video_frames()
augmented = transform(video_batch)
```

❌ **DON'T**: Loop over batch dimension if you don't need per-image randomness

```python
# Inefficient if you don't need per-image randomness
results = [transform(img.unsqueeze(0)) for img in batch]
```

❌ **DON'T**: Expect different params when calling transform on batched tensor

```python
batch = torch.rand(32, 3, 224, 224, device='cuda')
transform(batch)  # Same params for all 32 images
```


# Float16 (Half Precision) Support

All Triton-Augment operations fully support float16, which can provide additional speedup and reduce memory usage.

## Basic Usage

```python
import torch
import triton_augment as ta

# Create float16 images
images = torch.rand(32, 3, 224, 224, device='cuda', dtype=torch.float16)

# Apply transforms (works seamlessly with float16)
transform = ta.TritonColorJitterNormalize(
    brightness=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)  # Output is also float16
```

## Benefits

🚀 **Additional speedup**: Float16 operations are faster on modern GPUs (Turing, Ampere, Ada architectures)

💾 **Half the memory**: Use 2x less VRAM, enabling:
- Larger batch sizes
- Higher resolution images
- More models in memory

✅ **Maintained accuracy**: Augmentations are robust to lower precision (~3-4 decimal digits)

## Performance Comparison

Expected speedup with float16 vs float32:

| GPU Architecture | Speedup | Notes |
|-----------------|---------|-------|
| **RTX 40xx (Ada)** | 1.5-2x | Best float16 support |
| **RTX 30xx (Ampere)** | 1.3-1.8x | Good float16 support |
| **RTX 20xx (Turing)** | 1.2-1.5x | First with tensor cores |
| **GTX 10xx (Pascal)** | 1.0-1.1x | Limited float16 benefit |
| **T4 (Turing)** | 1.3-1.6x | Data center GPU |
| **A100 (Ampere)** | 1.5-2x | Optimized for float16 |

Run `examples/benchmark_triton.py` to measure on your hardware.

## When to Use Float16

### ✅ Recommended For

**Training with mixed precision**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
transform = ta.TritonColorJitterNormalize(...)

for images, labels in loader:
    with autocast():  # Automatic float16 conversion
        images = images.cuda()
        augmented = transform(images)
        output = model(augmented)
        loss = criterion(output, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Large batch sizes**:
```python
# Use float16 to fit larger batches in memory
images = images.half()  # Convert to float16
transform = ta.TritonColorJitterNormalize(...)
augmented = transform(images)  # 2x memory savings
```

**Inference pipelines**:
```python
# Faster inference with float16
model.half()  # Convert model to float16
transform = ta.TritonColorJitterNormalize(...)

@torch.no_grad()
def inference(image):
    image = image.half().cuda()
    augmented = transform(image)
    return model(augmented)
```

### ❌ Not Needed For

- CPU-only workflows (float16 is GPU-specific)
- Small batch sizes (< 16) where memory isn't constrained
- When exact float32 precision is required (rare for augmentation)

## Precision Considerations

Float16 results will differ slightly from float32:

```python
img = torch.rand(1, 3, 224, 224, device='cuda')

# Float32
result_fp32 = ta.fused_color_normalize(img, brightness_factor=1.2, ...)

# Float16
result_fp16 = ta.fused_color_normalize(img.half(), brightness_factor=1.2, ...)

# Difference is small (~0.1% relative error)
diff = (result_fp32 - result_fp16.float()).abs().mean()
print(f"Mean absolute difference: {diff:.6f}")  # Typically < 0.001
```

This is **expected and acceptable** for data augmentation:
- Models are robust to small input perturbations
- Augmentation inherently introduces variation
- Training dynamics smooth out precision differences

## Best Practices

✅ **DO**: Use float16 for training with `torch.cuda.amp`

```python
with autocast():
    augmented = transform(images)
```

✅ **DO**: Convert to float16 early in pipeline

```python
images = images.half().cuda()  # Convert once
augmented = transform(images)  # Stay in float16
```

✅ **DO**: Use float16 for memory-constrained scenarios

```python
# Fit 2x larger batch
batch_size = 64  # Instead of 32 with float32
images = images.half()
```

❌ **DON'T**: Mix float16 and float32 unnecessarily

```python
# Inefficient: type conversions are slow
images_fp16 = images.half()
result = transform(images_fp16.float())  # ← Unnecessary conversion
```

❌ **DON'T**: Worry about precision differences

```python
# This is fine - augmentation doesn't need perfect precision
result_fp16 = transform(images.half())
# Small differences from fp32 won't impact training
```

## Benchmarking

Compare float16 vs float32 performance:

```python
import torch
import triton_augment as ta
from triton.testing import do_bench

batch = 32
img_fp32 = torch.rand(batch, 3, 224, 224, device='cuda', dtype=torch.float32)
img_fp16 = img_fp32.half()

transform = ta.TritonColorJitterNormalize(brightness=0.2, saturation=0.2)

# Benchmark
time_fp32 = do_bench(lambda: transform(img_fp32))
time_fp16 = do_bench(lambda: transform(img_fp16))

print(f"Float32: {time_fp32:.3f} ms")
print(f"Float16: {time_fp16:.3f} ms")
print(f"Speedup: {time_fp32/time_fp16:.2f}x")
```

## Technical Details

### Why is Float16 Faster?

1. **Tensor cores**: Modern GPUs have specialized hardware for float16 operations
2. **Memory bandwidth**: Float16 requires half the memory bandwidth
3. **Cache efficiency**: More data fits in GPU caches

### Precision Characteristics

Float16 (IEEE 754 half precision):
- **Range**: ±65,504 (sufficient for image data in [0, 1] or [-3, 3] normalized)
- **Precision**: ~3-4 decimal digits (vs ~7-8 for float32)
- **Special values**: Supports NaN, Inf (rare in augmentation)

For image augmentation:
- ✅ Brightness, saturation: Very robust to float16
- ✅ Normalization: Works well with float16
- ✅ Contrast: Minor differences, acceptable for training


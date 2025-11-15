# Float16 (Half Precision) Support

Triton-Augment fully supports float16, providing memory savings and potential speedup on modern GPUs.

## Basic Usage

```python
import torch
import triton_augment as ta

# Create float16 images
images = torch.rand(32, 3, 224, 224, device='cuda', dtype=torch.float16)

# Apply fused transform (works seamlessly with float16)
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)  # Output is also float16
```

## Benefits

💾 **Half the memory**: Float16 uses 2x less VRAM, enabling:
- Larger batch sizes
- Higher resolution images
- More models in memory

⚡ **Potential speedup**: On Tesla T4, we observed ~1.3-1.4x speedup for large images (1024×1024+)

✅ **Maintained accuracy**: Data augmentation is robust to lower precision

## Benchmark Results (Tesla T4)

Our measurements on **Ultimate Fusion** (all operations in one kernel):

| Image Size | Float32 | Float16 | Speedup |
|------------|---------|---------|---------|
| 256×256    | 0.41 ms | 0.47 ms | **0.88x** (slower) |
| 512×512    | 0.48 ms | 0.47 ms | **1.03x** |
| 640×640    | 0.57 ms | 0.49 ms | **1.15x** |
| 1024×1024  | 0.93 ms | 0.72 ms | **1.29x** |
| 1280×1280  | 1.27 ms | 0.93 ms | **1.36x** |

**Conclusion**: Float16 provides meaningful speedup for **large images** (600×600+), but offers minimal benefit for small images.

💡 **Your mileage may vary**: Run `examples/benchmark_triton.py` to measure on your GPU.

## When to Use Float16

### ✅ Use Float16 When:

- **Training with mixed precision** (`torch.cuda.amp`)
- **Memory constrained**: Need to fit larger batches or higher resolution images
- **Large images**: 600×600+ where float16 shows speedup (based on T4 benchmarks)

### ❌ Skip Float16 When:

- **Small images**: < 512×512 (minimal or negative speedup on T4)
- **CPU-only training**: Float16 is GPU-specific
- **Debugging**: Float32 is easier to inspect

## Precision Considerations

Float16 results will differ slightly from float32 due to reduced precision. This is **expected and acceptable** for data augmentation:

- Models are robust to small input perturbations
- Augmentation inherently introduces variation
- Training with mixed precision is a standard practice

## Usage Example

**With mixed precision training**:

```python
from torch.cuda.amp import autocast, GradScaler
import triton_augment as ta

transform = ta.TritonFusedAugment(...)
scaler = GradScaler()

for images, labels in loader:
    with autocast():  # Images automatically converted to float16
        images = images.cuda()
        augmented = transform(images)
        output = model(augmented)
        loss = criterion(output, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Manual float16 conversion**:

```python
# Convert to float16 for memory savings
images = images.half().cuda()
augmented = transform(images)  # Stays in float16
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

transform = ta.TritonFusedAugment(
    crop_size=112, brightness=0.2, saturation=0.2
)

# Benchmark
time_fp32 = do_bench(lambda: transform(img_fp32))
time_fp16 = do_bench(lambda: transform(img_fp16))

print(f"Float32: {time_fp32:.3f} ms")
print(f"Float16: {time_fp16:.3f} ms")
print(f"Speedup: {time_fp32/time_fp16:.2f}x")
```

## Why Float16 Can Be Faster

Float16 benefits come from:
1. **Memory bandwidth**: Half the data to transfer (2 bytes vs 4 bytes per value)
2. **Cache efficiency**: More data fits in GPU caches
3. **GPU hardware**: Modern GPUs have specialized float16 units

**Note**: Speedup varies by GPU architecture and operation complexity. Always benchmark on your specific hardware.

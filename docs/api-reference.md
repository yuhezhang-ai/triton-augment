# API Reference

Complete auto-generated API reference for all Triton-Augment operations.

---

## Transform Classes

Transform classes provide stateful, random augmentations similar to torchvision. Recommended for training pipelines.


::: triton_augment.TritonFusedAugment
    options:
      show_root_heading: true
      show_source: false
      members_order: source

---

::: triton_augment.TritonColorJitterNormalize
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.TritonRandomCropFlip
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.TritonColorJitter
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.TritonNormalize
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.TritonRandomGrayscale
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.TritonGrayscale
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.TritonRandomCrop
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.TritonCenterCrop
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.TritonRandomHorizontalFlip
    options:
      show_root_heading: true
      show_source: false

---

## Functional API

Low-level functional interface for fine-grained control with fixed parameters. Use when you need deterministic operations.

::: triton_augment.functional.fused_augment
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.adjust_brightness
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.adjust_contrast
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.adjust_contrast_fast
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.adjust_saturation
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.normalize
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.rgb_to_grayscale
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.crop
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.center_crop
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.functional.horizontal_flip
    options:
      show_root_heading: true
      show_source: false

---

## Utility Functions

::: triton_augment.enable_autotune
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.disable_autotune
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.is_autotune_enabled
    options:
      show_root_heading: true
      show_source: false

---

::: triton_augment.warmup_cache
    options:
      show_root_heading: true
      show_source: false

---

## Input Requirements

### Transform Classes

Transform classes (e.g., `TritonFusedAugment`, `TritonColorJitter`, etc.) accept:

- **Device**: CUDA (GPU) or CPU - *CPU tensors are automatically moved to GPU*
- **Shape**: `(C, H, W)`, `(N, C, H, W)`, or `(N, T, C, H, W)` - *3D, 4D, or 5D (video)*
- **Dtype**: float32 or float16
- **Range**: [0, 1] for color operations (required)

**Notes:**

- 3D tensors `(C, H, W)` are automatically converted to `(1, C, H, W)` internally for processing
- 5D tensors `(N, T, C, H, W)` are supported for video augmentation (batch, frames, channels, height, width)

- For 5D inputs, use `same_on_frame=True` (default) for consistent augmentation across frames, or `same_on_frame=False` for independent per-frame augmentation

- After normalization, values can be outside [0, 1] range

- CPU tensors are automatically transferred to CUDA for GPU processing

### Functional API

Functional functions (e.g., `fused_augment()`, `crop()`, `normalize()`, etc.) expect:

- **Device**: CUDA (GPU) - *must be on CUDA device*
- **Shape**: `(N, C, H, W)` - *4D tensors only*
- **Dtype**: float32 or float16
- **Range**: [0, 1] for color operations (required)

**Note:** Transform classes handle 3D/5D normalization internally. If using the functional API directly, ensure inputs are already in 4D format `(N, C, H, W)`.

---

## Performance Tips

### 1. Use Fused Kernel Even for Partial Operations

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

### 2. Auto-Tuning

Enable auto-tuning for optimal performance on your specific GPU and data sizes:

```python
import triton_augment as ta

ta.enable_autotune()  # Enable once at start of training

# Optional: Pre-compile kernels for your data sizes
ta.warmup_cache(batch_sizes=(32, 64), image_sizes=(224, 512))
```

See [Auto-Tuning Guide](auto-tuning.md) for detailed configuration.

---

## Additional Resources

- **[Quick Start Guide](quickstart.md)**: Training integration examples
- **[Float16 Support](float16.md)**: Half-precision performance and memory savings
- **[Contrast Notes](contrast.md)**: Differences between fast and torchvision-exact contrast
- **[Batch Behavior](batch-behavior.md)**: Understanding `same_on_batch` parameter
- **[Benchmark Results](index.md#performance)**: Detailed performance comparisons

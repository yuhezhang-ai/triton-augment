# Auto-Tuning Guide

!!! info "Default Behavior"
    **Auto-tuning is DISABLED by default** for faster startup and good-enough performance. Enable it only if you need the extra performance boost.

## What is Auto-Tuning?

Auto-tuning automatically finds the optimal kernel configuration (block size, warp count, pipeline depth) for your specific GPU and image sizes.

When enabled, Triton-Augment auto-tunes the **fused kernel** (`fused_color_normalize`) while simple operations (brightness, saturation, normalize) always use fixed defaults.

## Enabling Auto-Tuning

### Option 1: Python API

```python
import triton_augment as ta

# Enable auto-tuning for optimal performance
ta.enable_autotune()

# Check status
print(ta.is_autotune_enabled())  # True

# Disable if needed
ta.disable_autotune()
```

### Option 2: Environment Variable

```bash
export TRITON_AUGMENT_ENABLE_AUTOTUNE=1
python train.py
```

## Auto-Tuning Details

### Auto-Tuned Kernel

The `fused_color_normalize` kernel is auto-tuned across these parameters:

- **BLOCK_SIZE**: Number of elements processed per thread block (256, 512, 1024)
- **num_warps**: Thread group size for parallelism (4, 8)
- **num_stages**: Memory pipeline depth for memory-compute overlap (2, 3, 4)

Triton tests **4 configurations** and caches the fastest one.

### Fixed Kernels

Simple operations use `BLOCK_SIZE=1024` for optimal performance on most GPUs:
- `adjust_brightness`
- `adjust_saturation`
- `normalize`

These don't need auto-tuning as they're simple memory-bound operations.

## How It Works

1. **First run**: Auto-tuning tests 4 configurations and caches the best one (2-5 seconds)
2. **Subsequent runs**: Uses cached optimal configuration (zero overhead)
3. **Per GPU + size**: Cache is specific to your GPU model and total elements (N×C×H×W)

## Cache Warm-Up (When Auto-Tuning is Enabled)

To avoid auto-tuning delays during training, warm up the cache once:

### CLI

```bash
# Enable auto-tuning and warm up
TRITON_AUGMENT_ENABLE_AUTOTUNE=1 python -m triton_augment.warmup \
    --batch-sizes 64,128 \
    --image-sizes 320,384,640
```

### Python API

```python
import triton_augment as ta

# Enable and warm up
ta.enable_autotune()
ta.warmup_cache(
    batch_sizes=(64, 128),
    image_sizes=(320, 384, 640)
)
```

!!! warning "Important Notes"
    - **Auto-tuning must be ENABLED** for warmup to test multiple configs
    - Without auto-tuning, warmup just compiles the default config
    - Auto-tuning is **size-specific**: A cache for 224×224 won't help with 320×320
    - Always use your **actual training dimensions**

## When Auto-Tuning is Disabled (Default)

- Uses a single well-tuned default configuration
- Zero auto-tuning overhead
- No need to warm up cache
- Good performance on most GPUs
- First use still takes ~1-2 seconds to compile (but no testing/selection)

## Performance Impact

### Expected Improvements (When Enabled)

Auto-tuning typically provides:
- **5-15% speedup** on most operations
- Best gains on:
  - Larger images (512×512+)
  - Larger batch sizes (64+)
  - Data center GPUs (A100, H100)
- Minimal gains on:
  - Small images (< 128×128)
  - Small batches (< 16)
  - Consumer GPUs (RTX 30xx series)

### Trade-offs

| Aspect | Disabled (Default) | Enabled |
|--------|-------------------|---------|
| **First-run speed** | Fast (~1-2 sec) | Slow (~5-10 sec) |
| **Steady-state performance** | Good (95-98%) | Optimal (100%) |
| **Cache warmup needed** | No | Yes (recommended) |
| **Best for** | Most users, rapid iteration | Production, max performance |

## Benchmarking With/Without Auto-Tuning

Compare the difference:

```python
import torch
import triton_augment as ta
from triton.testing import do_bench

img = torch.rand(32, 3, 224, 224, device='cuda')

# Benchmark without auto-tuning
ta.disable_autotune()
time_default = do_bench(lambda: ta.fused_color_normalize(
    img, brightness_factor=1.2, saturation_factor=0.9,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
))

# Benchmark with auto-tuning
ta.enable_autotune()
# Wait for auto-tuning to complete (first call only)
ta.fused_color_normalize(img, brightness_factor=1.2, saturation_factor=0.9,
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
time_tuned = do_bench(lambda: ta.fused_color_normalize(
    img, brightness_factor=1.2, saturation_factor=0.9,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
))

print(f"Default config: {time_default:.3f} ms")
print(f"Auto-tuned:     {time_tuned:.3f} ms")
print(f"Speedup:        {time_default/time_tuned:.2f}x")
```

## Recommendation

- **For most users**: Keep auto-tuning **disabled** (default)
  - Faster startup
  - Good performance out-of-the-box
  - No cache management needed

- **For production/max performance**: Enable auto-tuning
  - Warm up cache during deployment
  - Squeeze out last 5-15% performance
  - Worth it for long training runs


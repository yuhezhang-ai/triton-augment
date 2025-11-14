# Auto-Tuning Guide

!!! info "Default Behavior"
    **Auto-tuning is DISABLED by default** for faster startup and good-enough performance. Enable it only if you need the extra performance boost.

## What is Auto-Tuning?

Auto-tuning automatically finds the optimal kernel configuration (block size, warp count, pipeline depth) for your specific GPU and image sizes.

When enabled, Triton-Augment auto-tunes the **fused kernel** (`fused_augment` / `TritonFusedAugment`) while simple operations (brightness, saturation, normalize) always use fixed defaults.

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

The fused kernel (`fused_augment`) is auto-tuned across these parameters:

- **BLOCK_SIZE**: Number of elements processed per thread block (256, 512, 1024, 2048)
- **num_warps**: Thread group size for parallelism (2, 4, 8)
- **num_stages**: Memory pipeline depth for memory-compute overlap (2, 3, 4)

Triton tests **12 configurations** and caches the fastest one for your specific workload.

### Fixed Kernels

Simple operations use `BLOCK_SIZE=1024` for optimal performance on most GPUs:
- `adjust_brightness`
- `adjust_saturation`
- `normalize`

These don't need auto-tuning as they're simple memory-bound operations.

## How It Works

1. **First run**: Auto-tuning tests 12 configurations and caches the best one (5-10 seconds)
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

The recommended workflow is simple: **benchmark the default config first, then enable auto-tuning and benchmark again**.

### Using Benchmark Scripts

!!! warning "Run Default Config First!"
    Always benchmark **without** `--autotune` first, then **with** `--autotune`. Once auto-tuning runs, it caches the config and you can't go back without clearing cache.

```bash
# Step 1: Benchmark default config first
python examples/benchmark.py

# Step 2: Then benchmark with auto-tuning
python examples/benchmark.py --autotune
```

Or the comprehensive benchmark:
```bash
python examples/benchmark_triton.py --autotune
```

### Standard Benchmarking Workflow (Custom Code)

```python
import torch
import triton_augment as ta
from triton.testing import do_bench

img = torch.rand(32, 3, 224, 224, device='cuda')
transform = ta.TritonFusedAugment(
    crop_size=224,
    brightness=(0.8, 1.2),
    saturation=(0.5, 1.5),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# Step 1: Benchmark default config (auto-tuning is disabled by default), run it first!
print("Benchmarking default config...")
time_default = do_bench(lambda: transform(img), warmup=25, rep=100)
print(f"Default config: {time_default:.3f} ms")

# Step 2: Enable auto-tuning and benchmark
print("\nEnabling auto-tuning...")
ta.enable_autotune()

# First call triggers auto-tuning (takes 5-10 seconds, only once)
print("Running auto-tuning (this will take ~5-10 seconds)...")
_ = transform(img)

# Now benchmark with optimal config
print("Benchmarking auto-tuned config...")
time_tuned = do_bench(lambda: transform(img), warmup=25, rep=100)
print(f"Auto-tuned config: {time_tuned:.3f} ms")

# Compare
speedup = time_default / time_tuned
print(f"\nSpeedup: {speedup:.2f}x")
```

### For Reproducible Comparisons (Google Colab)

If you need truly isolated benchmarks on **Google Colab** (where cache may persist across runtime restarts), use two separate colab notebooks:

**Notebook 1 - Default Config:**
```python
import torch
import triton_augment as ta
from triton.testing import do_bench

# Auto-tuning is disabled by default
img = torch.rand(32, 3, 224, 224, device='cuda')
transform = ta.TritonFusedAugment(
    crop_size=224,
    brightness=(0.8, 1.2),
    saturation=(0.5, 1.5),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

time_ms = do_bench(lambda: transform(img), warmup=25, rep=100)
print(f"Default config: {time_ms:.3f} ms")
```

**Notebook 2 - Auto-Tuned Config:**
```python
import torch
import triton_augment as ta
from triton.testing import do_bench

# Enable auto-tuning in fresh notebook
ta.enable_autotune()

img = torch.rand(32, 3, 224, 224, device='cuda')
transform = ta.TritonFusedAugment(
    crop_size=224,
    brightness=(0.8, 1.2),
    saturation=(0.5, 1.5),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# Trigger auto-tuning
_ = transform(img)

time_ms = do_bench(lambda: transform(img), warmup=25, rep=100)
print(f"Auto-tuned config: {time_ms:.3f} ms")
```

!!! note "Colab Cache Behavior"
    Google Colab's cache directory (`~/.triton/cache`) **may persist** across runtime restarts within the same session. Using separate notebooks ensures completely independent benchmarks.

## Benchmarking on Shared/Cloud Services

!!! warning "Instability on Colab, Kaggle, and Cloud GPUs"
    If you're benchmarking on **Google Colab**, **Kaggle Notebooks**, or other **shared cloud services**, you may see unstable or inconsistent results.

### Why Benchmarks Can Be Unstable

Shared GPU services can cause significant performance variability:

1. **Shared Physical GPU** - Multiple users on the same GPU compete for resources
2. **Variable GPU Allocation** - You might get different GPU models between sessions
3. **Thermal Throttling** - GPU performance degrades when hot from other users' workloads
4. **Background Processes** - Cloud platform monitoring and management overhead
5. **Network I/O** - Data transfers can interfere with kernel execution timing

### Symptoms of Instability

You might see:
- **Wildly varying benchmark times** (e.g., 0.5ms one run, 200ms the next)
- **Incorrect speedups** (e.g., 0.00x or negative speedups)
- **Different results between runs** with identical code
- **Auto-tuning picking suboptimal configs** due to noisy measurements

### Best Practices for Stable Benchmarks

If you must benchmark on shared services:

1. **Run multiple iterations and take the median**:
   ```python
   from triton.testing import do_bench
   
   # do_bench already uses median of multiple runs
   time_ms = do_bench(lambda: transform(img), warmup=25, rep=100)
   ```

2. **Warm up thoroughly** before benchmarking:
   ```python
   # Warm up: compile kernels and stabilize GPU state
   for _ in range(10):
       _ = transform(img)
   torch.cuda.synchronize()
   
   # Now benchmark
   time_ms = do_bench(lambda: transform(img))
   ```

3. **Use a dedicated session** - Close other notebooks/tabs using the GPU

4. **Restart runtime** if results seem anomalous

5. **Run at off-peak times** - Early morning or late night (timezone-dependent)

6. **Compare trends, not absolute numbers** - Look for consistent relative speedups

### For Production Benchmarks

For reliable, production-grade benchmarks:

- **Use dedicated GPU instances** (AWS P3/P4, GCP A2, Azure NC-series)
- **Lock GPU clocks** to prevent throttling (requires root):
  ```bash
  sudo nvidia-smi -lgc 1410,1410  # Lock to max clock
  ```
- **Isolate the GPU** - No other processes using it
- **Multiple runs** - Run benchmarks 5-10 times and report mean ± std dev

## Recommendation

- **For most users**: Keep auto-tuning **disabled** (default)
  - Faster startup
  - Good performance out-of-the-box
  - No cache management needed

- **For production/max performance**: Enable auto-tuning
  - Warm up cache during deployment
  - Squeeze out last 5-15% performance
  - Worth it for long training runs


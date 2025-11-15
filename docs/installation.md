# Installation Guide

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Triton >= 2.0.0
- CUDA-capable GPU

## Installation from PyPI (Recommended)

```bash
pip install triton-augment
```

## Development Installation (From Source)

For contributors or those who want to modify the code:

```bash
git clone https://github.com/yuhezhang-ai/triton-augment.git
cd triton-augment
pip install -e ".[dev]"
```

## Input Requirements

!!! warning "Input Requirements"
    - **Range**: Pixel values must be in `[0, 1]` (use `transforms.ToTensor()` if loading from PIL)
    - **Device**: GPU only (CPU tensors are automatically moved to CUDA)
    - **Shape**: Supports both 3D `(C, H, W)` and 4D `(N, C, H, W)` tensors (automatic batching)
    - **Dtype**: `float32` or `float16`

## First Run Behavior

On first use, Triton will compile kernels for your GPU (~1-2 seconds per image size with default config). This is normal and only happens once per GPU and image size.

!!! note "Optional: Cache Warm-Up"
    To avoid compilation delays during training, you can optionally warm up the cache after installation:
    
    ```bash
    python -m triton_augment.warmup
    ```
    
    For more details and auto-tuning optimization, see the [Auto-Tuning Guide](auto-tuning.md).

### What to expect

- **First import**: Helpful message about auto-tuning status (can be suppressed with `TRITON_AUGMENT_SUPPRESS_FIRST_RUN_MESSAGE=1`)
- **First use** of each image size: ~1-2 seconds (kernel compilation)
- **Subsequent uses**: Instant (kernels are cached)

## Verification

Test your installation:

```python
import torch
import triton_augment as ta

# Should work without errors
img = torch.rand(4, 3, 224, 224, device='cuda')
transform = ta.TritonColorJitterNormalize(brightness=0.2)
result = transform(img)
print("✅ Installation successful!")
```


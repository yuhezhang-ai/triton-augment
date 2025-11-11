# Installation Guide

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Triton >= 2.0.0
- CUDA-capable GPU

## From Source (Recommended: uv)

!!! note "About Virtual Environments"
    Virtual environments create a local `.venv/` folder in your project, isolating dependencies per-project.

### Using uv (10-100x faster than pip)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yuhezhang-ai/triton-augment.git
cd triton-augment

# Create .venv/ and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Using pip (Traditional)

```bash
git clone https://github.com/yuhezhang-ai/triton-augment.git
cd triton-augment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e .
```

## Current Limitations

- **Tensor Dimensions**: Only 4D tensors `(N, C, H, W)` are currently supported. Support for 3D `(C, H, W)` and 5D `(N, T, C, H, W)` will be added in a future release.
- **Device**: All tensors must be on CUDA. CPU execution is not supported (Triton requires GPU).

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


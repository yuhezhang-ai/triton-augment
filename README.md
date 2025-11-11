# Triton-Augment

**GPU-Accelerated Image Augmentation with Kernel Fusion**

Triton-Augment is a high-performance image augmentation library that leverages [OpenAI Triton](https://github.com/openai/triton) to fuse common per-pixel operations, providing significant speedups over standard PyTorch implementations.

## 🚀 Key Features

- **Kernel Fusion**: Fuse brightness, contrast, saturation, and normalization into a single GPU kernel
- **Zero Intermediate Memory**: Eliminate DRAM reads/writes between operations
- **Auto-Tuned Performance**: Triton automatically selects optimal kernel configurations for your GPU and image sizes
- **Drop-in Replacement**: Familiar torchvision-like API
- **Significant Speedup**: Faster than sequential PyTorch operations
- **PyTorch Compatible**: Works seamlessly with PyTorch data loading pipelines

## 📦 Installation

### From Source (Recommended: uv)

> **Note**: Virtual environments create a local `.venv/` folder in your project. This isolates dependencies per-project.

**Using uv (10-100x faster than pip)**:

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

**Using pip (Traditional)**:

```bash
git clone https://github.com/yuhezhang-ai/triton-augment.git
cd triton-augment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Triton >= 2.0.0
- CUDA-capable GPU

### Current Limitations

- **Tensor Dimensions**: Only 4D tensors `(N, C, H, W)` are currently supported. Torchvision supports 3D `(C, H, W)` and 5D `(N, T, C, H, W)` as well. Support for these will be added in a future release (see Roadmap).
- **Device**: All tensors must be on CUDA. CPU execution is not supported (Triton requires GPU).

### 🔥 Recommended: Warm Up the Cache

On first use, Triton will auto-tune kernels for your GPU (5-10 seconds per image size). To avoid this delay during training, **warm up the cache once after installation**.

#### Option 1: CLI (Easiest)

```bash
# Use defaults (batch=32,64; size=224,256,512)
python -m triton_augment.warmup

# Specify YOUR training sizes
python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 320,384,640
```

#### Option 2: Python API

```python
import triton_augment as ta

# Use defaults
ta.warmup_cache()

# Specify YOUR training sizes
ta.warmup_cache(batch_sizes=(64, 128), image_sizes=(320, 384, 640))
```

**⚠️ Important**: Specify your **actual training sizes**! Auto-tuning is size-specific. Warming up 224×224 won't help if you train with 320×320.

You only need to do this **once per GPU**. All subsequent runs will be instant!

**What happens if you skip this?**
- **First import** will show a helpful message (printed to stderr)
- **First use** of each image size will take 5-10 seconds (auto-tuning)
- **After that**, performance is optimal (configs are cached)

**Suppress the first-run message** (useful for CI/CD):
```bash
export TRITON_AUGMENT_SUPPRESS_FIRST_RUN_MESSAGE=1
python my_script.py  # No message printed
```

## 🎯 Quick Start

### Basic Usage

```python
import torch
import triton_augment as ta

# Create a batch of images on GPU
images = torch.rand(4, 3, 224, 224, device='cuda')

# Apply fused color jitter and normalization
transform = ta.TritonColorJitterNormalize(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)
```

### Integration with PyTorch DataLoader

```python
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import triton_augment as ta

class GPUTransform:
    """Move tensors to GPU and apply Triton augmentations."""
    def __init__(self):
        self.transform = ta.TritonColorJitterNormalize(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    
    def __call__(self, batch):
        # Move to GPU and apply augmentation
        images = batch[0].cuda()
        labels = batch[1].cuda()
        augmented = self.transform(images)
        return augmented, labels

# Standard PyTorch DataLoader
dataset = ImageFolder('path/to/data')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Apply GPU transforms
gpu_transform = GPUTransform()
for images, labels in loader:
    augmented, labels = gpu_transform((images, labels))
    # ... training code ...
```

## ⚖️ Important: Contrast Difference for Speed

**TL;DR**: This library implements a **different contrast algorithm** than torchvision for speed and fusion.
- `fused_color_normalize()` uses **fast contrast** (not torchvision-exact)
- For exact torchvision results, use individual functions

### Three Equivalent Ways (All Produce Identical Output)

All three approaches below produce **pixel-perfect identical results**:

#### 1. Torchvision

```python
import torchvision.transforms.v2.functional as tvF

img = torch.rand(1, 3, 224, 224, device='cuda')
result = tvF.adjust_brightness(img, 1.2)
result = tvF.adjust_contrast(result, 1.1)
result = tvF.adjust_saturation(result, 0.9)
mean_t = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
std_t = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
result = (result - mean_t) / std_t
```

⏱️ **Speed**: Baseline

#### 2. Triton Individual Functions (Exact)

```python
import triton_augment.functional as F

img = torch.rand(1, 3, 224, 224, device='cuda')
result = F.adjust_brightness(img, 1.2)
result = F.adjust_contrast(result, 1.1)        # Torchvision-exact
result = F.adjust_saturation(result, 0.9)
result = F.normalize(result, 
                     mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225))
```

⏱️ **Speed**: Faster (optimized Triton kernels) ⚡

#### 3. Triton Contrast + Fused (Exact + Fast)

```python
import triton_augment.functional as F

img = torch.rand(1, 3, 224, 224, device='cuda')
# Apply exact contrast first
result = F.adjust_brightness(img, 1.2)
result = F.adjust_contrast(result, 1.1)        # Torchvision-exact

# Then fuse remaining ops (no contrast)
result = F.fused_color_normalize(
    result,
    brightness_factor=1.0,                     # Identity (already applied)
    contrast_factor=1.0,                       # Identity (already applied)
    saturation_factor=0.9,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

⏱️ **Speed**: Fast (2 kernel launches) ⚡⚡

### For Maximum Speed (Not Exact)

If you don't need exact torchvision reproduction:

```python
import triton_augment.functional as F

# Single fused kernel - fastest!
result = F.fused_color_normalize(
    img,
    brightness_factor=1.2,
    contrast_factor=1.1,                       # Fast contrast (different from torchvision)
    saturation_factor=0.9,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

⏱️ **Speed**: Fastest (single fused kernel) 🚀

**⚠️ Note**: Fast contrast uses `(pixel - 0.5) * factor + 0.5` instead of torchvision's blend-with-mean.

### Fast Contrast: Comparison to Other Libraries

| Library | Formula | Type | Speed |
|---------|---------|------|-------|
| **NVIDIA DALI** | `(x - 0.5) * f + 0.5` | Linear (centered) | Fastest ✅ |
| **Triton-Augment (fast)** | `(x - 0.5) * f + 0.5` | Linear (centered) | Fastest ✅ |
| **OpenCV** | `alpha * x + beta` | Linear | Fast |
| **Torchvision** | `x * f + mean * (1-f)` | Linear (mean) | Slower |
| **Scikit-image** | `1/(1+exp(-gain*(x-cut)))` | Sigmoid (S-curve) | Slowest |

**Why this formula?**
- ✅ **Production-proven**: Same as NVIDIA DALI
- ✅ **Fast & fusible**: No mean computation required
- ✅ **Effective for training**: Models learn robustness to augmentation variations
- ⚠️ **Different from torchvision**: Uses fixed 0.5 instead of computed mean

For **exact torchvision reproduction**, use `adjust_contrast()` instead of fast mode.

### Which Should You Use?

| Goal | Method | Speed |
|------|--------|-------|
| Exact torchvision match | Individual functions (#2) | Faster |
| Best speed + exact | Contrast + fused (#3) | Fast |
| Maximum speed | Full fused (#4) | Fastest |

## 🔄 Batch Behavior & Random Augmentation

**Important**: Random augmentations apply the **same parameters to all images in a batch**.

```python
# This applies SAME random augmentation to all 32 images
batch = torch.rand(32, 3, 224, 224, device='cuda')
result = ta.fused_color_normalize(
    batch, 
    random_grayscale_p=0.5,  # Same random decision for all 32
    ...
)
```

### For Different Augmentations Per Image

**Standard practice (recommended)**:
```python
class MyDataset(Dataset):
    def __getitem__(self, idx):
        img = load_image(idx)
        img = self.transform(img)  # Applied to single image
        return img

# DataLoader batches AFTER transforms
# Each image gets different random augmentations
loader = DataLoader(dataset, batch_size=32)
```

**Alternative (if you have pre-batched tensors)**:
```python
# Loop over batch dimension
results = torch.stack([
    ta.fused_color_normalize(img.unsqueeze(0), ...)
    for img in batch
])
```

### Future: Per-Image Randomness

If you need per-image random parameters on pre-batched tensors (like Kornia's `same_on_batch=False`), please open an issue! Implementation is straightforward - just need to pass per-image parameters to the kernel.

## 📚 API Reference

### Transform Classes

#### `TritonColorJitter`

Randomly change the brightness, contrast, and saturation of an image.

```python
ta.TritonColorJitter(
    brightness=0,      # float or (min, max) tuple
    contrast=0,        # float or (min, max) tuple  
    saturation=0       # float or (min, max) tuple
)
```

**Parameters:**
- `brightness`: How much to jitter brightness. brightness_factor is chosen uniformly from `[max(0, 1-brightness), 1+brightness]`. For example, `brightness=0.2` gives range `[0.8, 1.2]`. If tuple, range is `(min, max)`. Default: 0
- `contrast`: How much to jitter contrast. contrast_factor is chosen uniformly from `[max(0, 1-contrast), 1+contrast]`. For example, `contrast=0.2` gives range `[0.8, 1.2]`. If tuple, range is `(min, max)`. Default: 0
- `saturation`: How much to jitter saturation. saturation_factor is chosen uniformly from `[max(0, 1-saturation), 1+saturation]`. For example, `saturation=0.2` gives range `[0.8, 1.2]`. If tuple, range is `(min, max)`. Default: 0

**Example:**
```python
transform = ta.TritonColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
img = torch.rand(1, 3, 224, 224, device='cuda')
augmented = transform(img)
```

#### `TritonNormalize`

Normalize a tensor image with mean and standard deviation.

```python
ta.TritonNormalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

**Parameters:**
- `mean`: Tuple of means for each channel (R, G, B)
- `std`: Tuple of standard deviations for each channel (R, G, B)

**Example:**
```python
normalize = ta.TritonNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
img = torch.rand(1, 3, 224, 224, device='cuda')
normalized = normalize(img)
```

#### `TritonColorJitterNormalize`

Combined color jitter and normalization in a single fused operation. **This is the recommended transform for best performance.**

```python
ta.TritonColorJitterNormalize(
    brightness=0,
    contrast=0,
    saturation=0,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

**Example:**
```python
# Single fused transform (fastest)
transform = ta.TritonColorJitterNormalize(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

### Functional API

The functional API provides lower-level access to the Triton kernels.

#### `fused_color_normalize`

```python
ta.fused_color_normalize(
    input_tensor,
    brightness_factor=0.0,
    contrast_factor=1.0,
    saturation_factor=1.0,
    mean=None,
    std=None
)
```

Apply fused color jitter and normalization in a single kernel.

**Example:**
```python
import triton_augment.functional as F

img = torch.rand(1, 3, 224, 224, device='cuda')
augmented = F.fused_color_normalize(
    img,
    brightness_factor=0.1,
    contrast_factor=1.2,
    saturation_factor=0.8,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

#### `adjust_brightness` / `apply_brightness`

```python
ta.adjust_brightness(input_tensor, brightness_factor)
```

Apply brightness adjustment: `output = input * brightness_factor` (MULTIPLICATIVE)

- `brightness_factor=1.0`: no change (identity)
- `brightness_factor=0.0`: black image
- `brightness_factor=2.0`: doubles brightness

#### `adjust_contrast` / `apply_contrast`

```python
ta.adjust_contrast(input_tensor, contrast_factor)
```

Apply contrast adjustment using blend with grayscale mean (torchvision-exact):
```python
grayscale_mean = mean(rgb_to_grayscale(input))
output = input * contrast_factor + grayscale_mean * (1 - contrast_factor)
```

- `contrast_factor=1.0`: no change (identity)
- `contrast_factor=0.0`: uniform gray image
- `contrast_factor=2.0`: doubles contrast

#### `adjust_contrast_fast`

```python
ta.adjust_contrast_fast(input_tensor, contrast_factor)
```

Apply **FAST** contrast adjustment using centered scaling (same as NVIDIA DALI):
```python
output = (input - 0.5) * contrast_factor + 0.5
```

**Use this when:**
- You want maximum speed with fusion
- Exact torchvision reproduction is not critical
- Training performance is prioritized over exact reproducibility

**Differences from `adjust_contrast()`:**
- ⚡ Faster (no grayscale mean computation)
- ✅ Fully fusible with other operations
- ✅ Same formula as NVIDIA DALI (production-proven)
- ⚠️ NOT pixel-perfect with torchvision
- ✅ Effective for DNN training

See the [comparison table](#fast-contrast-comparison-to-other-libraries) for how this compares to other libraries.

#### `adjust_saturation` / `apply_saturation`

```python
ta.adjust_saturation(input_tensor, saturation_factor)
```

Apply saturation adjustment using blend with grayscale:
```python
grayscale = rgb_to_grayscale(input)  # 0.2989*R + 0.587*G + 0.114*B
output = input * saturation_factor + grayscale * (1 - saturation_factor)
```

- `saturation_factor=0.0`: grayscale image
- `saturation_factor=1.0`: original image
- `saturation_factor=2.0`: highly saturated

#### `normalize` / `apply_normalize`

```python
ta.normalize(input_tensor, mean, std)
```

Apply per-channel normalization: `output[c] = (input[c] - mean[c]) / std[c]`

## 🔥 Performance

Triton-Augment achieves speedups by fusing operations into a single kernel, eliminating intermediate memory reads/writes.

Run `examples/benchmark_triton.py` to benchmark on your hardware.

### Why is it faster?

Traditional approach (PyTorch):
```
GPU Memory ←→ Brightness ←→ GPU Memory ←→ Contrast ←→ GPU Memory ←→ Normalize ←→ GPU Memory
```

Triton-Augment approach:
```
GPU Memory ←→ [Brightness + Contrast + Saturation + Normalize] ←→ GPU Memory
```

By fusing operations, we eliminate 3 round-trips to GPU memory, resulting in significant speedups.

### Auto-Tuning

**Auto-tuning is DISABLED by default** for faster startup and good-enough performance. Enable it only if you need the extra performance boost.

#### Enabling Auto-Tuning (Optional)

**Option 1: Python API**
```python
import triton_augment as ta

# Enable auto-tuning for optimal performance
ta.enable_autotune()

# Check status
print(ta.is_autotune_enabled())  # True
```

**Option 2: Environment Variable**
```bash
export TRITON_AUGMENT_ENABLE_AUTOTUNE=1
python train.py
```

#### Auto-Tuning Details

When enabled, Triton-Augment auto-tunes the **fused kernel** to find the optimal configuration for your GPU. Simple operations (brightness, saturation, normalize) always use fixed defaults.

**Auto-tuned kernel** (`fused_color_normalize`):
- **BLOCK_SIZE**: Number of elements processed per thread block (256, 512, 1024)
- **num_warps**: Thread group size for parallelism (4, 8)
- **num_stages**: Memory pipeline depth for memory-compute overlap (2, 3, 4)

**Fixed kernels** (simple operations): Use `BLOCK_SIZE=1024` for optimal performance on most GPUs

**How it works:**
1. **First run**: Auto-tuning tests 4 configurations and caches the best one (2-5 seconds)
2. **Subsequent runs**: Uses cached optimal configuration (zero overhead)
3. **Per GPU + size**: Cache is specific to your GPU model and total elements

**Recommended workflow (when auto-tuning is enabled):**
```bash
# Enable auto-tuning and warm up the cache (one-time, ~30 seconds)
TRITON_AUGMENT_ENABLE_AUTOTUNE=1 python -m triton_augment.warmup --batch-sizes 64,128 --image-sizes 320,384

# Or Python API:
python -c "import triton_augment as ta; ta.enable_autotune(); ta.warmup_cache(batch_sizes=(64, 128), image_sizes=(320, 384))"
```

**⚠️ Important Notes:**
- **Auto-tuning must be ENABLED** for warmup to test multiple configs. Without it, warmup just compiles the default config.
- Auto-tuning is **size-specific**! A cache for 224×224 won't help with 320×320 images. Always use your actual training dimensions.
- **When auto-tuning is disabled** (default), there's no need to warm up - the default config compiles instantly on first use.

## 🛠️ Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Running Benchmarks

```bash
python examples/benchmark.py
```

## 📋 Roadmap

### Phase 1: MVP (Current) ✅
- [x] Fused color jitter (brightness, contrast, saturation)
- [x] Fused normalization
- [x] Functional and transform APIs
- [x] Documentation and examples

### Phase 2: Advanced Geometrics (Future)
- [ ] Fused random crop (avoid loading discarded pixels)
- [ ] Fused random flip
- [ ] Combined geometric + color transformations
- [ ] Random rotation and affine transforms

### Phase 3: Extended Operations
- [ ] Gaussian blur
- [ ] Random erasing
- [ ] CutMix and MixUp
- [ ] Advanced color transformations (hue, sharpness)
- [ ] Multi-dimensional tensor support (3D: single image, 5D: video batches)
- [ ] Per-image random parameters (like Kornia's `same_on_batch=False`)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Triton](https://github.com/openai/triton) for the incredible GPU programming framework
- [PyTorch](https://pytorch.org/) for the deep learning foundation
- [torchvision](https://github.com/pytorch/vision) for API inspiration

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for the deep learning community**


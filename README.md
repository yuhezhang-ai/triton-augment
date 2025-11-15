# Triton-Augment

**GPU-Accelerated Image Augmentation with Kernel Fusion**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Triton-Augment is a high-performance image augmentation library that leverages [OpenAI Triton](https://github.com/openai/triton) to fuse common transform operations, providing significant speedups over standard PyTorch implementations.

**Key Idea**: Fuse multiple GPU operations into a single kernel → eliminate intermediate memory transfers → faster augmentation.

```python
# Traditional (torchvision Compose): 7 kernel launches
crop → flip → brightness → contrast → saturation → grayscale → normalize

# Triton-Augment Ultimate Fusion: 1 kernel launch 🚀
[crop + flip + brightness + contrast + saturation + grayscale + normalize]
```

---

## 🚀 Features

- **One Kernel, All Operations**: Fuse crop, flip, color jitter, grayscale, and normalize in a single kernel - significantly faster, scales with image size! 🚀
- **Different Parameters Per Sample**: Each image in batch gets different random augmentations (not just batch-wide)
- **Transform & Functional APIs**: Random parameters (transforms) or fixed parameters (functional) - your choice
- **Zero Memory Overhead**: No intermediate buffers between operations
- **Float16 Ready**: ~1.3x speedup on large images + 50% memory savings
- **Drop-in Replacement**: torchvision-like API, easy migration
- **Auto-Tuning**: Optional performance optimization for your GPU

---

## 📦 Quick Start

### Installation

```bash
pip install triton-augment
```

**Requirements**: Python 3.8+, PyTorch 2.0+, CUDA-capable GPU

### Basic Usage

**Recommended: Ultimate Fusion** 🚀

```python
import torch
import triton_augment as ta

# Create batch of images on GPU
images = torch.rand(32, 3, 224, 224, device='cuda')

# Replace torchvision Compose (7 kernel launches)
# With Triton-Augment (1 kernel launch - significantly faster!)
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    random_grayscale_p=0.1,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)  # 🚀 Single kernel for entire pipeline!
```

**Need only some operations?** Use `TritonFusedAugment` with default/no-op values, or use specialized APIs (they all use the same fused kernel internally):

```python
# Option 1: Ultimate API with partial operations (set unused to 0/default)
transform = ta.TritonFusedAugment(
    crop_size=224,
    horizontal_flip_p=0.0,  # No flip
    brightness=0.0,         # No brightness
    saturation=0.2,         # Only saturation
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

# Option 2: Specialized APIs (convenience wrappers, same kernel internally)
color_only = ta.TritonColorJitterNormalize(brightness=0.2, saturation=0.2, ...)
geo_only = ta.TritonRandomCropFlip(size=112, horizontal_flip_p=0.5)
```

[→ More Examples](https://yuhezhang-ai.github.io/triton-augment/quickstart/)

### ⚠️ Input Requirements

- **Range**: Images must be in `[0, 1]` range (e.g., use `torchvision.transforms.ToTensor()`)
- **Device**: GPU (CUDA) - *CPU tensors automatically moved to GPU*
- **Shape**: `(C, H, W)` or `(N, C, H, W)` - *3D tensors automatically batched*
- **Dtype**: `float32` or `float16`

---

## 📚 Documentation

**Full documentation**: https://yuhezhang-ai.github.io/triton-augment (or see `docs/` folder)

| Guide | Description |
|-------|-------------|
| [Quick Start](docs/quickstart.md) | Get started in 5 minutes with examples |
| [Installation](docs/installation.md) | Setup and requirements |
| [API Reference](https://yuhezhang-ai.github.io/triton-augment/api-reference/) | Complete API documentation for all functions and classes |
| [Float16 Support](docs/float16.md) | Use half-precision for ~1.3x speedup (large images) and 50% memory savings |
| [Contrast Notes](docs/contrast.md) | Fused kernel uses fast contrast (different from torchvision). See how to get exact torchvision results |
| [Auto-Tuning](docs/auto-tuning.md) | Optional performance optimization for your GPU and data size (disabled by default). Includes cache warm-up guide |
| [Batch Behavior](docs/batch-behavior.md) | Different parameters per sample (default) vs batch-wide parameters. Understanding `same_on_batch` flag |

---

## ⚡ Performance

### Benchmark Results (Tesla T4 - Google Colab Free Tier)

Real training scenario with random augmentations:

| Image Size | Batch | Crop Size | Torchvision | Triton Fused | Speedup |
|------------|-------|-----------|-------------|--------------|---------|
| 256×256    | 32    | 224×224   | 2.48 ms     | 0.56 ms      | **4.5x** |
| 256×256    | 64    | 224×224   | 4.51 ms     | 0.69 ms      | **6.5x** |
| 600×600    | 32    | 512×512   | 11.82 ms    | 1.26 ms      | **9.4x** |
| 1280×1280  | 32    | 1024×1024 | 48.91 ms    | 4.07 ms      | **12.0x** |

**Average Speedup: 8.1x** 🚀

> Operations: RandomCrop + RandomHorizontalFlip + ColorJitter + RandomGrayscale + Normalize

**Performance scales with image size** — larger images benefit more from kernel fusion:

<p align="center" style="margin: 0;">
  <img src="docs/images/ultimate-fusion-performance.png" alt="Ultimate Fusion Performance" width="480"/>
</p>

*Speedup advantage increases dramatically for larger images (600×600+). Triton maintains near-constant runtime while Torchvision scales linearly.*

<details>
<summary><b>📊 Additional Benchmarks (NVIDIA A100 on Google Colab)</b></summary>

| Image Size | Batch | Crop Size | Torchvision | Triton Fused | Speedup |
|------------|-------|-----------|-------------|--------------|---------|
| 256×256    | 32    | 224×224   | 0.61 ms     | 0.44 ms      | **1.4x** |
| 256×256    | 64    | 224×224   | 0.93 ms     | 0.43 ms      | **2.1x** |
| 600×600    | 32    | 512×512   | 2.19 ms     | 0.50 ms      | **4.4x** |
| 1280×1280  | 32    | 1024×1024 | 8.23 ms     | 0.94 ms      | **8.7x** |

**Average: 4.1x** (A100's high memory bandwidth makes torchvision already fast, so relative improvement is smaller)

> **💡 Why better speedup on T4?** Kernel fusion reduces memory bandwidth bottlenecks, which matters more on bandwidth-limited GPUs like T4 (320 GB/s) vs A100 (1,555 GB/s). This means **greater benefits on consumer and mid-range hardware**.

</details>

### Run Your Own Benchmarks

**Quick Benchmark** (Ultimate Fusion only):
```bash
# Simple, clean table output - easy to run!
python examples/benchmark.py
```

> **Note**: Benchmarks use `torchvision.transforms.v2` (not the legacy v1 API) for comparison.

**Detailed Benchmark** (All operations):
```bash
# Comprehensive analysis with visualizations
python examples/benchmark_triton.py
```

> **💡 Auto-Tuning**: The results above use default configurations. Auto-tuning can provide additional speedup on **dedicated GPUs** (local workstations, cloud instances). On **shared cloud services** (Colab, Kaggle), auto-tuning benefits may be limited due to variable GPU utilization. See [Auto-Tuning Guide](docs/auto-tuning.md) for details.

---

## 🎯 When to Use Triton-Augment?

**💡 Use Triton-Augment + Torchvision together:**
- **Torchvision**: Data loading, resize, ToTensor, rotation, affine, etc.
- **Triton-Augment**: Replace supported operations (currently: crop, flip, color jitter, grayscale, normalize; more coming) with fused GPU kernels

**Best speedup when:**
- Large images (600×600+) or large batches
- Data augmentations are your bottleneck

**Stick with Torchvision only if:**
- Small images (< 256×256) on high-end GPUs (A100+)
- CPU-only training

💡 **TL;DR**: Use both! Triton-Augment replaces Torchvision's fusible ops for 8-12x speedup.

---

<details>
<summary><h2>🎓 Training Examples</h2></summary>

Clean, focused examples showing Triton-Augment integration in real training pipelines:

```bash
# MNIST training example (grayscale images, simple)
python examples/train_mnist.py

# CIFAR-10 training example (RGB images, recommended)
python examples/train_cifar10.py
```

**Key Integration Points:**

| Operation | Use | Why |
|-----------|-----|-----|
| Data Loading | torchvision.datasets | Standard data loading |
| Resize | torchvision.transforms | Not covered by Triton-Augment |
| ToTensor | torchvision.transforms | PIL Image → Tensor conversion |
| Crop, Flip, ColorJitter, Normalize | Triton-Augment | GPU-accelerated, fusible |

**Example Integration:**

```python
import torch
import triton_augment as ta
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Data loading on CPU with workers (fast async I/O!)
train_dataset = datasets.CIFAR10(
    './data', train=True,
    transform=transforms.ToTensor()  # Only ToTensor on CPU
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=128,
    num_workers=4,  # ✅ Use workers for fast async loading!
    pin_memory=True
)

# Step 2: Create GPU augmentation transform (define once, reuse)
augment = ta.TritonFusedAugment(
    crop_size=28,
    horizontal_flip_p=0.5,
    brightness=0.2, contrast=0.2, saturation=0.2,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
    same_on_batch=False  # Each image gets different random params (default)
)

# Step 3: Apply in training loop on GPU batches
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    images = augment(images)  # All ops in 1 kernel per batch! 🚀
    
    outputs = model(images)
    # ... rest of training ...
```

**Why This Pattern:**

- ✅ **Fast async data loading**: `num_workers > 0` for CPU parallelism
- ✅ **Fast GPU batch processing**: All augmentations in 1 fused kernel
- ✅ **Different parameters per sample**: Each image gets different random parameters (default)
- ✅ **Best of both worlds**: CPU for I/O, GPU for compute
- ✅ **Kernel fusion**: No intermediate memory allocations
- ✅ **Large batch advantage**: Speedup increases with batch size

**Note**: Set `same_on_batch=True` if you want all images to share the same random parameters.

💡 **Pro Tip**: Apply Triton-Augment transforms AFTER moving tensors to GPU for maximum performance!

</details>

---

<details>
<summary><h2>🛠️ API Overview</h2></summary>

### Transform Classes (Recommended)

**Multi-operation transforms use the fused kernel** (single kernel for best performance):

```python
import triton_augment as ta

# Ultimate API - full control, all operations (uses fused kernel)
ultimate = ta.TritonFusedAugment(
    crop_size=112, horizontal_flip_p=0.5,
    brightness=0.2, contrast=0.2, saturation=0.2,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

# Specialized APIs - convenience wrappers (also use fused kernel)
color_only = ta.TritonColorJitterNormalize(
    brightness=0.2, saturation=0.2,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)
geo_only = ta.TritonRandomCropFlip(size=112, horizontal_flip_p=0.5)

# Individual transforms (use separate kernels, for maximum control)
crop = ta.TritonRandomCrop(112)
flip = ta.TritonRandomHorizontalFlip(p=0.5)
jitter = ta.TritonColorJitter(brightness=0.2)
```

### Functional API (Low-level)

```python
import triton_augment.functional as F

# Ultimate fusion - ALL operations (single kernel)
result = F.fused_augment(
    img, top=20, left=30, height=112, width=112,
    flip_horizontal=True, brightness_factor=1.2,
    saturation_factor=0.9, mean=(...), std=(...)
)

# Individual operations (separate kernels)
cropped = F.crop(img, top=20, left=30, height=112, width=112)
flipped = F.horizontal_flip(img)
img = F.adjust_brightness(img, 1.2)
img = F.adjust_saturation(img, 0.9)
img = F.normalize(img, mean=(...), std=(...))
```

[→ Complete API Reference](docs/api-reference.md)

</details>

---

## 📋 Roadmap

- [x] **Phase 1**: Fused color operations (brightness, contrast, saturation, normalize)
- [x] **Phase 1.5**: Grayscale, float16 support, auto-tuning
- [x] **Phase 2**: Basic Geometric operations (crop, flip) + Ultimate fusion 🚀
- [ ] **Phase 3**: Extended operations (resize, rotation, blur, erasing, mixup)

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -e ".[dev]"

# Useful commands
make help        # Show all available commands
make test        # Run tests
```

---

## 📝 License

Apache License 2.0 - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [OpenAI Triton](https://github.com/openai/triton) - GPU programming framework
- [PyTorch](https://pytorch.org/) - Deep learning foundation
- [torchvision](https://github.com/pytorch/vision) - API inspiration

---

## 👤 Author

**Yuhe Zhang**

- 💼 LinkedIn: [Yuhe Zhang](https://www.linkedin.com/in/yuhe-zhang-phd/)
- 📧 Email: yuhezhang.zju@gmail.com

*Research interests: Applied ML, Computer Vision, Efficient Deep Learning, GPU Acceleration*

---

<div align="center">

Feel free to file issues or feature requests: [GitHub Issues](https://github.com/yuhezhang-ai/triton-augment/issues)


⭐ **If you find this library useful, please consider starring the repo!** ⭐

[Documentation](https://yuhezhang-ai.github.io/triton-augment) • [GitHub](https://github.com/yuhezhang-ai/triton-augment) • [PyPI](https://pypi.org/project/triton-augment/)

</div>

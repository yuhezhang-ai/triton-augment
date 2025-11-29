<!-- This file is separate from ../README.md
     - README.md: Optimized for GitHub (uses collapsible sections)
     - docs/index.md: Optimized for MkDocs (all sections expanded)
     Update them independently! -->

# Triton-Augment

**GPU-Accelerated Image Augmentation with Kernel Fusion**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/yuhezhang-ai/triton-augment/blob/main/LICENSE)

Triton-Augment is a high-performance image augmentation library that leverages [OpenAI Triton](https://github.com/openai/triton) to fuse common transform operations, providing significant speedups over standard PyTorch implementations.

## ⚡ 5 - 73x Faster than Torchvision/Kornia on Image and Video Augmentation

Replace your augmentation pipeline with a **single fused kernel** and get:

- **Image Speedup**: **8x average speedup** on Tesla T4 and **up to 15.6x faster** on large images (1280×1280) - compared to torchvision.transforms.v2.

- **Video Speedup**: **5D video tensor support** with `same_on_batch=False, same_on_frame=True` control. Speedup: **8.6x vs Torchvision**, **73.7x vs Kornia** 🚀

[📊 See full benchmarks →](#-performance)

**Key Idea**: Fuse multiple GPU operations into a single kernel → eliminate intermediate memory transfers → faster augmentation.

```python
# Traditional (torchvision Compose): 8 kernel launches
affine → crop → flip → brightness → contrast → saturation → grayscale → normalize

# Triton-Augment Ultimate Fusion: 1 kernel launch 🚀
[affine + crop + flip + brightness + contrast + saturation + grayscale + normalize]
```

---

## 🚀 Features

- **One Kernel, All Operations**: Fuse affine (rotation, translation, scaling, shearing), crop, flip, color jitter, grayscale, and normalize in a single kernel - significantly faster, scales with image size! 🚀
- **5D Video Tensor Support**: Native support for `[N, T, C, H, W]` video tensors with `same_on_frame` control for consistent augmentation across frames
- **Different Parameters Per Sample**: Each image in batch gets different random augmentations (not just batch-wide)
- **Zero Memory Overhead**: No intermediate buffers between operations
- **Drop-in Replacement**: torchvision-like transforms & functional APIs, easy migration
- **Auto-Tuning**: Optional performance optimization for your GPU
- **Float16 Ready**: ~1.3x speedup on large images + 50% memory savings

---

## 📦 Quick Start

### Installation

```bash
pip install triton-augment
```

**Requirements**: Python 3.8+, PyTorch 2.0+, CUDA-capable GPU

**Try it now**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aN0V3zjtINXZmj2gZPv9zwNiPrD48qcW) - Test correctness and run benchmarks without local setup

> **Note**: Colab is a shared service - performance may vary due to GPU allocation and resource contention. For stable benchmarking, use a dedicated GPU.

### Basic Usage

**Recommended: Ultimate Fusion** 🚀

```python
import torch
import triton_augment as ta

# Create batch of images on GPU
images = torch.rand(32, 3, 224, 224, device='cuda')

# Replace torchvision Compose (8 kernel launches)
# With Triton-Augment (1 kernel launch - significantly faster!)
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    # Affine parameters
    degrees=15, # rotation
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
    shear=5,
    # Color parameters
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    grayscale_p=0.1,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)  # 🚀 Single kernel for entire pipeline!
```

**Video (5D) Support**: Native support for video tensors `[N, T, C, H, W]`:

```python
# Video batch: 8 videos × 16 frames × 3 channels × 224×224
videos = torch.rand(8, 16, 3, 224, 224, device='cuda')

transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2, contrast=0.2, saturation=0.2,
    same_on_frame=True,  # Same augmentation for all frames (default)
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

augmented = transform(videos)  # Shape: [8, 16, 3, 112, 112]
```

**Need only some operations?** Set unused parameters to their default values:

```python
# Example: Only saturation adjustment + horizontal flip
transform = ta.TritonFusedAugment(
    crop_size=None,          # No crop (pass None or pass same size as input)
    saturation=0.2,         # Only saturation jitter
    horizontal_flip_p=0.5,  # Only random flip
)
```

**Specialized APIs**: For convenience, also available: `TritonColorJitterNormalize`, `TritonRandomCropFlip`, etc.

### 🔗 Combine with Torchvision Transforms

For operations not yet supported by Triton-Augment (like perspective transforms, resize, etc.), combine with torchvision transforms:

```python
import torchvision.transforms.v2 as transforms

# Triton-Augment + Torchvision (per-image randomness + unsupported ops)
transform = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Torchvision (no per-image randomness)
    ta.TritonFusedAugment(              # Triton-Augment (per-image randomness)
        crop_size=224,
        horizontal_flip_p=0.5,
        degrees=15,  # Affine rotation supported!
        brightness=0.2, contrast=0.2, saturation=0.2,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
])
```

**Note**: Torchvision transforms.v2 apply the same random parameters to all images in a batch, while Triton-Augment provides true per-image randomness. [Kornia](https://kornia.readthedocs.io/) also supports per-image randomness, but is slower in our benchmarks.

[→ More Examples](quickstart.md)

### ⚠️ Input Requirements

- **Range**: Images must be in `[0, 1]` range (e.g., use `torchvision.transforms.ToTensor()`)
- **Device**: GPU (CUDA) - *CPU tensors automatically moved to GPU*
- **Shape**: `(C, H, W)`, `(N, C, H, W)`, or `(N, T, C, H, W)` - *5D for video*
- **Dtype**: `float32` or `float16`

---

## 📚 Documentation

**Full documentation**: Navigation menu on the left (or see [GitHub repo](https://github.com/yuhezhang-ai/triton-augment) `docs/` folder)

| Guide | Description |
|-------|-------------|
| [Quick Start](quickstart.md) | Get started in 5 minutes with examples |
| [Installation](installation.md) | Setup and requirements |
| [API Reference](api-reference.md) | Complete API documentation for all functions and classes |
| [Contrast Notes](contrast.md) | Fused kernel uses fast contrast (different from torchvision). See how to get exact torchvision results |
| [Auto-Tuning](auto-tuning.md) | Optional performance optimization for your GPU and data size (disabled by default). Includes cache warm-up guide |
| [Batch Behavior](batch-behavior.md) | Different parameters per sample (default) vs batch-wide parameters. Understanding `same_on_batch` flag |
| [Float16 Support](float16.md) | Use half-precision for ~1.3x speedup (large images) and 50% memory savings |
| [Comparison with Other Libraries](comparison-other-libs.md) | How Triton-Augment compares to DALI, Kornia, and when to use each |

---

## ⚡ Performance

**📊 [Run benchmarks yourself on Google Colab](https://colab.research.google.com/drive/1aN0V3zjtINXZmj2gZPv9zwNiPrD48qcW)** - Verify correctness and performance on free GPU  
*Note: Colab performance may vary due to shared resources*

### Image Augmentation Benchmark Results

**Real training scenario with random augmentations on Tesla T4 (Google Colab Free Tier):**

| Image Size | Batch | Crop Size | Torchvision | Triton Fused | Speedup |
|------------|-------|-----------|-------------|--------------|---------|
| 256×256    | 32    | 224×224   | 3.94 ms     | 1.34 ms      | **2.9x** |
| 256×256    | 64    | 224×224   | 6.84 ms     | 1.42 ms      | **4.8x** |
| 600×600    | 32    | 512×512   | 17.86 ms    | 2.05 ms      | **8.7x** |
| 1280×1280  | 32    | 1024×1024 | 78.48 ms    | 5.02 ms      | **15.6x** |

**Average Speedup: 8.0x** 🚀

> **Operations**: RandomAffine + RandomCrop + RandomHorizontalFlip + ColorJitter + RandomGrayscale + Normalize

> **Note**: Benchmarks use `torchvision.transforms.v2` (not the legacy v1 API) for comparison.

**Performance scales with image size** — larger images benefit more from kernel fusion:
<p align="left" style="margin: 0;">
  <img src="images/ultimate-fusion-performance.png" alt="Ultimate Fusion Performance" width="480"/>
</p>

**📊 Additional Benchmarks (NVIDIA A100 on Google Colab):**

| Image Size | Batch | Crop Size | Torchvision | Triton Fused | Speedup |
|------------|-------|-----------|-------------|--------------|---------|
| 256×256    | 32    | 224×224   | 0.61 ms     | 0.44 ms      | **1.4x** |
| 256×256    | 64    | 224×224   | 0.93 ms     | 0.43 ms      | **2.1x** |
| 600×600    | 32    | 512×512   | 2.19 ms     | 0.50 ms      | **4.4x** |
| 1280×1280  | 32    | 1024×1024 | 8.23 ms     | 0.94 ms      | **8.7x** |

**Average: 4.1x**

> **Why better speedup on T4?** Kernel fusion reduces memory bandwidth bottlenecks, which matters more on bandwidth-limited GPUs like T4 (320 GB/s) vs A100 (1,555 GB/s). This means **greater benefits on consumer and mid-range hardware**.

### Video (5D Tensor) Benchmarks

**Video augmentation on Tesla T4 (Google Colab Free Tier) - Input shape `[N, T, C, H, W]`:**

| Batch | Frames | Image Size | Crop Size | Torchvision | Kornia VideoSeq | Triton Fused | Speedup vs TV | Speedup vs Kornia |
|-------|--------|------------|-----------|-------------|-----------------|--------------|---------------|-------------------|
|     8 |     16 | 256×256    | 224×224   | 13.96 ms    | 88.80 ms        | 1.80 ms      | **7.8x**      | **49.5x**         |
|     8 |     32 | 256×256    | 224×224   | 26.51 ms    | 177.58 ms       | 2.65 ms      | **10.0x**     | **67.1x**         |
|    16 |     32 | 256×256    | 224×224   | 50.12 ms    | 346.25 ms       | 3.86 ms      | **13.0x**     | **89.7x**         |
|     8 |     32 | 512×512    | 448×448   | 107.20 ms   | 612.65 ms       | 6.83 ms      | **15.7x**     | **89.7x**         |

**Average Speedup vs Torchvision: 11.62x**  
**Average Speedup vs Kornia: 73.97x** 🚀

### Run Your Own Benchmarks

**Quick Benchmark** (Ultimate Fusion only):
```bash
# Simple, clean table output - easy to run!
python examples/benchmark.py
python examples/benchmark_video.py
```

**Detailed Benchmark** (All operations):
```bash
# Comprehensive analysis with visualizations
python examples/benchmark_triton.py
```

---

## 💡 Auto-Tuning

All benchmark results shown above use default kernel configurations. Auto-tuning can provide **additional speedup** on dedicated GPUs.

**What is Auto-Tuning?**

Triton kernels have tunable parameters (block sizes, warps per thread, etc.) that affect performance. Auto-tuning automatically searches for the optimal configuration for **your specific GPU and data sizes**.

**When to use:**

- ✅ **Dedicated GPUs** (local workstations, cloud instances): 10-30% additional speedup

- ⚠️ **Shared services** (Colab, Kaggle): Limited benefits, but can help stabilize performance

**Quick start:**
```python
import triton_augment as ta

ta.set_autotune(True)  # Enable auto-tuning (one-time cost, results cached)
transform = ta.TritonFusedAugment(...)
augmented = transform(images)  # First run: tests configs; subsequent: uses cache
```

**⚠️ Performance Variability**: Our highly optimized kernels are more sensitive to resource contention. If you experience **sudden latency spikes** on shared services, this is expected due to competing workloads. Auto-tuning can help find more stable configurations.

📖 **Full guide**: [Auto-Tuning Guide](auto-tuning.md) - Detailed instructions, cache management, and warm-up strategies

---

## 🎯 When to Use Triton-Augment?

**Use Triton-Augment + Torchvision together:**

- **Torchvision**: Data loading, resize, ToTensor, perspective transforms, etc.
- **Triton-Augment**: Replace supported operations (currently: affine, rotate, crop, flip, color jitter, grayscale, normalize; more coming) with fused GPU kernels

**Best speedup when:**

- Large images (500x500+) or large batches
- Data augmentations are your bottleneck

**Stick with Torchvision only if:**

- CPU-only training
- Experiencing **extreme latency variability** on **shared services** (e.g., consistent 10x+ spikes) - our optimized kernels are more sensitive to resource contention. Try auto-tuning first; if instability persists, Torchvision may be more stable

💡 **TL;DR**: Use both! Triton-Augment replaces Torchvision's fusible ops for 8-12x speedup.

---

## 🎓 Training Integration

Want to use Triton-Augment in your training pipeline? See the **[Quick Start Guide](quickstart.md)** for:

- Complete training examples (MNIST, CIFAR-10)
- DataLoader integration patterns
- Best practices for CPU data loading + GPU augmentation
- Why this architecture is fast

Quick snippet:

```python
# Step 1: Load data on CPU with workers
train_loader = DataLoader(..., num_workers=4)

# Step 2: Create GPU augmentation (once)
augment = ta.TritonFusedAugment(crop_size=28, ...)

# Step 3: Apply in training loop on GPU batches
for images, labels in train_loader:
    images = images.cuda()
    images = augment(images)  # 🚀 1 kernel for all ops!
    outputs = model(images)
```

[→ Full Training Guide](quickstart.md)

---

## 📋 Roadmap

- [x] **Phase 1**: Fused color operations (brightness, contrast, saturation, normalize)
- [x] **Phase 1.5**: Grayscale, float16 support, auto-tuning
- [x] **Phase 2**: Basic Geometric operations (crop, flip) + all fusion 🚀
- [x] **Phase 2.5**: 5D video tensor support `[N, T, C, H, W]` with `same_on_frame` parameter
- [x] **Phase 3.0**: Affine transformations (rotation, translation, scaling, shearing) in fused kernel
- [ ] **Phase 3.5**: Extended operations (blur, erasing, mixup, etc.)
- [ ] **Future**: Differentiable augmentation (autograd support, available in Kornia) - evaluate demand vs performance tradeoff

[→ Submit Feature Request](https://github.com/yuhezhang-ai/triton-augment/issues)

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://github.com/yuhezhang-ai/triton-augment/blob/main/CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -e ".[dev]"

# Useful commands
make help        # Show all available commands
make test        # Run tests
```

[→ Complete Contributing Guide](https://github.com/yuhezhang-ai/triton-augment/blob/main/CONTRIBUTING.md)

---

## 📝 License

Apache License 2.0 - see [LICENSE](https://github.com/yuhezhang-ai/triton-augment/blob/main/LICENSE) file.

---

## 🙏 Acknowledgments

- [OpenAI Triton](https://github.com/openai/triton) - GPU programming framework
- [PyTorch](https://pytorch.org/) - Deep learning foundation
- [torchvision](https://github.com/pytorch/vision) - API inspiration

---

## 👤 Author

**Yuhe Zhang**

- 💼 LinkedIn: [Yuhe Zhang](https://www.linkedin.com/in/yuhe-zhang-phd/)
- 📧 Email: yuhezhang.zju @ gmail.com

*Research interests: Applied ML, Computer Vision, Efficient Deep Learning, GPU Acceleration*

---

## 📧 Project

- **Issues and feature requests**: [GitHub Issues](https://github.com/yuhezhang-ai/triton-augment/issues)
- **PyPI Package**: [pypi.org/project/triton-augment](https://pypi.org/project/triton-augment/)

---

⭐ **If you find this library useful, please consider starring the repo!** ⭐

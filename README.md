# Triton-Augment

**GPU-Accelerated Image Augmentation with Kernel Fusion**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Triton-Augment is a high-performance image augmentation library that leverages [OpenAI Triton](https://github.com/openai/triton) to fuse common transform operations, providing significant speedups over standard PyTorch implementations.

**Key Idea**: Fuse multiple GPU operations into a single kernel → eliminate intermediate memory transfers → faster augmentation.

```python
# Traditional (torchvision Compose): 7 kernel launches
crop → flip → brightness → contrast → saturation → grayscale (optional) → normalize

# Triton-Augment Ultimate Fusion: 1 kernel launch 🚀
[crop + flip + brightness + contrast + saturation + grayscale + normalize]
```

---

## 🚀 Features

- **One Kernel, All Operations**: Fuse crop, flip, color jitter, grayscale, and normalize in a single kernel - ~3-5x faster! 🚀
- **Different Parameters Per Sample**: Each image in batch gets different random augmentations (not just batch-wide)
- **Transform & Functional APIs**: Random parameters (transforms) or fixed parameters (functional) - your choice
- **Zero Memory Overhead**: No intermediate buffers between operations
- **Float16 Ready**: Additional 1.3-2x speedup with half-precision
- **Drop-in Replacement**: torchvision-like API, easy migration
- **Auto-Tuning**: Optional performance optimization for your GPU

---

## 📦 Quick Start

### Installation

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -e .

# Using pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Requirements**: Python 3.8+, PyTorch 2.0+, CUDA-capable GPU

[→ Full Installation Guide](https://triton-augment.readthedocs.io/installation/)

### Basic Usage

**Option 1: Ultimate Fusion (Fastest - Recommended)** 🚀

```python
import torch
import triton_augment as ta

# Create batch of images on GPU
images = torch.rand(32, 3, 224, 224, device='cuda')

# Replace torchvision Compose (6 kernel launches)
# With Triton-Augment (1 kernel launch - 3-5x faster!)
transform = ta.TritonFusedAugment(
    crop_size=112,
    horizontal_flip_p=0.5,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)  # 🚀 Single kernel for entire pipeline!
```

**Option 2: Pixel-Only Fusion**

```python
# If you only need color operations (no crop/flip)
transform = ta.TritonColorJitterNormalize(
    brightness=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)  # Single kernel for color + normalize
```

[→ More Examples](https://triton-augment.readthedocs.io/quickstart/)

---

## 📚 Documentation

**Full documentation**: https://triton-augment.readthedocs.io (or see `docs/` folder)

| Guide | Description |
|-------|-------------|
| [Quick Start](docs/quickstart.md) | Get started in 5 minutes with examples |
| [Installation](docs/installation.md) | Setup and requirements |
| [API Reference](docs/api-reference.md) | Complete API documentation for all functions and classes |
| [Float16 Support](docs/float16.md) | Use half-precision for 1.3-2x speedup and 50% memory savings |
| [Contrast Notes](docs/contrast.md) | **Important**: Fused kernel uses fast contrast (different from torchvision). See how to get exact torchvision results |
| [Auto-Tuning](docs/auto-tuning.md) | Optional performance optimization for your GPU and data size (disabled by default). Includes cache warm-up guide |
| [Batch Behavior](docs/batch-behavior.md) | Different parameters per sample (default) vs batch-wide parameters. Understanding `same_on_batch` flag |

---

## ⚡ Performance

By fusing operations, Triton-Augment eliminates intermediate memory transfers:

```
Traditional:  GPU ←→ Crop ←→ GPU ←→ Flip ←→ GPU ←→ Brightness ←→ GPU ←→ Contrast ←→ GPU ←→ Saturation ←→ GPU ←→ Normalize ←→ GPU
                     ❌ Slow (multiple memory transfers)

Triton:       GPU ←→ [Crop + Flip + Brightness + Contrast + Saturation + Normalize] ←→ GPU
                     ✅ Fast (single memory transfer)
```

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

---

## 🔥 Use Cases

✅ Large-scale training (ImageNet, COCO)  
✅ Real-time inference pipelines  
✅ Mixed-precision training (float16)  
✅ Memory-constrained scenarios  

---

## 🎓 Training Examples

Clean, focused examples showing Triton-Augment integration in real training pipelines:

```bash
# MNIST training example (grayscale images, simple)
python examples/train_mnist.py

# CIFAR-10 training example (RGB images, recommended)
python examples/train_cifar10.py
```

These examples demonstrate:
- ✅ Fast async data loading with `num_workers > 0`
- ✅ GPU batch augmentation in training loop
- ✅ All operations fused in 1 kernel per batch
- ✅ Best of both worlds: CPU for I/O, GPU for compute
- ✅ Real production training setup

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

**Note**: Set `same_on_batch=True` if you want all images to share the same random parameters (slightly faster, but less useful for training).

💡 **Pro Tip**: Apply Triton-Augment transforms AFTER moving tensors to GPU for maximum performance!

---

## 🛠️ API Overview

### Transform Classes (Recommended)

```python
import triton_augment as ta

# Ultimate fusion - ALL operations in ONE kernel (fastest!) 🚀
ultimate = ta.TritonFusedAugment(
    crop_size=112, horizontal_flip_p=0.5,
    brightness=0.2, contrast=0.2, saturation=0.2,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

# Pixel fusion - color operations + normalize
pixel_fused = ta.TritonColorJitterNormalize(
    brightness=0.2, saturation=0.2,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

# Geometric fusion - crop + flip
geo_fused = ta.TritonRandomCropFlip(size=112, horizontal_flip_p=0.5)

# Individual transforms
crop = ta.TritonRandomCrop(112)
flip = ta.TritonRandomHorizontalFlip(p=0.5)
jitter = ta.TritonColorJitter(brightness=0.2)
```

### Functional API (Low-level)

```python
import triton_augment.functional as F

# Ultimate fusion - ALL operations
result = F.ultimate_fused_augment(
    img, top=20, left=30, height=112, width=112,
    flip_horizontal=True, brightness_factor=1.2,
    saturation_factor=0.9, mean=(...), std=(...)
)

# Geometric operations
cropped = F.crop(img, top=20, left=30, height=112, width=112)
flipped = F.horizontal_flip(img)
# Geometric fused
result = F.fused_crop_flip(img, 20, 30, 112, 112, flip_horizontal=True)

# Pixel operations
img = F.adjust_brightness(img, 1.2)
img = F.adjust_saturation(img, 0.9)
img = F.normalize(img, mean=(...), std=(...))
# Pixel fused
result = F.fused_color_normalize(
    img, brightness_factor=1.2, saturation_factor=0.9,
    mean=(...), std=(...)
)
```

[→ Complete API Reference](docs/api-reference.md)

---

## 📋 Roadmap

- [x] **Phase 1**: Fused color operations (brightness, contrast, saturation, normalize)
- [x] **Phase 1.5**: Grayscale, float16 support, auto-tuning
- [x] **Phase 2**: Geometric operations (crop, flip) + Ultimate fusion 🚀
- [ ] **Phase 3**: Extended operations (rotation, blur, erasing, mixup)

[→ Detailed Roadmap](https://github.com/yuhezhang-ai/triton-augment/issues)

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -e ".[dev]"
pytest tests/
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

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yuhezhang-ai/triton-augment/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yuhezhang-ai/triton-augment/discussions)

---

<div align="center">

**Made with ❤️ for the deep learning community**

[Documentation](https://triton-augment.readthedocs.io) • [GitHub](https://github.com/yuhezhang-ai/triton-augment) • [PyPI](https://pypi.org/project/triton-augment/)

</div>

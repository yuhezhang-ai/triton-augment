# Triton-Augment

**GPU-Accelerated Image Augmentation with Kernel Fusion**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Triton-Augment is a high-performance image augmentation library that leverages [OpenAI Triton](https://github.com/openai/triton) to fuse common per-pixel operations, providing significant speedups over standard PyTorch implementations.

**Key Idea**: Fuse multiple GPU operations into a single kernel → eliminate intermediate memory transfers → faster augmentation.

```python
# Traditional (torchvision Compose): 6 kernel launches
crop → flip → brightness → contrast → saturation → normalize

# Triton-Augment Ultimate Fusion: 1 kernel launch 🚀
[crop + flip + brightness + contrast + saturation + normalize]
```

---

## 🚀 Features

- **Ultimate Fusion**: ALL augmentations (crop, flip, color jitter, normalize) in ONE kernel - ~3-5x faster! 🚀
- **Three Fusion Levels**: Ultimate (all ops), specialized (geometric/pixel), or individual operations
- **Zero Intermediate Memory**: No temporary buffers between operations
- **Float16 Support**: Additional 1.3-2x speedup on modern GPUs
- **torchvision-like API**: Familiar interface
- **Auto-Tuning**: Optional performance optimization

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
transform = ta.TritonUltimateAugment(
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
| [Auto-Tuning](docs/auto-tuning.md) | Optional performance optimization (disabled by default). Includes cache warm-up guide |
| [Batch Behavior](docs/batch-behavior.md) | **Important**: Random augmentations use same parameters for all images in batch. Learn how to get per-image randomness |

---

## ⚡ Performance

By fusing operations, Triton-Augment eliminates intermediate memory transfers:

```
Traditional:  GPU ←→ Op1 ←→ GPU ←→ Op2 ←→ GPU ←→ Op3 ←→ GPU
                     ❌ Multiple memory transfers

Triton:       GPU ←→ [Op1 + Op2 + Op3] ←→ GPU
                     ✅ Single memory transfer
```

**Quick Benchmark** (Ultimate Fusion only):
```bash
# Simple, clean table output - easy to run!
python examples/benchmark.py
```

> **Note**: Benchmarks use `torchvision.transforms.v2` (not the legacy v1 API).

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

## 🛠️ API Overview

### Transform Classes (Recommended)

```python
import triton_augment as ta

# Ultimate fusion - ALL operations in ONE kernel (fastest!) 🚀
ultimate = ta.TritonUltimateAugment(
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

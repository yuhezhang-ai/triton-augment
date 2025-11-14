# Welcome to Triton-Augment

<div class="hero-section" markdown>

**GPU-Accelerated Image Augmentation with Kernel Fusion**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/yuhezhang-ai/triton-augment/blob/main/LICENSE)

[Installation](installation.md){ .md-button .md-button--primary }
[Quick Start](quickstart.md){ .md-button }
[API Reference](api-reference.md){ .md-button }

</div>

---

## What is Triton-Augment?

Triton-Augment is a high-performance image augmentation library that leverages [OpenAI Triton](https://github.com/openai/triton) to **fuse common transform operations**, providing significant speedups over standard PyTorch implementations.

### Key Idea

Instead of launching separate GPU kernels for each operation:

```
Traditional: GPU ←→ Crop ←→ GPU ←→ Flip ←→ GPU ←→ Brightness ←→ GPU ←→ Contrast ←→ GPU ←→ Saturation ←→ GPU ←→ Normalize ←→ GPU
                    ❌ Slow (multiple memory transfers)
```

Triton-Augment fuses operations into a single kernel:

```
Triton-Augment: GPU ←→ [Crop + Flip + Brightness + Contrast + Saturation + Normalize] ←→ GPU
                       ✅ Fast (single memory transfer)
```

**Result**: Faster augmentation with zero intermediate memory allocations.

---

## 🚀 Key Features

- **One Kernel, All Operations**: Fuse crop, flip, color jitter, grayscale, and normalize in a single kernel - ~3-5x faster! 🚀
- **Per-Image Randomness**: Each image in batch gets different random augmentations (not just batch-wide)
- **Transform & Functional APIs**: Random parameters (transforms) or fixed parameters (functional) - your choice
- **Zero Memory Overhead**: No intermediate buffers between operations
- **Float16 Ready**: Additional 1.3-2x speedup with half-precision
- **Drop-in Replacement**: torchvision-like API, easy migration
- **Auto-Tuning**: Optional performance optimization for your GPU

---

## Quick Example

### Ultimate Fusion (Single Kernel for ALL Operations) 🚀

```python
import torch
import triton_augment as ta

# Create images on GPU
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

### Pixel-Only Fusion

```python
# If you only need color operations
transform = ta.TritonColorJitterNormalize(
    brightness=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

augmented = transform(images)  # Single kernel for color + normalize
```

---

## Performance

Triton-Augment achieves speedups by eliminating intermediate memory transfers:

- **Fused operations**: Single kernel launch for entire pipeline
- **Optimized kernels**: Triton-generated GPU code
- **Float16 support**: Additional 1.3-2x speedup on modern GPUs

### Benchmark Results (NVIDIA A100)

Real training scenario with random augmentations:

| Image Size | Batch | Torchvision | Triton Fused | Speedup |
|------------|-------|-------------|--------------|---------|
| 256×256    | 32    | 0.61 ms     | 0.44 ms      | **1.4x** |
| 256×256    | 64    | 0.93 ms     | 0.43 ms      | **2.1x** |
| 600×600    | 32    | 2.19 ms     | 0.50 ms      | **4.4x** |
| 1280×1280  | 32    | 8.23 ms     | 0.94 ms      | **8.7x** |

**Average Speedup: 4.1x** 🚀

> Operations: RandomCrop + RandomHorizontalFlip + ColorJitter + RandomGrayscale + Normalize

Run `python examples/benchmark.py` to benchmark on your hardware.

---

## Who Should Use This?

✅ **Deep learning researchers** training vision models  
✅ **ML engineers** building production pipelines  
✅ **Anyone** using GPU-based data augmentation  

Perfect for:
- Large-scale training (ImageNet, COCO, etc.)
- Real-time inference pipelines
- Mixed-precision training (float16)
- Memory-constrained scenarios

---

## Why Not torchvision?

Torchvision is excellent, but:

| Aspect | torchvision | Triton-Augment |
|--------|-------------|----------------|
| **Speed** | Good | Faster (kernel fusion) |
| **Memory** | Standard | Lower (no intermediate buffers) |
| **API** | Mature | torchvision-inspired |
| **Flexibility** | High | Focused on performance |

**Use torchvision** if you need maximum flexibility and CPU support.  
**Use Triton-Augment** if you want maximum GPU performance.

---

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __Installation__

    ---

    Install Triton-Augment and set up your environment

    [:octicons-arrow-right-24: Get started](installation.md)

-   :material-rocket-launch:{ .lg .middle } __Quick Start__

    ---

    Learn the basics with simple examples

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-book-open:{ .lg .middle } __User Guide__

    ---

    Deep dive into features and best practices

    [:octicons-arrow-right-24: User Guide](float16.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete API documentation

    [:octicons-arrow-right-24: API Reference](api-reference.md)

</div>

---

## Project Status

**Phase 1**: MVP with fused color operations ✅  
**Phase 2**: Geometric operations + Ultimate fusion ✅  
**Phase 3**: Extended operations (blur, erasing, rotation) 📋

See the [Roadmap](https://github.com/yuhezhang-ai/triton-augment#roadmap) for details.

### Latest Addition: Ultimate Fusion 🚀

Triton-Augment now supports **the ultimate fused kernel** that combines ALL 6 operations in a single GPU kernel:
- Crop + Flip (geometric)
- Brightness + Contrast + Saturation + Normalize (pixel)

**Result**: ~3-5x faster than torchvision Compose!

---

## Community

- **GitHub**: [yuhezhang-ai/triton-augment](https://github.com/yuhezhang-ai/triton-augment)
- **Issues**: Report bugs or request features
- **Contributions**: Pull requests welcome!

---

Made with ❤️ for the deep learning community


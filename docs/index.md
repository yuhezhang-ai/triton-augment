# Welcome to Triton-Augment

<div align="center">

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

Triton-Augment is a high-performance image augmentation library that leverages [OpenAI Triton](https://github.com/openai/triton) to **fuse common per-pixel operations**, providing significant speedups over standard PyTorch implementations.

### Key Idea

Instead of launching separate GPU kernels for each operation:

```
Traditional: GPU ←→ Brightness ←→ GPU ←→ Contrast ←→ GPU ←→ Normalize ←→ GPU
                    ❌ Slow (multiple memory transfers)
```

Triton-Augment fuses operations into a single kernel:

```
Triton-Augment: GPU ←→ [Brightness + Contrast + Saturation + Normalize] ←→ GPU
                       ✅ Fast (single memory transfer)
```

**Result**: Faster augmentation with zero intermediate memory allocations.

---

## 🚀 Key Features

- **Ultimate Fusion**: Combine ALL augmentations (crop, flip, color jitter, normalize) in a **single GPU kernel** - ~3-5x faster! 🚀
- **Three Fusion Levels**: Choose between ultimate (all ops), specialized (geometric/pixel), or individual operations
- **Zero Intermediate Memory**: Eliminate DRAM reads/writes between operations
- **Float16 Support**: Full support for half-precision with additional performance gains
- **Auto-Tuned Performance**: Optional auto-tuning for optimal kernel configurations
- **Drop-in Replacement**: Familiar torchvision-like API
- **PyTorch Compatible**: Works seamlessly with PyTorch data loading pipelines

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

Run `examples/benchmark_triton.py` to benchmark on your hardware.

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


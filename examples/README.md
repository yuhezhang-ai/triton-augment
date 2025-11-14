# Examples

This directory contains various examples demonstrating how to use Triton-Augment.

## 🎓 Training Examples (Start Here!)

Real-world training examples showing how to integrate Triton-Augment with torchvision:

### `train_mnist.py` - MNIST Training
```bash
python examples/train_mnist.py
```

Simple training example with grayscale images:
- Clean, focused demonstration of production-ready integration
- Fast async data loading with `num_workers > 0`
- GPU batch augmentation in training loop
- ~3-5 minutes to run on most GPUs

**What you'll learn:**
- Correct integration pattern for maximum performance
- Using num_workers for fast async data loading
- Applying augmentations on GPU batches in training loop
- Best practices for production training

### `train_cifar10.py` - CIFAR-10 Training (Recommended)
```bash
python examples/train_cifar10.py
```

Realistic example with RGB images:
- Full augmentation pipeline (crop, flip, color jitter, grayscale, normalize)
- Demonstrates `TritonFusedAugment` (all ops in 1 fused kernel per batch!)
- Fast async data loading + GPU batch augmentation
- Uses ResNet-18 for realistic model complexity
- ~10-20 minutes to run on most GPUs

**What you'll learn:**
- Production-ready integration pattern
- Full augmentation pipeline with Triton-Augment
- Why batch processing on GPU is faster than per-image
- Best practices for large-scale training

---

## 📊 Benchmark Scripts

Performance measurement and comparison tools:

### `benchmark.py` - Quick Benchmark (Recommended First)
```bash
python examples/benchmark.py
```

Simple, clean benchmark of ultimate fusion:
- Easy to run and understand
- Shows speedup for different batch sizes and crop sizes
- Outputs a clean table
- **Run this first** to see if Triton-Augment helps your use case

### `benchmark_triton.py` - Comprehensive Benchmark
```bash
python examples/benchmark_triton.py
```

Detailed performance analysis:
- 7 different benchmarks covering all operations
- Compares individual ops, fusion levels, and float16 vs float32
- Generates plots and detailed metrics
- Takes longer to run but provides deep insights

---

## 🎨 Visualization

### `visualize_augmentations.py` - Visual Inspection
```bash
python examples/visualize_augmentations.py
```

Visualize what augmentations look like:
- Compare Triton-Augment with torchvision side-by-side
- Save augmented images to disk
- Useful for debugging and understanding augmentations

---

## 🔥 Warmup Example

### `warmup_example.py` - Kernel Cache Warmup
```bash
python examples/warmup_example.py
```

Shows how to pre-compile kernels for specific sizes:
- Reduces first-run latency
- Useful for production deployments
- Only relevant when auto-tuning is enabled

---

## 🚀 Quick Start Guide

**New to Triton-Augment?** Follow this order:

1. **Start with a training example** to see real-world integration:
   ```bash
   python examples/train_cifar10.py  # Most realistic
   # OR
   python examples/train_mnist.py    # Simpler, faster
   ```

2. **Run a quick benchmark** to see performance gains:
   ```bash
   python examples/benchmark.py
   ```

3. **Explore comprehensive benchmarks** for deeper understanding:
   ```bash
   python examples/benchmark_triton.py
   ```

4. **Visualize augmentations** to understand what's happening:
   ```bash
   python examples/visualize_augmentations.py
   ```

---

## 💡 Key Integration Points

When using Triton-Augment in your own code:

| Operation | Use | Why |
|-----------|-----|-----|
| **Data Loading** | `torchvision.datasets` | Standard data loading |
| **Resize** | `torchvision.transforms.Resize` | Not covered by Triton-Augment currently |
| **ToTensor** | `torchvision.transforms.ToTensor` | PIL Image → Tensor |
| **Crop** | `triton_augment` | GPU-accelerated, fusible |
| **Flip** | `triton_augment` | GPU-accelerated, fusible |
| **ColorJitter** | `triton_augment` | GPU-accelerated, fusible |
| **RandomGrayscale** | `triton_augment` | GPU-accelerated, fusible |
| **Normalize** | `triton_augment` | GPU-accelerated, fusible |

**The Golden Rule:** 
- Use **torchvision** for CPU operations (data loading, resize, toTensor)
- Use **Triton-Augment** for GPU operations (crop, flip, color jitter, normalize)
- Apply Triton-Augment transforms **AFTER** moving tensors to GPU

---

## 📚 Further Reading

- [Full Documentation](../docs/README.md)
- [API Reference](../docs/api-reference.md)
- [Integration Guide](../docs/quickstart.md)


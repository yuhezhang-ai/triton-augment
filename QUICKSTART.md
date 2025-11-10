# Triton-Augment Quick Start Guide

Get started with Triton-Augment in 5 minutes!

## Installation

> **Quick explanation**: Virtual environments create a local `.venv/` folder in your project with isolated Python packages. You activate it to tell your shell to use those packages instead of system ones.

### Using uv (Recommended - 10-100x faster)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/yuhezhang-ai/triton-augment.git
cd triton-augment

# Create local .venv/ and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Using pip (Traditional)

```bash
# Clone the repository
git clone https://github.com/yuhezhang-ai/triton-augment.git
cd triton-augment

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Prerequisites

Make sure you have:
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- A CUDA-capable GPU

## Basic Usage

### 1. Simple Color Jitter

```python
import torch
import triton_augment as ta

# Create images on GPU
images = torch.rand(4, 3, 224, 224, device='cuda')

# Apply color jitter
transform = ta.TritonColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2
)

augmented = transform(images)
```

### 2. With Normalization (Recommended)

```python
import triton_augment as ta

# Fused color jitter + normalization for maximum speed
transform = ta.TritonColorJitterNormalize(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),  # ImageNet mean
    std=(0.229, 0.224, 0.225)     # ImageNet std
)

augmented = transform(images)
```

### 3. In a Training Loop

```python
import torch
import triton_augment as ta
from torch.utils.data import DataLoader

# Setup
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
augment = ta.TritonColorJitterNormalize(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# Training loop
for images, labels in train_loader:
    images = images.cuda()
    labels = labels.cuda()
    
    # Apply augmentation on GPU
    images = augment(images)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Running Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Performance benchmark
python examples/benchmark.py

# Visualize augmentations (requires matplotlib)
pip install matplotlib pillow
python examples/visualize_augmentations.py
```

## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=triton_augment
```

## Key Features

### 🚀 Fastest Approach

Use `TritonColorJitterNormalize` for the best performance - it fuses all operations into a single GPU kernel:

```python
transform = ta.TritonColorJitterNormalize(...)  # ✓ Fastest!
```

### 🎯 Custom Ranges

Specify custom parameter ranges:

```python
transform = ta.TritonColorJitter(
    brightness=(-0.3, 0.3),  # Custom range
    contrast=(0.7, 1.3),      # Custom range
    saturation=(0.5, 1.5)     # Custom range
)
```

### 🔧 Functional API

For more control, use the functional API:

```python
import triton_augment.functional as F

# Apply operations directly
result = F.fused_color_normalize(
    images,
    brightness_factor=0.1,
    contrast_factor=1.2,
    saturation_factor=0.8,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

## Performance Tips

1. **Use the fused transform**: `TritonColorJitterNormalize` is 3-4x faster than sequential operations
2. **Keep data on GPU**: Avoid moving data between CPU and GPU
3. **Batch your operations**: Larger batches can better utilize GPU parallelism
4. **Use appropriate dtypes**: `float16` can be faster but may have precision trade-offs

## Common Issues

### CUDA Out of Memory

If you get OOM errors:
- Reduce batch size
- Use smaller image sizes
- Use `float16` instead of `float32`

### Import Error

If you get import errors:
```bash
# Make sure triton-augment is installed
pip install -e .

# Verify installation
python -c "import triton_augment; print(triton_augment.__version__)"
```

### Performance Not as Expected

- Make sure you're running on a CUDA GPU
- Run the benchmark script to compare: `python examples/benchmark.py`
- Ensure CUDA drivers are up to date

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [examples/](examples/) for more usage patterns
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Review the [API Reference](README.md#-api-reference)

## Need Help?

- Open an issue on GitHub
- Check existing issues for solutions
- Review the examples in `examples/` directory

---

Happy augmenting! 🎉


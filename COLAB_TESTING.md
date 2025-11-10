# Testing Triton-Augment on Google Colab

Since Triton requires Linux + CUDA, you can test on Google Colab's free GPU.

## Quick Start (Copy-Paste into Colab)

### 1. Open Google Colab
Go to: https://colab.research.google.com/

### 2. Enable GPU
- Click `Runtime` → `Change runtime type`
- Select `Hardware accelerator`: **GPU** (T4 is free tier)
- Click `Save`

### 3. Run This Code

```python
# Cell 1: Check GPU
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ No GPU! Enable it: Runtime → Change runtime type → GPU")
```

```python
# Cell 2: Install Triton-Augment
!git clone https://github.com/yuhezhang-ai/triton-augment.git
%cd triton-augment
!pip install -q -e ".[dev]"
print("\n✓ Installation complete!")
```

```python
# Cell 3: Quick Test
import triton_augment as ta
import torch

print(f"✓ Triton-Augment {ta.__version__}")

# Test basic function
img = torch.rand(2, 3, 64, 64, device='cuda')
result = ta.adjust_brightness(img, 1.2)
print(f"✓ Basic test passed! Shape: {result.shape}")
```

```python
# Cell 4: Run Correctness Tests
!pytest tests/test_correctness.py -v --tb=short
```

```python
# Cell 5: Quick Benchmark
import time
import triton_augment as ta
import torchvision.transforms.v2.functional as tvF

# Setup
img = torch.rand(32, 3, 224, 224, device='cuda')
params = (1.2, 1.1, 0.9, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Warmup
for _ in range(10):
    _ = ta.fused_color_normalize(img, *params)
torch.cuda.synchronize()

# Benchmark torchvision
start = time.perf_counter()
for _ in range(100):
    result = tvF.adjust_brightness(img, 1.2)
    result = tvF.adjust_contrast(result, 1.1)
    result = tvF.adjust_saturation(result, 0.9)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    result = (result - mean_t) / std_t
torch.cuda.synchronize()
tv_time = (time.perf_counter() - start) / 100 * 1000

# Benchmark triton-augment
start = time.perf_counter()
for _ in range(100):
    result = ta.fused_color_normalize(img, *params)
torch.cuda.synchronize()
ta_time = (time.perf_counter() - start) / 100 * 1000

print(f"\n{'='*60}")
print(f"Benchmark: batch=32, size=224x224")
print(f"{'='*60}")
print(f"torchvision: {tv_time:.3f} ms")
print(f"triton-augment: {ta_time:.3f} ms")
print(f"{'='*60}")
print(f"🚀 Speedup: {tv_time/ta_time:.2f}x faster!")
print(f"{'='*60}")
```

```python
# Cell 6: Visual Test
import matplotlib.pyplot as plt

# Create test image
img = torch.zeros(1, 3, 224, 224, device='cuda')
img[0, 0, :112, :112] = torch.linspace(0, 1, 112, device='cuda').unsqueeze(0).repeat(112, 1)
img[0, 1, :112, 112:] = torch.linspace(0, 1, 112, device='cuda').unsqueeze(1).repeat(1, 112)
img[0, 2, 112:, :112] = torch.linspace(1, 0, 112, device='cuda').unsqueeze(0).repeat(112, 1)
img[0, :, 112:, 112:] = 0.5

# Apply augmentations
results = {
    'Original': img,
    'Bright +20%': ta.adjust_brightness(img, 1.2),
    'Contrast +50%': ta.adjust_contrast(img, 1.5),
    'Saturation -50%': ta.adjust_saturation(img, 0.5),
    'Grayscale': ta.adjust_saturation(img, 0.0),
    'Combined': ta.fused_color_normalize(img, 1.2, 1.1, 0.8, None, None),
}

# Plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for idx, (name, img_t) in enumerate(results.items()):
    ax = axes[idx // 3, idx % 3]
    img_np = img_t[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    ax.imshow(img_np)
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.show()
print("✓ Visualizations complete!")
```

## Expected Results

✅ **Correctness Tests**: All tests should pass with pixel-perfect accuracy  
✅ **Benchmark**: Should show 2-5x speedup vs torchvision  
✅ **Visual**: Augmentations should look correct  

## Troubleshooting

**No GPU available?**
- Make sure you selected GPU in runtime settings
- Free tier has limited GPU hours (~12h/day)

**Installation fails?**
- Colab should have CUDA pre-installed
- Try: `!nvidia-smi` to check GPU

**Tests fail?**
- Check if it's a correctness issue or GPU OOM
- Reduce batch size if OOM

## Next Steps

Once tests pass on Colab:
1. ✅ Code is verified to work correctly
2. ✅ Performance benefits confirmed
3. ✅ Ready for production use!

For local GPU testing, you'll need:
- Linux OS
- NVIDIA GPU with CUDA
- Triton installed

---

**Author**: yuhezhang-ai  
**Last Updated**: November 10, 2025


# Bug Fixes - Clamping and Test Tolerances

## Issue
Tests were failing because:
1. **Missing clamping**: Triton kernels didn't clamp output values to [0, 1] range like torchvision does
2. **Custom tolerances**: Tests used custom `rtol=1e-4, atol=1e-4` instead of PyTorch defaults

## Root Cause
Looking at torchvision source (`transforms/v2/functional/_color.py`):
- Line 96: `_blend()` uses `.clamp_(0, bound)` 
- Line 124: `adjust_brightness_image()` uses `.clamp_(0, bound)`

**Torchvision ALWAYS clamps color adjustment outputs to [0, 1]**

## Changes Made

### 1. Updated Test Tolerances (`tests/test_correctness.py`)
Changed from custom tolerances to PyTorch defaults:
```python
# Before
torch.testing.assert_close(ta_result, tv_result, rtol=1e-4, atol=1e-4)

# After (uses default rtol=1e-05, atol=1e-08)
torch.testing.assert_close(ta_result, tv_result)
```

### 2. Added Clamping to All Kernels (`triton_augment/kernels/color_normalize_kernel.py`)

Added `tl.maximum(0.0, tl.minimum(1.0, value))` after:
- ✅ Brightness adjustment (fused + individual kernel)
- ✅ Contrast adjustment (fused + individual kernel)
- ✅ Saturation adjustment (fused + individual kernel)

Example:
```python
# Brightness adjustment
r = r * brightness_factor
g = g * brightness_factor
b = b * brightness_factor
# Clamp to [0, 1] as torchvision does
r = tl.maximum(0.0, tl.minimum(1.0, r))
g = tl.maximum(0.0, tl.minimum(1.0, g))
b = tl.maximum(0.0, tl.minimum(1.0, b))
```

## Test Results Expected
All 49 tests should now pass:
- ✅ Brightness: 0.0, 0.5, 1.0, 1.5, 2.0 (was failing for 1.5, 2.0)
- ✅ Contrast: 0.0, 0.5, 1.0, 1.5, 2.0 (was failing for 1.5, 2.0)
- ✅ Saturation: 0.0, 0.5, 1.0, 1.5, 2.0 (was failing for 1.5, 2.0)
- ✅ Fused operations: All batch sizes and image sizes

## Verification Command
```bash
pytest tests/test_correctness.py -v
```


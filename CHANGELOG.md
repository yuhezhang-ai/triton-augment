# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Nov 29, 2025

### Added

#### Affine Transformations Support

- **Affine Transformations**: Full support for rotation, translation, scaling, and shearing in the fused kernel
- **`TritonRandomAffine` Class**: torchvision-compatible random affine transform with per-image parameters
- **`TritonRandomRotation` Class**: torchvision-compatible random rotation transform
- **`InterpolationMode` Enum**: Consistent interpolation modes (nearest, bilinear) matching torchvision
- **Enhanced `TritonFusedAugment`**: Now supports affine parameters alongside crop/flip/color operations
- **Functional API**: `affine()` and `rotate()` functions with torchvision-exact behavior

#### Kernel Improvements

- **Unified Geometric Pipeline**: Refactored fused kernel to apply coordinate transforms in correct order: Flip → Crop → Affine, which is equivalent to Affine → Crop → Flip in image.
- **Enhanced Sampling**: Better nearest neighbor and bilinear interpolation accuracy

#### Testing & Quality

- **Affine Correctness Tests**: Comprehensive tests comparing with torchvision for rotation, translation, scaling, shearing
- **Batch Processing Tests**: Tests for per-image parameter handling across different batch sizes
- **Video Tensor Tests**: 5D tensor correctness validation for affine transforms
- **Interpolation Mode Tests**: Separate validation for nearest and bilinear sampling

### Performance

- **Updated Benchmarks**: Improved performance results with 8.0x average speedup on Tesla T4 (up to 15.6x on large images)
- **Kernel Fusion Improvements**: Better memory coalescing and reduced register pressure

### Changed

- **API Updates**: `TritonFusedAugment` constructor now accepts `degrees`, `translate`, `scale`, `shear` parameters
- **Default Interpolation**: Changed default interpolation to `"nearest"` to match torchvision v2 behavior
- **Color Helper Refactoring**: `TritonFusedAugment` now uses `TritonColorJitterNormalize` as internal helper for cleaner code

### Technical

- **Kernel Architecture**: Split geometric processing into sequential steps instead of matrix composition
- **Memory Layout**: Optimized parameter tensor layouts for better GPU memory access patterns

## [0.2.0] - Nov 18, 2025

### Added

- **5D Video Tensor Support**: Native support for `[N, T, C, H, W]` video tensors with `same_on_frame` parameter for controlling augmentation consistency across frames
- **`same_on_frame` Parameter**: Control whether augmentation parameters are shared across frames in video tensors (default: `True` for consistent augmentation)
- Video benchmark script (`examples/benchmark_video.py`) comparing performance vs Torchvision and Kornia

### Changed

- **Normalization defaults**: `mean` and `std` parameters now default to `None` in `TritonColorJitterNormalize` and `TritonFusedAugment` (normalization is now optional)
- Updated documentation to include 5D tensor support and `same_on_frame` parameter usage

### Performance

- Video augmentation benchmarks: **8.6x average speedup vs Torchvision, 73.7x vs Kornia** on Tesla T4

## [0.1.0] - Nov 14, 2025

### Added

#### Core Features

- **Ultimate Fused Kernel**: All 7 operations in a single GPU kernel (crop, flip, brightness, contrast, saturation, grayscale, normalize)
- **Per-Image Randomness**: Each image in batch gets different random parameters by default (`same_on_batch` flag for control)
- **Auto-Tuning**: Optional GPU-specific performance optimization with 12 configurations
- **Float16 Support**: ~1.3x speedup on large images (1024×1024+) with half-precision

#### Transform Classes

- `TritonFusedAugment`: Ultimate fusion - all operations in 1 kernel
- `TritonColorJitterNormalize`: Color operations + normalize (fused)
- `TritonRandomCropFlip`: Geometric operations (fused)
- `TritonColorJitter`: Fast contrast mode
- `TritonRandomCrop`, `TritonRandomHorizontalFlip`, `TritonRandomGrayscale`: Individual operations
- `TritonNormalize`, `TritonGrayscale`, `TritonCenterCrop`: Utility transforms

#### Functional API

- `fused_augment()`: Ultimate fusion function
- Geometric: `crop()`, `horizontal_flip()`
- Color: `adjust_brightness()`, `adjust_contrast()`, `adjust_contrast_fast()`, `adjust_saturation()`, `rgb_to_grayscale()`
- Utility: `normalize()`

#### User Experience

- Automatic 3D `(C, H, W)` to 4D `(N, C, H, W)` tensor handling
- Automatic CPU to GPU transfer
- torchvision-like API for easy migration
- Input validation and helpful error messages

#### Documentation & Testing

- Comprehensive README with benchmark results (8.1x average speedup on T4, up to 12x on large images)
- Full API reference documentation
- User guides: Float16, Auto-Tuning, Batch Behavior, Contrast Implementation
- 200+ unit tests covering correctness, float16, per-image randomness
- Training examples (MNIST, CIFAR-10)
- Benchmark scripts with visualization

### Performance

- **8.1x average speedup** vs torchvision (Tesla T4, Google Colab Free Tier)
- Scales with image size: 4.5x (256×256) → 12.0x (1280×1280)
- 4.1x average on A100 (high bandwidth makes relative improvement smaller)
- Zero intermediate memory allocations through kernel fusion
- Optimized for large batches and large images

### Technical Highlights

- Single unified fused kernel with compile-time operation flags
- Per-image parameter arrays for random augmentations
- Triton auto-tuning with extended configuration space (12 configs)
- Correct grayscale ordering (saturation → clamp → grayscale)
- Torchvision-exact correctness validation
- Float16 precision handling and testing

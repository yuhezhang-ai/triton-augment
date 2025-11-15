# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Extended geometric operations (rotation, affine transforms)
- Additional augmentations (Gaussian blur, random erasing, mixup)
- Hue adjustment
- Multi-GPU support

## [0.1.0] - TBD

### Added

#### Core Features
- **Ultimate Fused Kernel**: All 7 operations in a single GPU kernel (crop, flip, brightness, contrast, saturation, grayscale, normalize)
- **Per-Image Randomness**: Each image in batch gets different random parameters by default (`same_on_batch` flag for control)
- **Auto-Tuning**: Optional GPU-specific performance optimization with 12 configurations
- **Float16 Support**: Additional 1.3-2x speedup with half-precision

#### Transform Classes
- `TritonFusedAugment`: Ultimate fusion - all operations in 1 kernel
- `TritonColorJitterNormalize`: Color operations + normalize (fused)
- `TritonRandomCropFlip`: Geometric operations (fused)
- `TritonColorJitter`: Fast contrast mode (sequential)
- `TritonRandomCrop`, `TritonRandomHorizontalFlip`, `TritonRandomGrayscale`: Individual operations
- `TritonNormalize`, `TritonGrayscale`, `TritonCenterCrop`: Utility transforms

#### Functional API
- `fused_augment()`: Ultimate fusion function
- Geometric: `crop()`, `horizontal_flip()`
- Color: `adjust_brightness()`, `adjust_contrast()`, `adjust_saturation()`, `rgb_to_grayscale()`
- Utility: `normalize()`

#### User Experience
- Automatic 3D `(C, H, W)` to 4D `(N, C, H, W)` tensor handling
- Automatic CPU to GPU transfer
- torchvision-like API for easy migration
- Input validation and helpful error messages

#### Documentation & Testing
- Comprehensive README with benchmark results (4x average speedup, up to 8.7x on large images)
- Full API reference documentation
- User guides: Float16, Auto-Tuning, Batch Behavior, Contrast Implementation
- 200+ unit tests covering correctness, float16, per-image randomness
- Training examples (MNIST, CIFAR-10)
- Benchmark scripts with visualization

### Performance
- **4.1x average speedup** vs torchvision (A100, Google Colab)
- Scales with image size: 1.4x (256×256) → 8.7x (1280×1280)
- Zero intermediate memory allocations through kernel fusion
- Optimized for large batches and large images

### Technical Highlights
- Single unified fused kernel with compile-time operation flags
- Per-image parameter arrays for random augmentations
- Triton auto-tuning with extended configuration space (12 configs)
- Correct grayscale ordering (saturation → clamp → grayscale)
- Torchvision-exact correctness validation
- Float16 precision handling and testing

[Unreleased]: https://github.com/yuhezhang-ai/triton-augment/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yuhezhang-ai/triton-augment/releases/tag/v0.1.0


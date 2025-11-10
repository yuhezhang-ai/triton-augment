# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Fused random crop (Phase 2)
- Fused random flip (Phase 2)
- Gaussian blur
- Random erasing
- Hue adjustment

## [0.1.0] - 2025-11-10

### Added
- Initial release of Triton-Augment
- Core fused kernel for color jitter and normalization operations
- `TritonColorJitter` transform class
- `TritonNormalize` transform class
- `TritonColorJitterNormalize` combined transform class
- Functional API with the following operations:
  - `apply_brightness`: Brightness adjustment
  - `apply_contrast`: Contrast adjustment
  - `apply_normalize`: Per-channel normalization
  - `fused_color_jitter`: Fused color jitter operations
  - `fused_color_normalize`: Fully fused color jitter + normalization
- Comprehensive documentation and README
- Usage examples in `examples/` directory
- Benchmark script for performance comparison
- Unit tests with pytest
- Package setup with `setup.py` and requirements files

### Features
- 2-5x speedup over sequential PyTorch operations
- Zero intermediate memory allocations through kernel fusion
- Support for RGB images in NCHW format
- Support for float32 and float16 dtypes
- Compatible with PyTorch DataLoader pipelines
- Familiar torchvision-like API

### Technical Details
- Implements two versions of the fused kernel:
  - `fused_color_normalize_kernel`: Element-wise processing
  - `fused_color_normalize_kernel_v2`: Optimized with proper RGB handling
- Proper RGB to grayscale conversion for saturation adjustment
- Per-channel normalization with configurable mean and std
- Compile-time flags for enabling/disabling operations
- Block size optimization for different tensor sizes

[Unreleased]: https://github.com/yourusername/triton-augment/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/triton-augment/releases/tag/v0.1.0


# Triton-Augment Project Structure

This document provides an overview of the project structure and the purpose of each file and directory.

## Directory Tree

```
triton-augment/
├── triton_augment/              # Main package directory
│   ├── __init__.py             # Package initialization, exports public API
│   ├── functional.py           # Functional API (PyTorch-style functions)
│   ├── transforms.py           # Transform classes (nn.Module-based)
│   └── kernels/                # Triton kernel implementations
│       ├── __init__.py         # Kernel package initialization
│       └── color_normalize_kernel.py  # Fused color jitter & normalize kernels
│
├── examples/                    # Usage examples and demonstrations
│   ├── basic_usage.py          # Basic usage examples
│   ├── benchmark.py            # Performance benchmarking script
│   └── visualize_augmentations.py  # Visualization of augmentation effects
│
├── tests/                       # Unit tests and integration tests
│   ├── __init__.py             # Test package initialization
│   └── test_transforms.py      # Tests for transforms and functional API
│
├── docs/                        # Documentation (future)
│
├── README.md                    # Main project documentation
├── QUICKSTART.md               # Quick start guide
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md                # Version history and changes
├── LICENSE                      # MIT License
├── PROJECT_STRUCTURE.md        # This file
│
├── setup.py                     # Package setup configuration (legacy)
├── pyproject.toml              # Modern Python project configuration
├── MANIFEST.in                 # Package distribution manifest
│
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Development dependencies
│
└── .gitignore                  # Git ignore patterns
```

## File Descriptions

### Core Package (`triton_augment/`)

#### `__init__.py`
- Package entry point
- Exports public API (transforms and functional operations)
- Defines package version
- Provides convenient imports for end users

#### `functional.py`
- Functional API similar to `torchvision.transforms.functional`
- Wraps Triton kernels with PyTorch tensor operations
- Functions:
  - `apply_brightness()`: Brightness adjustment
  - `apply_contrast()`: Contrast adjustment
  - `apply_normalize()`: Per-channel normalization
  - `fused_color_jitter()`: Fused color jitter operations
  - `fused_color_normalize()`: Fully fused color jitter + normalization
- Input validation and error handling
- Grid size calculation for Triton kernel launches

#### `transforms.py`
- Transform classes similar to `torchvision.transforms`
- Stateful transforms for use in data pipelines
- Classes:
  - `TritonColorJitter`: Random color jitter
  - `TritonNormalize`: Per-channel normalization
  - `TritonColorJitterNormalize`: Combined transform (recommended)
- Random parameter sampling
- Inherits from `torch.nn.Module` for compatibility

#### `kernels/color_normalize_kernel.py`
- Raw Triton JIT-compiled kernels
- Two main kernel implementations:
  - `fused_color_normalize_kernel`: Element-wise processing
  - `fused_color_normalize_kernel_v2`: Optimized RGB processing (used in production)
- Individual operation kernels:
  - `brightness_kernel`: Standalone brightness adjustment
  - `contrast_kernel`: Standalone contrast adjustment
  - `normalize_kernel`: Standalone normalization
- Optimized for memory coalescence and GPU parallelism

### Examples (`examples/`)

#### `basic_usage.py`
- Comprehensive usage examples
- 6 different scenarios:
  1. Basic color jitter
  2. Normalization only
  3. Fused transform (recommended)
  4. Functional API
  5. Training pipeline integration
  6. Custom parameter ranges
- Demonstrates best practices
- Can be run directly to verify installation

#### `benchmark.py`
- Performance comparison with PyTorch sequential operations
- Supports multiple batch sizes and image dimensions
- Includes warmup runs for accurate measurements
- Generates performance summary table
- Usage: `python examples/benchmark.py --help`

#### `visualize_augmentations.py`
- Creates visual comparisons of augmentation effects
- Generates comparison images showing:
  - Different brightness levels
  - Different contrast levels
  - Different saturation levels
  - Combined effects
  - Random augmentation samples
- Requires matplotlib and pillow
- Useful for understanding parameter effects

### Tests (`tests/`)

#### `test_transforms.py`
- Comprehensive unit tests using pytest
- Test categories:
  - Functional API tests
  - Transform class tests
  - Batch size tests
  - Data type tests
  - Edge case tests
- Automatically skipped if CUDA is not available
- Run with: `pytest tests/`

### Documentation

#### `README.md`
- Main project documentation
- Installation instructions
- Quick start guide
- Complete API reference
- Performance benchmarks
- Usage examples
- Roadmap

#### `QUICKSTART.md`
- Condensed quick start guide
- 5-minute getting started tutorial
- Common usage patterns
- Troubleshooting tips

#### `CONTRIBUTING.md`
- Contribution guidelines
- Development setup instructions
- Coding standards
- Testing requirements
- Pull request process

#### `CHANGELOG.md`
- Version history
- Release notes
- Breaking changes
- New features per version

#### `PROJECT_STRUCTURE.md`
- This file
- Project organization overview
- File purpose descriptions

### Configuration Files

#### `setup.py`
- Legacy package setup configuration
- Package metadata
- Dependencies
- Entry points
- Compatible with older pip versions

#### `pyproject.toml`
- Modern Python project configuration (PEP 518)
- Package metadata and dependencies
- Tool configurations:
  - Black (code formatter)
  - isort (import sorter)
  - mypy (type checker)
  - pytest (test framework)
- Preferred over setup.py for new projects

#### `MANIFEST.in`
- Specifies which files to include in distribution
- Ensures examples and docs are packaged
- Excludes test files and cache

#### `requirements.txt`
- Core runtime dependencies
- Minimal set of packages needed to use the library
- Used for basic installation

#### `requirements-dev.txt`
- Development dependencies
- Includes testing, benchmarking, and code quality tools
- Used for contributors and developers

#### `.gitignore`
- Specifies files to exclude from version control
- Python artifacts, caches, compiled files
- IDE configurations
- Data files and checkpoints
- Triton cache directories

## Architecture Overview

### Kernel Fusion Strategy

```
Traditional Approach (PyTorch):
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐
│ Input   │ --> │ Bright   │ --> │ Contrast │ --> │ Normalize │
│ Image   │     │ (GPU)    │     │ (GPU)    │     │ (GPU)     │
└─────────┘     └──────────┘     └──────────┘     └───────────┘
    ^               v                v                 v
    └───────────────┴────────────────┴─────────────────┘
         Multiple DRAM reads/writes (SLOW)

Triton-Augment Approach:
┌─────────┐     ┌──────────────────────────────────┐
│ Input   │ --> │ Fused Kernel:                    │
│ Image   │     │ - Brightness                     │
└─────────┘     │ - Contrast                       │
                │ - Saturation                     │
                │ - Normalize                      │
                └──────────────────────────────────┘
    ^                        v
    └────────────────────────┘
      Single DRAM round-trip (FAST)
```

### Module Dependencies

```
User Code
    │
    ├── triton_augment.transforms.*
    │       │
    │       └── triton_augment.functional.*
    │               │
    │               └── triton_augment.kernels.*
    │                       │
    │                       └── Triton JIT Compiler
    │                               │
    │                               └── CUDA GPU
    │
    └── triton_augment.functional.* (direct use)
            │
            └── ... (same as above)
```

### Data Flow

1. **User creates transform** → `TritonColorJitterNormalize(...)`
2. **User calls transform** → `transform(images)`
3. **Random params sampled** → `_get_params()`
4. **Functional API called** → `functional.fused_color_normalize(...)`
5. **Input validated** → `_validate_image_tensor(...)`
6. **Grid calculated** → `triton.cdiv(...)`
7. **Kernel launched** → `fused_color_normalize_kernel_v2[grid](...)`
8. **GPU executes** → Parallel processing across image
9. **Result returned** → Output tensor

## Design Principles

1. **Familiarity**: API mirrors torchvision for easy adoption
2. **Performance**: Kernel fusion minimizes memory bandwidth
3. **Flexibility**: Both functional and transform APIs available
4. **Correctness**: Comprehensive tests ensure accuracy
5. **Documentation**: Extensive docs and examples
6. **Modularity**: Clear separation between kernels, functional, and transforms
7. **Compatibility**: Works with existing PyTorch workflows

## Future Extensions

As per the roadmap, future additions will include:

1. **Phase 2: Geometric Transforms**
   - `triton_augment/kernels/geometric_kernel.py`
   - `TritonRandomCrop` in transforms.py
   - `fused_geom_color_normalize` in functional.py

2. **Phase 3: Extended Operations**
   - `triton_augment/kernels/blur_kernel.py`
   - `triton_augment/kernels/erasing_kernel.py`
   - Additional transform classes

3. **Documentation**
   - `docs/` directory with Sphinx documentation
   - API reference
   - Tutorials and guides

## Quick Reference

| Need to...                        | Look in...                  |
|-----------------------------------|----------------------------|
| Use the library                   | `triton_augment/__init__.py`, `README.md` |
| Understand the API                | `README.md`, `QUICKSTART.md` |
| See examples                      | `examples/`                |
| Run benchmarks                    | `examples/benchmark.py`    |
| Modify kernels                    | `triton_augment/kernels/`  |
| Add new transforms                | `triton_augment/transforms.py` |
| Add functional operations         | `triton_augment/functional.py` |
| Write tests                       | `tests/`                   |
| Contribute                        | `CONTRIBUTING.md`          |
| Install                           | `setup.py`, `pyproject.toml` |

---

**Last Updated**: November 10, 2025
**Version**: 0.1.0


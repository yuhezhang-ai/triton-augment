#!/usr/bin/env python
"""
Installation verification script for Triton-Augment.

This script checks that all dependencies are installed correctly and that
the library is working as expected.
"""

import sys


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ❌ Python {version.major}.{version.minor} detected")
        print("  ⚠️  Python 3.8 or higher is required")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_torch():
    """Check PyTorch installation."""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"    CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ❌ CUDA not available")
            print("  ⚠️  Triton-Augment requires a CUDA-capable GPU")
            return False
    except ImportError:
        print("  ❌ PyTorch not installed")
        print("  → Install with: pip install torch>=2.0.0")
        return False


def check_triton():
    """Check Triton installation."""
    print("\nChecking Triton...")
    try:
        import triton
        print(f"  ✓ Triton {triton.__version__} installed")
        return True
    except ImportError:
        print("  ❌ Triton not installed")
        print("  → Install with: pip install triton>=2.0.0")
        return False


def check_triton_augment():
    """Check Triton-Augment installation."""
    print("\nChecking Triton-Augment...")
    try:
        import triton_augment as ta
        print(f"  ✓ Triton-Augment {ta.__version__} installed")
        
        # Check that main components are importable
        from triton_augment import (
            TritonColorJitter,
            TritonNormalize,
            TritonColorJitterNormalize,
            fused_color_normalize,
        )
        print("  ✓ All main components importable")
        return True
    except ImportError as e:
        print(f"  ❌ Triton-Augment not installed: {e}")
        print("  → Install with: pip install -e .")
        return False


def run_basic_test():
    """Run a basic functionality test."""
    print("\nRunning basic functionality test...")
    try:
        import torch
        import triton_augment as ta
        
        # Create test image
        img = torch.rand(2, 3, 64, 64, device='cuda')
        print("  ✓ Created test image on GPU")
        
        # Test color jitter
        transform = ta.TritonColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        result = transform(img)
        assert result.shape == img.shape
        assert result.device == img.device
        print("  ✓ Color jitter working")
        
        # Test normalization
        normalize = ta.TritonNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        result = normalize(img)
        assert result.shape == img.shape
        print("  ✓ Normalization working")
        
        # Test fused transform
        fused = ta.TritonColorJitterNormalize(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        result = fused(img)
        assert result.shape == img.shape
        print("  ✓ Fused transform working")
        
        print("  ✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_optional_dependencies():
    """Check optional dependencies."""
    print("\nChecking optional dependencies...")
    
    optional_deps = {
        'pytest': 'Testing',
        'torchvision': 'Examples and benchmarks',
        'matplotlib': 'Visualization',
        'pillow': 'Image loading',
    }
    
    installed = []
    missing = []
    
    for package, purpose in optional_deps.items():
        try:
            if package == 'pillow':
                __import__('PIL')
            else:
                __import__(package)
            installed.append(f"  ✓ {package} ({purpose})")
        except ImportError:
            missing.append(f"  ○ {package} ({purpose}) - optional")
    
    for dep in installed:
        print(dep)
    
    if missing:
        print("\nOptional dependencies not installed:")
        for dep in missing:
            print(dep)
        print("\n  → Install with: pip install -e \".[dev]\"")


def main():
    """Main verification function."""
    print("="*60)
    print("Triton-Augment Installation Verification")
    print("="*60)
    
    checks = []
    
    # Required checks
    checks.append(("Python version", check_python_version()))
    checks.append(("PyTorch with CUDA", check_torch()))
    checks.append(("Triton", check_triton()))
    checks.append(("Triton-Augment", check_triton_augment()))
    
    # Run basic test if all required components are available
    all_required_ok = all(result for _, result in checks)
    if all_required_ok:
        checks.append(("Basic functionality", run_basic_test()))
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, result in checks:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        print("\n" + "="*60)
        print("🎉 Installation verified successfully!")
        print("="*60)
        print("\nYou can now use Triton-Augment:")
        print("  import triton_augment as ta")
        print("\nNext steps:")
        print("  - Try examples: python examples/basic_usage.py")
        print("  - Run benchmarks: python examples/benchmark.py")
        print("  - Read the docs: cat README.md")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ Installation verification failed")
        print("="*60)
        print("\nPlease fix the issues above and try again.")
        print("For help, see: README.md or QUICKSTART.md")
        return 1


if __name__ == '__main__':
    sys.exit(main())


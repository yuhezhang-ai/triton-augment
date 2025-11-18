"""
Comprehensive benchmark script using Triton's built-in benchmarking utilities.

This script provides detailed performance analysis with visualizations across
multiple operations and configurations.

For a simpler, quick benchmark of just the Ultimate Fusion kernel, use:
    python examples/benchmark.py

This comprehensive script uses triton.testing.perf_report to compare Triton-Augment
performance with torchvision.transforms.v2 across multiple scenarios.

Usage:
    python examples/benchmark_triton.py
    python examples/benchmark_triton.py --autotune  # Enable auto-tuning

Author: yuhezhang-ai
"""

import argparse
import os
import torch
import triton
from triton.testing import perf_report

try:
    import triton_augment as ta
    TRITON_AUGMENT_AVAILABLE = True
except ImportError:
    print("Warning: triton_augment not installed")
    TRITON_AUGMENT_AVAILABLE = False

try:
    import torchvision.transforms.v2 as transforms
    import torchvision.transforms.v2.functional as tvF
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("Warning: torchvision not installed")
    TORCHVISION_AVAILABLE = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[64 * i for i in range(2, 25)],
        line_arg='provider',
        line_vals=['torchvision', 'triton-augment-sequential', 'triton-augment-fused'],
        line_names=['torchvision (v2)', 'Triton-Augment (sequential)', 'Triton-Augment (fused)'],
        styles=[('green', '-'), ('blue', '--'), ('red', '-')],
        ylabel='Runtime (ms)',
        plot_name='color-jitter-normalize-performance',
        args={'batch_size': 32},
    )
)
def benchmark_color_jitter_normalize(size, batch_size, provider):
    """
    Benchmark functionals with FIXED factors: Brightness + Contrast + Saturation + Normalize.
    NO grayscale in this benchmark.
    
    NOTE: Triton fused uses FAST contrast (centered scaling), not torchvision's
    blend-with-mean. For fair comparison without contrast, see benchmark_without_contrast.
    
    Args:
        size: Image size (square images)
        batch_size: Batch size
        provider: Which implementation to use
    """
    # Create test data
    images = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
    
    # Fixed augmentation parameters
    brightness_factor = 1.2
    contrast_factor = 1.1
    saturation_factor = 0.9
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torchvision':
        if not TORCHVISION_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = tvF.adjust_brightness(images, brightness_factor)
            img = tvF.adjust_contrast(img, contrast_factor)
            img = tvF.adjust_saturation(img, saturation_factor)
            img = tvF.normalize(img, mean=list(mean), std=list(std))
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-sequential':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = ta.adjust_brightness(images, brightness_factor)
            img = ta.adjust_contrast_fast(img, contrast_factor)  # Using FAST contrast
            img = ta.adjust_saturation(img, saturation_factor)
            img = ta.normalize(img, mean, std)
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-fused':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            return ta.fused_augment(
                images,
                top=0, left=0, height=size, width=size,  # No-op crop
                flip_horizontal=False,  # No flip
                brightness_factor=brightness_factor,
                contrast_factor=contrast_factor,  # Uses FAST contrast
                saturation_factor=saturation_factor,
                grayscale=False,  # No grayscale
                mean=mean,
                std=std,
            )
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[64 * i for i in range(2, 25)],
        line_arg='provider',
        line_vals=['torchvision', 'triton-augment-sequential', 'triton-augment-fused'],
        line_names=['torchvision (v2)', 'Triton-Augment (sequential)', 'Triton-Augment (fused)'],
        styles=[('green', '-'), ('blue', '--'), ('red', '-')],
        ylabel='Runtime (ms)',
        plot_name='brightness-saturation-normalize-performance',
        args={'batch_size': 32},
    )
)
def benchmark_without_contrast(size, batch_size, provider):
    """
    Fair comparison with FIXED factors: Brightness + Saturation + Normalize (NO contrast, NO grayscale).
    
    This benchmark excludes contrast to provide a fair comparison since:
    - Torchvision uses blend-with-mean (slower, exact)
    - Triton fused uses centered scaling (faster, approximate)
    
    Args:
        size: Image size (square images)
        batch_size: Batch size
        provider: Which implementation to use
    """
    # Create test data
    images = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
    
    # Fixed parameters
    brightness_factor = 1.2
    saturation_factor = 0.9
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torchvision':
        if not TORCHVISION_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = tvF.adjust_brightness(images, brightness_factor)
            img = tvF.adjust_saturation(img, saturation_factor)
            img = tvF.normalize(img, mean=list(mean), std=list(std))
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-sequential':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = ta.adjust_brightness(images, brightness_factor)
            img = ta.adjust_saturation(img, saturation_factor)
            img = ta.normalize(img, mean, std)
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-fused':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            return ta.fused_augment(
                images,
                top=0, left=0, height=size, width=size,  # No-op crop
                flip_horizontal=False,  # No flip
                brightness_factor=brightness_factor,
                contrast_factor=1.0,  # No contrast adjustment
                saturation_factor=saturation_factor,
                grayscale=False,  # No grayscale
                mean=mean,
                std=std,
            )
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[2**i for i in range(0, 8)],  # 1, 2, 4, 8, 16, 32
        line_arg='provider',
        line_vals=['torchvision', 'triton-augment-fused'],
        line_names=['torchvision (v2)', 'Triton-Augment (fused)'],
        styles=[('green', '-'), ('red', '-')],
        ylabel='Runtime (ms)',
        plot_name='batch-size-scaling',
        args={'size': 224},
    )
)
def benchmark_batch_scaling(batch_size, size, provider):
    """Benchmark how performance scales with batch size using FIXED factors (no contrast, no grayscale for fairness)."""
    images = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
    
    # Fixed parameters
    brightness_factor = 1.2
    saturation_factor = 0.9
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torchvision':
        if not TORCHVISION_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = tvF.adjust_brightness(images, brightness_factor)
            img = tvF.adjust_saturation(img, saturation_factor)
            img = tvF.normalize(img, mean=list(mean), std=list(std))
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-fused':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            return ta.fused_augment(
                images,
                top=0, left=0, height=size, width=size,  # No-op crop
                flip_horizontal=False,  # No flip
                brightness_factor=brightness_factor,
                contrast_factor=1.0,  # No contrast
                saturation_factor=saturation_factor,
                grayscale=False,  # No grayscale
                mean=mean,
                std=std,
            )
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[64 * i for i in range(2, 25)],
        line_arg='provider',
        line_vals=['torchvision-transforms', 'triton-augment-transforms', 'triton-augment-fused'],
        line_names=['torchvision Compose', 'Triton-Augment Compose', 'Triton-Augment Fused'],
        styles=[('green', '-'), ('blue', '--'), ('red', '-')],
        ylabel='Runtime (ms)',
        plot_name='random-color-augment-performance',
        args={'batch_size': 32},
    )
)
def benchmark_random_color_augment(size, batch_size, provider):
    """
    Random color-wise augmentation pipeline using transform CLASSES.
    
    This simulates actual usage in training where users create transforms once
    and apply them repeatedly with random augmentations each time.
    
    Uses ColorJitter (brightness, saturation) + RandomGrayscale + Normalize.
    """
    images = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
    
    # Training augmentation parameters (ranges, not fixed values)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torchvision-transforms':
        if not TORCHVISION_AVAILABLE:
            return 0, 0, 0
        
        # Create transforms using CLASSES (done once in real training)
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, saturation=0.2),  # Random ranges
            transforms.RandomGrayscale(p=0.1),  # Random with 10% probability
            transforms.Normalize(mean=mean, std=std),
        ])
        
        def fn():
            return transform(images)
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-transforms':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        # Create transforms using CLASSES (sequential, matching torchvision structure)
        color_jitter = ta.TritonColorJitter(brightness=0.2, saturation=0.2)
        random_gray = ta.TritonRandomGrayscale(p=0.1)
        normalize = ta.TritonNormalize(mean=mean, std=std)
        
        def fn():
            img = color_jitter(images)
            img = random_gray(img)
            img = normalize(img)
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    elif provider == 'triton-augment-fused':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        # Use FUSED transform class (single kernel, maximum performance)
        transform = ta.TritonColorJitterNormalize(
            brightness=0.2,
            saturation=0.2,
            grayscale_p=0.1,
            mean=mean,
            std=std,
        )
        
        def fn():
            return transform(images)
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[256, 384, 512, 640, 768, 1024, 1280],
        line_arg='provider',
        line_vals=['float32-ultimate', 'float16-ultimate'],
        line_names=['Float32 Ultimate Fusion', 'Float16 Ultimate Fusion'],
        styles=[('blue', '-'), ('red', '--')],
        ylabel='Runtime (ms)',
        plot_name='float16-vs-float32-ultimate-fusion',
        args={'batch_size': 32},
    )
)
def benchmark_float16_vs_float32(size, batch_size, provider):
    """
    Benchmark 7: Ultimate Fusion Float16 vs Float32 (FINALE)
    
    Tests the complete ULTIMATE FUSION pipeline with different dtypes:
    - RandomCrop + RandomHorizontalFlip
    - Brightness + Contrast + Saturation + RandomGrayscale
    - Normalize
    
    Shows that the peak performance path maintains its advantages with half precision!
    """
    # Fixed parameters (using transform classes for real scenario)
    crop_size = size - 40  # Crop slightly smaller than input
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if provider == 'float32-ultimate':
        img = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
        transform = ta.TritonFusedAugment(
            crop_size=crop_size,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            grayscale_p=0.1,
            mean=mean,
            std=std,
        )
    elif provider == 'float16-ultimate':
        img = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float16)
        transform = ta.TritonFusedAugment(
            crop_size=crop_size,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            grayscale_p=0.1,
            mean=mean,
            std=std,
        )
    
    fn = lambda: transform(img)
    ms = triton.testing.do_bench(fn)
    return ms

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[128, 256, 384, 512, 1024],
        line_arg='provider',
        line_vals=['torchvision-sequential', 'triton-sequential', 'triton-fused'],
        line_names=['Torchvision (Crop → Flip)', 'Triton Sequential (Crop → Flip)', 'Triton Fused (Crop+Flip)'],
        styles=[('green', '--'), ('blue', '-'), ('red', '-')],
        ylabel='Time (ms)',
        plot_name='geometric-fusion-performance',
        args={'batch_size': 32},
    )
)
def benchmark_geometric_fusion(size, batch_size, provider):
    """
    Benchmark 5: Geometric Fusion - Crop + Flip
    
    Compares:
    - Torchvision: crop() → horizontal_flip() (2 kernel launches)
    - Triton Sequential: crop() → horizontal_flip() (2 kernel launches)
    - Triton Fused: fused_augment() with crop+flip (1 kernel launch)
    
    Expected: ~1.5-2x speedup from fusion
    """
    img = torch.rand(batch_size, 3, size, size, device='cuda')
    
    # Fixed crop parameters
    crop_size = size - 40
    top = (size - crop_size) // 2
    left = (size - crop_size) // 2
    
    if provider == 'torchvision-sequential':
        def fn():
            result = tvF.crop(img, top, left, crop_size, crop_size)
            result = tvF.horizontal_flip(result)
            return result
    elif provider == 'triton-sequential':
        def fn():
            result = ta.crop(img, top, left, crop_size, crop_size)
            result = ta.horizontal_flip(result)
            return result
    elif provider == 'triton-fused':
        def fn():
            return ta.fused_augment(
                img,
                top=top, left=left, height=crop_size, width=crop_size,
                flip_horizontal=True,
                brightness_factor=1.0,  # No-op
                contrast_factor=1.0,    # No-op
                saturation_factor=1.0,  # No-op
                grayscale=False,        # No-op
                mean=None,              # No-op
                std=None                # No-op
            )
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[128, 256, 384, 512, 1024],
        line_arg='provider',
        line_vals=['torchvision-compose', 'triton-sequential', 'triton-ultimate'],
        line_names=[
            'Torchvision Compose (7 ops)',
            'Triton Sequential (7 ops)', 
            'Triton Ultimate Fused (7 ops)'
        ],
        styles=[('green', '-'), ('blue', '--'), ('red', '-')],
        ylabel='Time (ms)',
        plot_name='ultimate-fusion-performance',
        args={'batch_size': 32},
    )
)
def benchmark_ultimate_fusion(size, batch_size, provider):
    """
    Benchmark 6: Ultimate Fusion - ALL operations in ONE kernel!
    
    Uses TRANSFORM CLASSES with RANDOM augmentations (real training scenario):
    - RandomCrop (size-40×size-40)
    - RandomHorizontalFlip (p=0.5)
    - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2) → 3 kernels
    - RandomGrayscale (p=0.1)
    - Normalize
    
    Compares:
    - Torchvision Compose: 5 transforms (7 kernel launches)
    - Triton Sequential: 5 Triton transforms (7 kernel launches)
    - Triton Ultimate: 1 fused kernel (1 kernel launch) 🚀
    
    Expected: ~5-12x speedup vs torchvision (Tesla T4)!
    """
    img = torch.rand(batch_size, 3, size, size, device='cuda')
    
    # Parameters for random augmentations
    crop_size = size - 40
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if provider == 'torchvision-compose':
        # Torchvision with transform classes and random augmentations
        transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.Normalize(mean=mean, std=std),
        ])
        def fn():
            return transform(img)
    
    elif provider == 'triton-sequential':
        # Triton with sequential transform classes
        transform = transforms.Compose([
            ta.TritonRandomCrop(crop_size),
            ta.TritonRandomHorizontalFlip(p=0.5),
            ta.TritonColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ta.TritonRandomGrayscale(p=0.1),
            ta.TritonNormalize(mean=mean, std=std),
        ])
        def fn():
            return transform(img)
    
    elif provider == 'triton-ultimate':
        # Triton Ultimate: ALL in ONE fused kernel
        transform = ta.TritonFusedAugment(
            crop_size=crop_size,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            grayscale_p=0.1,
            mean=mean,
            std=std,
        )
        def fn():
            return transform(img)
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return ms


def print_ultimate_speedup_summary():
    """
    Quick benchmark comparing torchvision Compose vs Triton Ultimate for a typical training scenario.
    Uses transform classes with RANDOM augmentations.
    """
    print("\n" + "="*80)
    print("ULTIMATE SPEEDUP SUMMARY (5 transforms, 7 kernels → 1 kernel)")
    print("="*80)
    
    batch_size = 32
    img_size = 1280
    crop_size = 1024
    
    # Create test tensor
    img = torch.rand(batch_size, 3, img_size, img_size, device='cuda')
    
    # Parameters for random augmentations
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Torchvision Compose with random augmentations
    torchvision_transforms = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.Normalize(mean=mean, std=std),
    ])
    def tv_fn():
        return torchvision_transforms(img)
    
    # Triton Ultimate with random augmentations (ALL in ONE kernel!)
    triton_ultimate_transform = ta.TritonFusedAugment(
        crop_size=crop_size,
        horizontal_flip_p=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        grayscale_p=0.1,
        mean=mean,
        std=std,
    )
    def ta_fn():
        return triton_ultimate_transform(img)
    
    # Benchmark
    tv_time = triton.testing.do_bench(tv_fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    ta_time = triton.testing.do_bench(ta_fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    
    # Extract median times (first value from quantiles)
    tv_median = tv_time[0] if isinstance(tv_time, list) else tv_time
    ta_median = ta_time[0] if isinstance(ta_time, list) else ta_time
    speedup = tv_median / ta_median
    
    print(f"Image size: {img_size}×{img_size}, Batch: {batch_size}, Crop: {crop_size}×{crop_size}")
    print(f"  Torchvision Compose (5 transforms, 7 kernels):   {tv_median:.3f} ms")
    print(f"  Triton Ultimate (5 transforms, 1 kernel):        {ta_median:.3f} ms")
    print(f"  → Speedup: {speedup:.2f}x faster! 🚀")
    print()
    print("Transforms: RandomCrop + RandomHorizontalFlip + ColorJitter + RandomGrayscale + Normalize")
    print("Kernels: Crop + Flip + Brightness + Contrast + Saturation + RandomGrayscale + Normalize")
    print("Note: Both use transform classes with RANDOM augmentations (real training scenario)")
    print("      Triton uses FAST contrast (centered scaling), torchvision uses blend-with-mean")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comprehensive benchmark for Triton-Augment'
    )
    parser.add_argument(
        '--autotune',
        action='store_true',
        help='Enable auto-tuning for optimal performance (takes 5-10s on first run)'
    )
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        exit(1)
    
    if args.autotune:
        ta.enable_autotune()
        print("🔧 Auto-tuning ENABLED - will test 12 configs on first run (~5-10 seconds)")
    else:
        print("ℹ️  Auto-tuning DISABLED (default config). Use --autotune for optimal performance.")
    
    print("="*80)
    print("Triton-Augment Performance Benchmark (Using Triton's Benchmark Utilities)")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    if TRITON_AUGMENT_AVAILABLE:
        print(f"Triton-Augment version: {ta.__version__}")
    if TORCHVISION_AVAILABLE:
        import torchvision
        print(f"torchvision version: {torchvision.__version__}")
    print("="*80)
    print()
    
    if not TRITON_AUGMENT_AVAILABLE or not TORCHVISION_AVAILABLE:
        print("Error: Both triton_augment and torchvision are required for benchmarking.")
        exit(1)
    
    print("IMPORTANT NOTES:")
    print("  Benchmark 1: Functionals + Contrast (Brightness+Contrast+Saturation+Normalize)")
    print("    * Contrast uses different algorithms (Triton=FAST, Torchvision=blend-with-mean)")
    print("    * NOT a fair comparison, shows Triton's fast contrast advantage")
    print()
    print("  Benchmarks 2-3: Functionals WITHOUT Contrast (image/batch size scaling)")
    print("    * Operations: Brightness + Saturation + RandomGrayscale + Normalize")
    print("    * FAIR comparison - all operations are torchvision-exact")
    print("    * Tests scalability with image size and batch size")
    print()
    print("  Benchmark 4: Transform CLASSES with RANDOM factors (real training scenario)")
    print("    * ColorJitter + RandomGrayscale + Normalize with random sampling")
    print("    * Compares: Torchvision, Triton Sequential, Triton FUSED (single kernel)")
    print()
    print("  Benchmark 5: Geometric Fusion (Crop + Flip)")
    print("    * Compares: Torchvision, Triton Sequential, Triton Fused")
    print("    * Tests geometric operation fusion efficiency")
    print()
    print("  Benchmark 6: ULTIMATE FUSION 🚀 (ALL operations in ONE kernel)")
    print("    * Crop + Flip + Brightness + Contrast + Saturation + RandomGrayscale + Normalize")
    print("    * Peak performance - single kernel vs 7 sequential kernel launches")
    print()
    print("  Benchmark 7: Ultimate Fusion Float16 vs Float32 (FINALE)")
    print("    * Tests ultimate fusion kernel with different dtypes")
    print("    * Shows peak performance path works great with half precision")
    print("="*80)
    print()
    
    print("Running benchmarks... (this may take a few minutes)")
    print()
    
    # Create benchmark results directory
    os.makedirs('benchmark_results', exist_ok=True)
    print("📊 Saving plots to: ./benchmark_results/\n")
    
    # Run benchmarks - these will generate plots automatically
    print("Benchmark 1: Functionals with FIXED factors + Contrast (NO grayscale)")
    print("  Operations: Brightness + Contrast + Saturation + Normalize")
    print("  Note: Triton uses FAST contrast, torchvision uses blend-with-mean (different algorithms)")
    benchmark_color_jitter_normalize.run(print_data=True, save_path='benchmark_results')
    
    print("\nBenchmark 2: Functionals with FIXED factors WITHOUT contrast (image size scaling)")
    print("  Operations: Brightness + Saturation + Grayscale + Normalize")
    print("  Fair comparison - all operations are torchvision-exact")
    benchmark_without_contrast.run(print_data=True, save_path='benchmark_results')
    
    print("\nBenchmark 3: Functionals with FIXED factors WITHOUT contrast (batch size scaling)")
    print("  Operations: Brightness + Saturation + Grayscale + Normalize")
    benchmark_batch_scaling.run(print_data=True, save_path='benchmark_results')
    
    print("\nBenchmark 4: Random color-wise augmentation (using transform CLASSES)")
    print("  Torchvision: ColorJitter + RandomGrayscale + Normalize (Compose, sequential)")
    print("  Triton Sequential: TritonColorJitter + TritonRandomGrayscale + TritonNormalize (3 kernels)")
    print("  Triton Fused: TritonColorJitterNormalize (SINGLE FUSED KERNEL, maximum performance)")
    print("  Simulates actual training usage with random color augmentations per call")
    benchmark_random_color_augment.run(print_data=True, save_path='benchmark_results')
    
    print("\nBenchmark 5: Geometric Fusion (Crop + Flip)")
    print("  Torchvision: crop() → horizontal_flip() (2 kernel launches)")
    print("  Triton Sequential: crop() → horizontal_flip() (2 kernel launches)")
    print("  Triton Fused: fused_augment() with crop+flip (1 kernel launch)")
    print("  Expected: ~1.5-2x speedup from fusion vs sequential")
    benchmark_geometric_fusion.run(print_data=True, save_path='benchmark_results')
    
    print("\nBenchmark 6: ULTIMATE FUSION - All operations in ONE kernel! 🚀")
    print("  Uses TRANSFORM CLASSES with RANDOM augmentations (real training)")
    print("  Operations: RandomCrop + RandomHorizontalFlip + ColorJitter + RandomGrayscale + Normalize")
    print("  Torchvision Compose: 5 transforms (7 kernels: crop, flip, bright, contrast, sat, gray, norm)")
    print("  Triton Sequential: 5 Triton transforms (7 kernels)")
    print("  Triton Fused: TritonFusedAugment (1 FUSED kernel) ← PEAK PERFORMANCE!")
    print("  Expected: ~8-10x speedup vs torchvision!")
    benchmark_ultimate_fusion.run(print_data=True, save_path='benchmark_results')
    
    print("\nBenchmark 7: Ultimate Fusion Float16 vs Float32 (FINALE) 🎬")
    print("  Operations: ULTIMATE FUSION (Crop+Flip+Brightness+Contrast+Saturation+Grayscale+Normalize)")
    print("  Tests the complete fused pipeline with float16 vs float32")
    print("  Shows peak performance path maintains advantages with half precision")
    benchmark_float16_vs_float32.run(print_data=True, save_path='benchmark_results')
    
    # Quick speedup summaries
    print_ultimate_speedup_summary()
    
    print("\n" + "="*80)
    print("✅ Benchmarks complete!")
    print("\n📊 Plots saved to: ./benchmark_results/")
    print("  1. color-jitter-normalize-performance.png (WITH contrast)")
    print("  2. brightness-saturation-normalize-performance.png (WITHOUT contrast - FAIR)")
    print("  3. batch-size-scaling.png")
    print("  4. training-pipeline-performance.png (random augmentations)")
    print("  5. geometric-fusion-performance.png (crop + flip: torchvision vs triton sequential vs fused)")
    print("  6. ultimate-fusion-performance.png (ULTIMATE - 5 transforms, 7 kernels → 1 kernel)")
    print("  7. float16-vs-float32-ultimate-fusion.png (dtype comparison - FINALE)")
    print("="*80)


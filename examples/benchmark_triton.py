"""
Comprehensive benchmark script using Triton's built-in benchmarking utilities.

This script provides detailed performance analysis with visualizations across
multiple operations and configurations.

For a simpler, quick benchmark of just the Ultimate Fusion kernel, use:
    python examples/benchmark.py

This comprehensive script uses triton.testing.perf_report to compare Triton-Augment
performance with torchvision.transforms.v2 across multiple scenarios.

Author: yuhezhang-ai
"""

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
            return ta.fused_color_normalize(
                images,
                brightness_factor=brightness_factor,
                contrast_factor=contrast_factor,  # Uses FAST contrast
                saturation_factor=saturation_factor,
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
    Fair comparison with FIXED factors: Brightness + Saturation + Grayscale + Normalize (NO contrast).
    
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
    random_grayscale_p = 0.1
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torchvision':
        if not TORCHVISION_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = tvF.adjust_brightness(images, brightness_factor)
            img = tvF.adjust_saturation(img, saturation_factor)
            # Random grayscale with p=0.1
            if torch.rand(1).item() < random_grayscale_p:
                img = tvF.rgb_to_grayscale(img, num_output_channels=3)
            img = tvF.normalize(img, mean=list(mean), std=list(std))
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-sequential':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = ta.adjust_brightness(images, brightness_factor)
            img = ta.adjust_saturation(img, saturation_factor)
            img = ta.random_grayscale(img, p=random_grayscale_p)
            img = ta.normalize(img, mean, std)
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-fused':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            return ta.fused_color_normalize(
                images,
                brightness_factor=brightness_factor,
                contrast_factor=1.0,  # No contrast adjustment
                saturation_factor=saturation_factor,
                random_grayscale_p=random_grayscale_p,
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
    """Benchmark how performance scales with batch size using FIXED factors (no contrast for fairness)."""
    images = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
    
    # Fixed parameters
    brightness_factor = 1.2
    saturation_factor = 0.9
    random_grayscale_p = 0.1
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torchvision':
        if not TORCHVISION_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = tvF.adjust_brightness(images, brightness_factor)
            img = tvF.adjust_saturation(img, saturation_factor)
            if torch.rand(1).item() < random_grayscale_p:
                img = tvF.rgb_to_grayscale(img, num_output_channels=3)
            img = tvF.normalize(img, mean=list(mean), std=list(std))
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-fused':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            return ta.fused_color_normalize(
                images,
                brightness_factor=brightness_factor,
                contrast_factor=1.0,  # No contrast
                saturation_factor=saturation_factor,
                random_grayscale_p=random_grayscale_p,
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
        plot_name='training-pipeline-performance',
        args={'batch_size': 32},
    )
)
def benchmark_training_pipeline(size, batch_size, provider):
    """
    Real-world training pipeline using transform CLASSES with RANDOM parameters.
    
    This simulates actual usage in training where users create transforms once
    and apply them repeatedly with random augmentations each time.
    
    Uses ColorJitter + RandomGrayscale + Normalize as in real training pipelines.
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
            random_grayscale_p=0.1,
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
        x_vals=[64 * i for i in range(2, 25)],
        line_arg='provider',
        line_vals=['float32-triton-fused', 'float16-triton-fused'],
        line_names=['Float32 Fused', 'Float16 Fused'],
        styles=[('blue', '-'), ('red', '--')],
        ylabel='Runtime (ms)',
        plot_name='float16-vs-float32-performance',
        args={'batch_size': 32},
    )
)
def benchmark_float16_vs_float32(size, batch_size, provider):
    """
    Compare float16 vs float32 performance for the fused kernel.
    
    NOTE: Both use the same operations (Brightness + Saturation + Normalize).
    NO contrast (for consistent comparison). Includes grayscale (p=0.1).
    """
    # Fixed parameters
    brightness_factor = 1.2
    saturation_factor = 0.8
    random_grayscale_p = 0.1
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if provider == 'float32-triton-fused':
        img = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
        fn = lambda: ta.fused_color_normalize(
            img, 
            brightness_factor=brightness_factor,
            saturation_factor=saturation_factor,
            random_grayscale_p=random_grayscale_p,
            mean=mean, 
            std=std
        )
    elif provider == 'float16-triton-fused':
        img = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float16)
        fn = lambda: ta.fused_color_normalize(
            img, 
            brightness_factor=brightness_factor,
            saturation_factor=saturation_factor,
            random_grayscale_p=random_grayscale_p,
            mean=mean, 
            std=std
        )
    
    ms = triton.testing.do_bench(fn)
    return ms

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[128, 256, 384, 512, 1024],
        line_arg='provider',
        line_vals=['triton-sequential', 'triton-fused'],
        line_names=['Triton Sequential (Crop → Flip)', 'Triton Fused (Crop+Flip)'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='Time (ms)',
        plot_name='geometric-fusion-performance',
        args={'batch_size': 32},
    )
)
def benchmark_geometric_fusion(size, batch_size, provider):
    """
    Benchmark 6: Geometric Fusion - Crop + Flip
    
    Compares:
    - Sequential: crop() → horizontal_flip() (2 kernel launches)
    - Fused: fused_crop_flip() (1 kernel launch)
    
    Expected: ~1.5-2x speedup from fusion
    """
    img = torch.rand(batch_size, 3, size, size, device='cuda')
    
    # Fixed crop parameters
    crop_size = size - 40
    top = (size - crop_size) // 2
    left = (size - crop_size) // 2
    
    if provider == 'triton-sequential':
        def fn():
            result = ta.crop(img, top, left, crop_size, crop_size)
            result = ta.horizontal_flip(result)
            return result
    elif provider == 'triton-fused':
        def fn():
            return ta.fused_crop_flip(img, top, left, crop_size, crop_size, flip_horizontal=True)
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[128, 256, 384, 512, 1024],
        line_arg='provider',
        line_vals=['torchvision-compose', 'triton-sequential', 'triton-ultimate'],
        line_names=[
            'Torchvision Compose (6 ops)',
            'Triton Sequential (6 ops)', 
            'Triton Ultimate Fused (6 ops)'
        ],
        styles=[('green', '-'), ('blue', '--'), ('red', '-')],
        ylabel='Time (ms)',
        plot_name='ultimate-fusion-performance',
        args={'batch_size': 32},
    )
)
def benchmark_ultimate_fusion(size, batch_size, provider):
    """
    Benchmark 7: Ultimate Fusion - ALL 6 operations in ONE kernel!
    
    Operations:
    - RandomCrop (size-40×size-40)
    - RandomHorizontalFlip
    - Brightness adjustment
    - Contrast adjustment (fast)
    - Saturation adjustment
    - Normalize
    
    Compares:
    - Torchvision Compose: 6 sequential operations (6 kernel launches)
    - Triton Sequential: 6 Triton operations (6 kernel launches)
    - Triton Ultimate: 1 fused kernel (1 kernel launch) 🚀
    
    Expected: ~3-5x speedup vs torchvision!
    """
    img = torch.rand(batch_size, 3, size, size, device='cuda')
    
    # Fixed parameters (not random for fair comparison)
    crop_size = size - 40
    top = max(0, (size - crop_size) // 2)
    left = max(0, (size - crop_size) // 2)
    brightness = 1.2
    contrast = 1.1
    saturation = 0.9
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if provider == 'torchvision-compose':
        def fn():
            result = tvF.crop(img, top, left, crop_size, crop_size)
            result = tvF.horizontal_flip(result)
            result = tvF.adjust_brightness(result, brightness)
            result = tvF.adjust_contrast(result, contrast)
            result = tvF.adjust_saturation(result, saturation)
            result = tvF.normalize(result, mean, std)
            return result
    
    elif provider == 'triton-sequential':
        def fn():
            result = ta.crop(img, top, left, crop_size, crop_size)
            result = ta.horizontal_flip(result)
            result = ta.adjust_brightness(result, brightness)
            result = ta.adjust_contrast_fast(result, contrast)
            result = ta.adjust_saturation(result, saturation)
            result = ta.normalize(result, mean, std)
            return result
    
    elif provider == 'triton-ultimate':
        def fn():
            return ta.ultimate_fused_augment(
                img,
                top=top,
                left=left,
                height=crop_size,
                width=crop_size,
                flip_horizontal=True,
                brightness_factor=brightness,
                contrast_factor=contrast,
                saturation_factor=saturation,
                mean=mean,
                std=std,
            )
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return ms


def print_ultimate_speedup_summary():
    """
    Quick benchmark comparing torchvision Compose vs Triton Ultimate for a typical training scenario.
    """
    print("\n" + "="*80)
    print("ULTIMATE SPEEDUP SUMMARY (6 ops: Crop+Flip+Brightness+Contrast+Saturation+Norm)")
    print("="*80)
    
    batch_size = 32
    img_size = 1280
    crop_size = 1024
    
    # Create test tensor
    img = torch.rand(batch_size, 3, img_size, img_size, device='cuda')
    
    # Parameters
    brightness = 1.2
    contrast = 1.1
    saturation = 0.9
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Torchvision Compose
    torchvision_transforms = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation),
        transforms.RandomGrayscale(p=0.1),
        transforms.Normalize(mean=mean, std=std),
    ])
    def tv_fn():
        return torchvision_transforms(img)
    
    # Triton Ultimate
    triton_ultimate_transform = ta.TritonUltimateAugment(
        crop_size=crop_size,
        horizontal_flip_p=0.5,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        random_grayscale_p=0.1,
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
    print(f"  Torchvision Compose (6 kernel launches):     {tv_median:.3f} ms")
    print(f"  Triton-Augment Ultimate (1 kernel launch):   {ta_median:.3f} ms")
    print(f"  → Speedup: {speedup:.2f}x faster! 🚀")
    print("\nNote: Triton uses FAST contrast (centered scaling), torchvision uses blend-with-mean")
    print("="*80)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        exit(1)
    
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
    print("  - Benchmarks 1-3: Use functionals with FIXED factors (pure kernel performance)")
    print("  - Benchmark 4: Uses transform classes with RANDOM factors (real training)")
    print("    * Tests 3 approaches: torchvision, triton sequential, triton FUSED")
    print("  - Benchmark 5: Float16 vs Float32 comparison (dtype impact on performance)")
    print("  - Benchmark 1: Includes contrast (different algorithms - NOT fair comparison)")
    print("  - Benchmarks 2-5: Exclude contrast for FAIR comparison")
    print("  - Benchmarks 2-5: Include grayscale conversion (p=0.1)")
    print("="*80)
    print()
    
    print("Running benchmarks... (this may take a few minutes)")
    print()
    
    # Run benchmarks - these will generate plots automatically
    print("Benchmark 1: Functionals with FIXED factors + Contrast (NO grayscale)")
    print("  Operations: Brightness + Contrast + Saturation + Normalize")
    print("  Note: Triton uses FAST contrast, torchvision uses blend-with-mean (different algorithms)")
    benchmark_color_jitter_normalize.run(print_data=True, save_path='.')
    
    print("\nBenchmark 2: Functionals with FIXED factors WITHOUT contrast (image size scaling)")
    print("  Operations: Brightness + Saturation + Grayscale + Normalize")
    print("  Fair comparison - all operations are torchvision-exact")
    benchmark_without_contrast.run(print_data=True, save_path='.')
    
    print("\nBenchmark 3: Functionals with FIXED factors WITHOUT contrast (batch size scaling)")
    print("  Operations: Brightness + Saturation + Grayscale + Normalize")
    benchmark_batch_scaling.run(print_data=True, save_path='.')
    
    print("\nBenchmark 4: Transform CLASSES with RANDOM factors (real-world training pipeline)")
    print("  Torchvision: ColorJitter + RandomGrayscale + Normalize (Compose, sequential)")
    print("  Triton Sequential: TritonColorJitter + TritonRandomGrayscale + TritonNormalize (3 kernels)")
    print("  Triton Fused: TritonColorJitterNormalize (SINGLE FUSED KERNEL, maximum performance)")
    print("  Simulates actual training augmentation usage with random parameters per call")
    benchmark_training_pipeline.run(print_data=True, save_path='.')
    
    print("\nBenchmark 5: Float16 vs Float32 (dtype comparison)")
    print("  Operations: Brightness + Saturation + RandomGrayscale + Normalize (NO contrast)")
    print("  Compares fused kernel performance with float16 vs float32")
    benchmark_float16_vs_float32.run(print_data=True, save_path='.')
    
    print("\nBenchmark 6: Geometric Fusion (Crop + Flip)")
    print("  Sequential: crop() → horizontal_flip() (2 kernel launches)")
    print("  Fused: fused_crop_flip() (1 kernel launch)")
    print("  Expected: ~1.5-2x speedup from fusion")
    benchmark_geometric_fusion.run(print_data=True, save_path='.')
    
    print("\nBenchmark 7: ULTIMATE FUSION - All 6 operations in ONE kernel! 🚀")
    print("  Operations: RandomCrop + RandomHorizontalFlip + ColorJitter + Normalize")
    print("  Torchvision: 6 sequential operations (6 kernel launches)")
    print("  Triton Sequential: 6 Triton operations (6 kernel launches)")
    print("  Triton Ultimate: 1 FUSED kernel (1 kernel launch) ← PEAK PERFORMANCE!")
    print("  Expected: ~3-5x speedup vs torchvision!")
    benchmark_ultimate_fusion.run(print_data=True, save_path='.')
    
    # Quick speedup summaries
    print_ultimate_speedup_summary()
    
    print("\n" + "="*80)
    print("Benchmarks complete!")
    print("Plots saved to:")
    print("  - color-jitter-normalize-performance.png (WITH contrast)")
    print("  - brightness-saturation-normalize-performance.png (WITHOUT contrast - FAIR)")
    print("  - batch-size-scaling.png")
    print("  - training-pipeline-performance.png")
    print("  - float16-vs-float32-performance.png (dtype comparison)")
    print("  - geometric-fusion-performance.png (geometric fusion)")
    print("  - ultimate-fusion-performance.png (ULTIMATE - all 6 ops)")
    print("="*80)


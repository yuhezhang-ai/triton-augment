"""
Benchmark script using Triton's built-in benchmarking utilities.

This script uses triton.testing.perf_report to compare Triton-Augment
performance with torchvision.transforms.v2.

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


def print_speedup_summary():
    """Print a quick speedup comparison for real-world training usage."""
    print("\n" + "="*80)
    print("⚡ Quick Speedup Summary (224x224, batch=32)")
    print("="*80)
    print("Real-world training pipeline: ColorJitter + RandomGrayscale + Normalize")
    print()
    
    # Standard ImageNet size for realistic comparison
    test_img = torch.rand(32, 3, 224, 224, device='cuda', dtype=torch.float32)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Torchvision transforms (baseline)
    transform_tv = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Triton fused transform (our solution)
    transform_fused = ta.TritonColorJitterNormalize(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        random_grayscale_p=0.1,
        mean=mean,
        std=std,
    )
    
    # Benchmark
    t_tv = triton.testing.do_bench(lambda: transform_tv(test_img))
    t_fused = triton.testing.do_bench(lambda: transform_fused(test_img))
    
    speedup = t_tv / t_fused
    
    print(f"  torchvision Compose:          {t_tv:.3f} ms")
    print(f"  Triton-Augment Fused:         {t_fused:.3f} ms  (uses FAST contrast)")
    print()
    print(f"  🚀 Speedup: {speedup:.2f}x faster")
    print()
    print(f"  Note: Triton uses fast contrast (centered scaling), not torchvision's blend-with-mean")


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
    print("  - Benchmark 1: Includes contrast (different algorithms - NOT fair comparison)")
    print("  - Benchmarks 2-4: Exclude contrast for FAIR comparison")
    print("  - Benchmarks 2-4: Include grayscale conversion (p=0.1)")
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
    
    # Quick speedup summary
    print_speedup_summary()
    
    print("\n" + "="*80)
    print("Benchmarks complete!")
    print("Plots saved to:")
    print("  - color-jitter-normalize-performance.png (WITH contrast)")
    print("  - brightness-saturation-normalize-performance.png (WITHOUT contrast - FAIR)")
    print("  - batch-size-scaling.png")
    print("  - training-pipeline-performance.png")
    print("="*80)


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
    import torchvision.transforms.v2.functional as tvF
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("Warning: torchvision not installed")
    TORCHVISION_AVAILABLE = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot
        x_vals=[64 * i for i in range(2, 25)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['torchvision', 'triton-augment-sequential', 'triton-augment-fused'],
        line_names=['torchvision (v2)', 'Triton-Augment (sequential)', 'Triton-Augment (fused)'],
        styles=[('green', '-'), ('blue', '--'), ('red', '-')],
        ylabel='Runtime (ms)',  # Label name for the y-axis
        plot_name='color-jitter-normalize-performance',  # Name for the plot, used also as a file name for saving the plot.
        args={'batch_size': 32},  # Values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_color_jitter_normalize(size, batch_size, provider):
    """
    Benchmark color jitter + normalize operations.
    
    Args:
        size: Image size (square images)
        batch_size: Batch size
        provider: Which implementation to use
    """
    # Create test data
    images = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
    
    # Parameters
    brightness_factor = 1.2
    contrast_factor = 1.1
    saturation_factor = 0.9
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Warmup
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torchvision':
        if not TORCHVISION_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = tvF.adjust_brightness(images, brightness_factor)
            img = tvF.adjust_contrast(img, contrast_factor)
            img = tvF.adjust_saturation(img, saturation_factor)
            # Normalize
            mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
            std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
            img = (img - mean_t) / std_t
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-sequential':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            img = ta.adjust_brightness(images, brightness_factor)
            img = ta.adjust_contrast(img, contrast_factor)
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
                contrast_factor=contrast_factor,
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
        x_names=['batch_size'],
        x_vals=[2**i for i in range(0, 6)],  # 1, 2, 4, 8, 16, 32
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
    """Benchmark how performance scales with batch size."""
    images = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
    
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
            mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
            std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
            img = (img - mean_t) / std_t
            return img
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        
    elif provider == 'triton-augment-fused':
        if not TRITON_AUGMENT_AVAILABLE:
            return 0, 0, 0
        
        def fn():
            return ta.fused_color_normalize(
                images,
                brightness_factor=brightness_factor,
                contrast_factor=contrast_factor,
                saturation_factor=saturation_factor,
                mean=mean,
                std=std,
            )
        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ms, min_ms, max_ms


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
    
    print("Running benchmarks... (this may take a few minutes)")
    print()
    
    # Run benchmarks - these will generate plots automatically
    print("Benchmark 1: Image size scaling (batch_size=32)")
    benchmark_color_jitter_normalize.run(print_data=True, save_path='.')
    
    print("\nBenchmark 2: Batch size scaling (image_size=224)")
    benchmark_batch_scaling.run(print_data=True, save_path='.')
    
    print("\n" + "="*80)
    print("Benchmarks complete!")
    print("Plots saved to:")
    print("  - color-jitter-normalize-performance.png")
    print("  - batch-size-scaling.png")
    print("="*80)


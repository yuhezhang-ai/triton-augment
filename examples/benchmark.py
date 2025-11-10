"""
Benchmark script comparing Triton-Augment with standard PyTorch operations.

This script measures the performance of fused operations versus sequential
PyTorch operations for various batch sizes and image dimensions.
"""

import torch
import time
import argparse
from typing import Tuple

try:
    import triton_augment as ta
    TRITON_AVAILABLE = True
except ImportError:
    print("Warning: triton_augment not installed. Install with: pip install -e .")
    TRITON_AVAILABLE = False


def pytorch_augment(
    images: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Sequential PyTorch implementation of color jitter and normalize.
    
    This represents the traditional approach with multiple separate operations.
    """
    # Brightness
    img = images + brightness
    
    # Contrast
    img = img * contrast
    
    # Saturation (simplified - full implementation would be more complex)
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    img = gray + saturation * (img - gray)
    
    # Normalize
    mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    img = (img - mean_t) / std_t
    
    return img


def benchmark_operation(
    func,
    *args,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    **kwargs
) -> Tuple[float, float]:
    """
    Benchmark a function with warmup and multiple runs.
    
    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup_runs):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return mean_time, std_time


def run_benchmark(
    batch_size: int,
    height: int,
    width: int,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
):
    """Run benchmark for specific configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: Batch={batch_size}, Size={height}x{width}")
    print(f"{'='*60}")
    
    # Create test data
    images = torch.rand(batch_size, 3, height, width, device='cuda')
    
    # Parameters
    brightness = 0.1
    contrast = 1.2
    saturation = 0.8
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Benchmark PyTorch sequential
    print("\nBenchmarking PyTorch (sequential operations)...")
    pytorch_mean, pytorch_std = benchmark_operation(
        pytorch_augment,
        images,
        brightness,
        contrast,
        saturation,
        mean,
        std,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )
    print(f"  Time: {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    
    if TRITON_AVAILABLE:
        # Benchmark Triton fused
        print("\nBenchmarking Triton-Augment (fused kernel)...")
        
        def triton_augment_func(img):
            return ta.fused_color_normalize(
                img,
                brightness_factor=brightness,
                contrast_factor=contrast,
                saturation_factor=saturation,
                mean=mean,
                std=std,
            )
        
        triton_mean, triton_std = benchmark_operation(
            triton_augment_func,
            images,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )
        print(f"  Time: {triton_mean:.3f} ± {triton_std:.3f} ms")
        
        # Calculate speedup
        speedup = pytorch_mean / triton_mean
        print(f"\n{'='*60}")
        print(f"Speedup: {speedup:.2f}x faster with Triton-Augment")
        print(f"{'='*60}")
        
        return pytorch_mean, triton_mean, speedup
    else:
        print("\nTriton-Augment not available for comparison.")
        return pytorch_mean, None, None


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Benchmark Triton-Augment vs PyTorch')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16, 32],
                        help='Batch sizes to benchmark')
    parser.add_argument('--image-sizes', nargs='+', type=int, default=[224, 384, 512],
                        help='Image sizes to benchmark (square images)')
    parser.add_argument('--warmup-runs', type=int, default=10,
                        help='Number of warmup runs')
    parser.add_argument('--benchmark-runs', type=int, default=100,
                        help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        return
    
    print("="*60)
    print("Triton-Augment Performance Benchmark")
    print("="*60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    if TRITON_AVAILABLE:
        print(f"Triton-Augment version: {ta.__version__}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Benchmark runs: {args.benchmark_runs}")
    
    # Store results for summary
    results = []
    
    # Run benchmarks for different configurations
    for image_size in args.image_sizes:
        for batch_size in args.batch_sizes:
            pytorch_time, triton_time, speedup = run_benchmark(
                batch_size=batch_size,
                height=image_size,
                width=image_size,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )
            results.append((batch_size, image_size, pytorch_time, triton_time, speedup))
    
    # Print summary
    if TRITON_AVAILABLE and results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Batch':<8} {'Size':<8} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
        print("-"*80)
        for batch, size, pt_time, tr_time, speedup in results:
            if tr_time is not None:
                print(f"{batch:<8} {size:<8} {pt_time:<15.3f} {tr_time:<15.3f} {speedup:<10.2f}x")
        print("="*80)


if __name__ == '__main__':
    main()


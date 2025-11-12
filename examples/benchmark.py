"""
Simple benchmark for TritonUltimateAugment (Ultimate Fusion).

This script compares three approaches:
1. Torchvision Compose (baseline)
2. Triton-Augment Sequential (individual transforms)
3. Triton-Augment Fused (single kernel - FASTEST!)

Outputs a clean markdown table you can paste into README.md.

Usage:
    python examples/benchmark.py
"""

import torch
import triton
from triton.testing import do_bench
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as tvF

import triton_augment as ta
import triton_augment.functional as F


def benchmark_ultimate(batch_size=32, image_size=224, crop_size=112):
    """
    Benchmark ultimate fusion vs sequential vs torchvision.
    
    Args:
        batch_size: Number of images in batch
        image_size: Input image size (square)
        crop_size: Output crop size (square)
    
    Returns:
        dict: Results with time in ms and speedup
    """
    img = torch.rand(batch_size, 3, image_size, image_size, device='cuda')
    
    # Fixed parameters for fair comparison
    top = (image_size - crop_size) // 3
    left = (image_size - crop_size) // 3
    brightness_factor = 1.2
    contrast_factor = 1.1
    saturation_factor = 0.9
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # ========================================================================
    # 1. Torchvision Compose (Baseline)
    # ========================================================================
    def torchvision_fn():
        result = tvF.crop(img, top, left, crop_size, crop_size)
        result = tvF.horizontal_flip(result)
        result = tvF.adjust_brightness(result, brightness_factor)
        # Note: Skipping contrast - torchvision uses different algorithm
        result = tvF.adjust_saturation(result, saturation_factor)
        result = tvF.normalize(result, mean, std)
        return result
    
    torchvision_time = do_bench(torchvision_fn, warmup=25, rep=100)
    
    # ========================================================================
    # 2. Triton-Augment Sequential (Individual transforms)
    # ========================================================================
    def triton_sequential_fn():
        result = F.crop(img, top, left, crop_size, crop_size)
        result = F.horizontal_flip(result)
        result = F.adjust_brightness(result, brightness_factor)
        result = F.adjust_contrast_fast(result, contrast_factor)
        result = F.adjust_saturation(result, saturation_factor)
        result = F.normalize(result, mean, std)
        return result
    
    triton_sequential_time = do_bench(triton_sequential_fn, warmup=25, rep=100)
    
    # ========================================================================
    # 3. Triton-Augment Fused (Single kernel - FASTEST!)
    # ========================================================================
    def triton_fused_fn():
        return F.ultimate_fused_augment(
            img,
            top=top,
            left=left,
            height=crop_size,
            width=crop_size,
            flip_horizontal=True,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            mean=mean,
            std=std,
        )
    
    triton_fused_time = do_bench(triton_fused_fn, warmup=25, rep=100)
    
    # ========================================================================
    # Calculate speedups
    # ========================================================================
    speedup_sequential = torchvision_time / triton_sequential_time
    speedup_fused = torchvision_time / triton_fused_time
    
    return {
        'batch_size': batch_size,
        'image_size': f"{image_size}x{image_size}",
        'crop_size': f"{crop_size}x{crop_size}",
        'torchvision_time': torchvision_time,
        'triton_sequential_time': triton_sequential_time,
        'triton_fused_time': triton_fused_time,
        'speedup_sequential': speedup_sequential,
        'speedup_fused': speedup_fused,
    }


def print_table(results):
    """Print results as a markdown table."""
    print("\n" + "="*80)
    print("ULTIMATE FUSION BENCHMARK RESULTS")
    print("="*80)
    print("\nOperations: Crop + Flip + Brightness + Contrast + Saturation + Normalize")
    print("Device:", torch.cuda.get_device_name(0))
    print("\n")
    
    # Header
    print("| Image Size | Batch | Torchvision (ms) | Triton Sequential (ms) | Triton Fused (ms) | Speedup (Sequential) | Speedup (Fused) |")
    print("|------------|-------|------------------|------------------------|-------------------|----------------------|-----------------|")
    
    # Rows
    for r in results:
        print(f"| {r['image_size']:10s} | {r['batch_size']:5d} | "
              f"{r['torchvision_time']:16.3f} | {r['triton_sequential_time']:22.3f} | "
              f"{r['triton_fused_time']:17.3f} | {r['speedup_sequential']:20.2f}x | "
              f"{r['speedup_fused']:15.2f}x |")
    
    print("\n")
    print("**Note**: Triton uses fast contrast (centered scaling), not torchvision's blend-with-mean.")
    print("="*80)


def main():
    """Run benchmarks for different configurations."""
    print("Starting Ultimate Fusion Benchmark...")
    print("This may take 1-2 minutes...\n")
    
    # Test configurations
    configs = [
        # (batch_size, image_size, crop_size)
        (32, 256, 224),   # Standard ImageNet
        (64, 256, 224),   # Larger batch
        (32, 600, 512),   # High resolution
        (32, 1280, 1024),   # Very high res
    ]
    
    results = []
    for i, (batch, img_size, crop_size) in enumerate(configs, 1):
        print(f"Running config {i}/{len(configs)}: batch={batch}, image={img_size}x{img_size}, crop={crop_size}x{crop_size}...")
        result = benchmark_ultimate(batch, img_size, crop_size)
        results.append(result)
    
    # Print table
    print_table(results)
    
    # Print summary
    avg_speedup_fused = sum(r['speedup_fused'] for r in results) / len(results)
    print(f"\n🚀 Average Speedup (Triton Fused vs Torchvision): {avg_speedup_fused:.2f}x\n")


if __name__ == '__main__':
    main()


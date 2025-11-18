"""
Simple benchmark for TritonFusedAugment (Fused Augmentation).

This script compares three approaches:
1. Torchvision Compose (baseline)
2. Triton-Augment Sequential (individual transforms)
3. Triton-Augment Fused (single kernel - FASTEST!)

Outputs a clean markdown table you can paste into README.md.

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --autotune  # Enable auto-tuning for optimal performance
"""

import argparse
import torch
import triton
from triton.testing import do_bench
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as tvF

import triton_augment as ta
import triton_augment.functional as F


def benchmark_ultimate(batch_size=32, image_size=224, crop_size=112):
    """
    Benchmark ultimate fusion vs sequential vs torchvision in REAL TRAINING scenario.
    
    Uses transform classes with random augmentations:
    - RandomCrop
    - RandomHorizontalFlip (p=0.5)
    - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)
    - RandomGrayscale (p=0.1)
    - Normalize (ImageNet mean/std)
    
    Args:
        batch_size: Number of images in batch
        image_size: Input image size (square)
        crop_size: Output crop size (square)
    
    Returns:
        dict: Results with time in ms and speedup
    """
    img = torch.rand(batch_size, 3, image_size, image_size, device='cuda')
    
    # Parameters for augmentation (random ranges and normalization)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # ========================================================================
    # 1. Torchvision Compose (Baseline) - WITH RANDOM AUGMENTATIONS
    # ========================================================================
    torchvision_transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    def torchvision_fn():
        return torchvision_transform(img)
    
    torchvision_time = do_bench(torchvision_fn, warmup=25, rep=100)
    
    # ========================================================================
    # 2. Triton-Augment Sequential (Individual transform classes)
    # ========================================================================
    triton_sequential_transform = transforms.Compose([
        ta.TritonRandomCrop(crop_size),
        ta.TritonRandomHorizontalFlip(p=0.5),
        ta.TritonColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ta.TritonRandomGrayscale(p=0.1),
        ta.TritonNormalize(mean=mean, std=std),
    ])
    
    def triton_sequential_fn():
        return triton_sequential_transform(img)
    
    triton_sequential_time = do_bench(triton_sequential_fn, warmup=25, rep=100)
    
    # ========================================================================
    # 3. Triton-Augment Ultimate Fused (Single kernel - FASTEST!)
    # ========================================================================
    triton_fused_transform = ta.TritonFusedAugment(
        crop_size=crop_size,
        horizontal_flip_p=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        grayscale_p=0.1,
        mean=mean,
        std=std,
    )
    
    def triton_fused_fn():
        return triton_fused_transform(img)
    
    triton_fused_time = do_bench(triton_fused_fn, warmup=25, rep=100)

    # 4. Kornia (generally slower than torchvision.transforms.v2, not included in this benchmark for clarity.)
    # import kornia.augmentation as K

    # kornia_transform = K.AugmentationSequential(
    #     K.RandomCrop(size=crop_size, align_corners=False, p=1.0), 
    #     K.RandomHorizontalFlip(p=0.5),
    #     K.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, p=1.0),
    #     K.RandomGrayscale(p=0.1),
    #     K.Normalize(mean=mean.tolist(), std=std.tolist()),    
    # )

    # def kornia_fn():
    #     return kornia_transform(img)

    # kornia_time = do_bench(kornia_fn, warmup=25, rep=100)
    
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
    print("ULTIMATE FUSION BENCHMARK RESULTS - REAL TRAINING SCENARIO")
    print("="*80)
    print("\nRandom Augmentations:")
    print("  - RandomCrop + RandomHorizontalFlip(p=0.5)")
    print("  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)")
    print("  - RandomGrayscale(p=0.1) + Normalize(ImageNet)")
    print("\nDevice:", torch.cuda.get_device_name(0))
    print("\n")
    
    # Header
    print("| Image Size | Batch |  Crop Size | Torchvision (ms) | Triton Sequential (ms) | Triton Fused (ms) | Speedup (Sequential)  | Speedup (Fused)  |")
    print("|------------|-------|------------|------------------|------------------------|-------------------|-----------------------|------------------|")
    
    # Rows
    for r in results:
        print(f"| {r['image_size']:10s} | {r['batch_size']:5d} | {r['crop_size']:10s} | "
              f"{r['torchvision_time']:16.3f} | {r['triton_sequential_time']:22.3f} | "
              f"{r['triton_fused_time']:17.3f} | {r['speedup_sequential']:20.2f}x | "
              f"{r['speedup_fused']:15.2f}x |")
    
    print("\n")
    print("**Note**: Triton uses fast contrast (centered scaling), not torchvision's blend-with-mean.")
    print("="*80)


def main():
    """Run benchmarks for different configurations with RANDOM augmentations."""
    print("Starting Ultimate Fusion Benchmark (Real Training Scenario)...")
    print("Using transform classes with random augmentations (RandomCrop, RandomFlip, etc.)")
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
    parser = argparse.ArgumentParser(
        description='Benchmark Triton-Augment vs Torchvision'
    )
    parser.add_argument(
        '--autotune',
        action='store_true',
        help='Enable auto-tuning for optimal performance (takes 5-10s on first run)'
    )
    args = parser.parse_args()
    
    if args.autotune:
        ta.enable_autotune()
        print("🔧 Auto-tuning ENABLED - will test 12 configs on first run (~5-10 seconds)\n")
    else:
        print("ℹ️  Auto-tuning DISABLED (default config). Use --autotune for optimal performance.\n")
    
    main()


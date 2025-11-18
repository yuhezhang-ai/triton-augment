"""
Simple benchmark for video (5D tensor) augmentation.

This script compares four approaches for video augmentation:
1. Torchvision Compose (baseline - processes frames independently)
2. Kornia VideoSequential (video-aware augmentation)
3. Triton-Augment Sequential (individual transforms)
4. Triton-Augment Fused (single kernel - FASTEST!)

Input shape: [N, T, C, H, W] where N=batch, T=num_frames

Outputs a clean markdown table you can paste into documentation.

Usage:
    python examples/benchmark_video.py
    python examples/benchmark_video.py --autotune  # Enable auto-tuning for optimal performance
"""

import argparse
import torch
import triton
from triton.testing import do_bench
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as tvF

import triton_augment as ta
import triton_augment.functional as F

try:
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: Kornia not available. Install with: pip install kornia")


def benchmark_video(batch_size=8, num_frames=16, image_size=224, crop_size=112):
    """
    Benchmark video augmentation approaches.
    
    Uses transform classes with random augmentations:
    - RandomCrop
    - RandomHorizontalFlip (p=0.5)
    - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)
    - RandomGrayscale (p=0.1)
    - Normalize (ImageNet mean/std)
    
    Args:
        batch_size: Number of videos in batch
        num_frames: Number of frames per video
        image_size: Input frame size (square)
        crop_size: Output crop size (square)
    
    Returns:
        dict: Results with time in ms and speedup
    """
    # 5D tensor: [N, T, C, H, W]
    img = torch.rand(batch_size, num_frames, 3, image_size, image_size, device='cuda')
    
    # Parameters for augmentation (random ranges and normalization)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # ========================================================================
    # 1. Torchvision Compose (Baseline) - Processes frames independently
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
    # 2. Kornia VideoSequential (if available)
    # ========================================================================
    kornia_time = None
    if KORNIA_AVAILABLE:
        kornia_transform = K.VideoSequential(
            K.RandomCrop(size=(crop_size, crop_size), align_corners=False, p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, p=1.0),
            K.RandomGrayscale(p=0.1),
            K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            same_on_frame=True,  # Consistent augmentation across frames
            data_format="BTCHW",
        )
        
        def kornia_fn():
            return kornia_transform(img)
        
        kornia_time = do_bench(kornia_fn, warmup=25, rep=100)
    
    # ========================================================================
    # 3. Triton-Augment Sequential (Individual transform classes)
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
    # 4. Triton-Augment Fused (Single kernel - FASTEST!)
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
        same_on_frame=True,  # Same augmentation for all frames in a video
    )
    
    def triton_fused_fn():
        return triton_fused_transform(img)
    
    triton_fused_time = do_bench(triton_fused_fn, warmup=25, rep=100)
    
    # ========================================================================
    # Calculate speedups
    # ========================================================================
    speedup_sequential_vs_tv = torchvision_time / triton_sequential_time
    speedup_fused_vs_tv = torchvision_time / triton_fused_time
    speedup_kornia_vs_tv = torchvision_time / kornia_time if kornia_time else None
    
    # Speedup vs Kornia (since Kornia is slower than Torchvision)
    speedup_sequential_vs_kornia = kornia_time / triton_sequential_time if kornia_time else None
    speedup_fused_vs_kornia = kornia_time / triton_fused_time if kornia_time else None
    
    result = {
        'batch_size': batch_size,
        'num_frames': num_frames,
        'image_size': f"{image_size}x{image_size}",
        'crop_size': f"{crop_size}x{crop_size}",
        'torchvision_time': torchvision_time,
        'triton_sequential_time': triton_sequential_time,
        'triton_fused_time': triton_fused_time,
        'speedup_sequential_vs_tv': speedup_sequential_vs_tv,
        'speedup_fused_vs_tv': speedup_fused_vs_tv,
    }
    
    if kornia_time:
        result['kornia_time'] = kornia_time
        result['speedup_kornia_vs_tv'] = speedup_kornia_vs_tv
        result['speedup_sequential_vs_kornia'] = speedup_sequential_vs_kornia
        result['speedup_fused_vs_kornia'] = speedup_fused_vs_kornia
    
    return result


def print_table(results):
    """Print results as a markdown table."""
    print("\n" + "="*100)
    print("VIDEO (5D TENSOR) AUGMENTATION BENCHMARK - REAL TRAINING SCENARIO")
    print("="*100)
    print("\nRandom Augmentations:")
    print("  - RandomCrop + RandomHorizontalFlip(p=0.5)")
    print("  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)")
    print("  - RandomGrayscale(p=0.1)")
    print("  - Normalize()")
    print("\nInput Shape: [N, T, C, H, W]")
    print("Device:", torch.cuda.get_device_name(0))
    print("\n")
    
    # Check if Kornia results are available
    has_kornia = 'kornia_time' in results[0]
    
    # Header
    if has_kornia:
        print("| Batch | Frames | Image Size | Crop Size | Torchvision  | Kornia VideoSeq  | Triton Sequential                    | Triton Fused                         |")
        print("|-------|--------|------------|-----------|--------------|------------------|--------------------------------------|--------------------------------------|")
    else:
        print("| Batch | Frames | Image Size | Crop Size | Torchvision   | Triton Sequential        | Triton Fused                        |")
        print("|-------|--------|------------|-----------|---------------|--------------------------|-------------------------------------|")
    
    # Rows
    for r in results:
        if has_kornia:
            triton_seq_str = f"{r['triton_sequential_time']:.2f}ms ({r['speedup_sequential_vs_tv']:.1f}x TV, {r['speedup_sequential_vs_kornia']:.1f}x Kornia)"
            triton_fused_str = f"{r['triton_fused_time']:.2f}ms ({r['speedup_fused_vs_tv']:.1f}x TV, {r['speedup_fused_vs_kornia']:.1f}x Kornia)"
            print(f"| {r['batch_size']:5d} | {r['num_frames']:6d} | {r['image_size']:10s} | {r['crop_size']:9s} | "
                  f"{r['torchvision_time']:10.2f}ms | {r['kornia_time']:14.2f}ms | "
                  f"{triton_seq_str:<36s} | {triton_fused_str:<36s} |")
        else:
            triton_seq_str = f"{r['triton_sequential_time']:.2f}ms ({r['speedup_sequential_vs_tv']:.1f}x TV)"
            triton_fused_str = f"{r['triton_fused_time']:.2f}ms ({r['speedup_fused_vs_tv']:.1f}x TV)"
            print(f"| {r['batch_size']:5d} | {r['num_frames']:6d} | {r['image_size']:10s} | {r['crop_size']:9s} | "
                  f"{r['torchvision_time']:10.2f}ms | "
                  f"{triton_seq_str:<26s} | {triton_fused_str:<34s} |")
    
    print("\n")
    print("**Notes**:")
    print("  - Torchvision: Processes all batches and frames with the same augmentations in multiple kernels launches; no same_on_frame=False support")
    if has_kornia:
        print("  - Kornia VideoSequential: Native 5D support with same_on_frame=True (typically slower than Torchvision)")
    print("  - Triton Sequential: Individual transforms (5 kernel launches)")
    print("  - Triton Fused: Single kernel launch with same_on_frame=True (consistent augmentation)")
    print("  - Speedup format: 'Xms (Yx TV, Wx Kornia)' means X milliseconds, Y times faster than Torchvision, W times faster than Kornia")
    print("  - Triton uses fast contrast (centered scaling), not torchvision's blend-with-mean")
    print("="*100)


def main():
    """Run benchmarks for different video configurations."""
    print("Starting Video (5D Tensor) Augmentation Benchmark...")
    print("Testing [N, T, C, H, W] shaped tensors with video-aware augmentation")
    if not KORNIA_AVAILABLE:
        print("⚠️  Kornia not available - install with: pip install kornia")
    print("This may take 2-3 minutes...\n")
    
    # Test configurations
    configs = [
        # (batch_size, num_frames, image_size, crop_size)
        (8, 16, 256, 224),     # Standard video batch
        (4, 32, 256, 224),     # Longer video
        (16, 8, 256, 224),     # More videos, fewer frames
        (8, 16, 512, 448),     # Higher resolution
    ]
    
    results = []
    for i, (batch, frames, img_size, crop_size) in enumerate(configs, 1):
        print(f"Running config {i}/{len(configs)}: batch={batch}, frames={frames}, "
              f"image={img_size}x{img_size}, crop={crop_size}x{crop_size}...")
        result = benchmark_video(batch, frames, img_size, crop_size)
        results.append(result)
    
    # Print table
    print_table(results)
    
    # Print summary
    avg_speedup_sequential_vs_tv = sum(r['speedup_sequential_vs_tv'] for r in results) / len(results)
    avg_speedup_fused_vs_tv = sum(r['speedup_fused_vs_tv'] for r in results) / len(results)
    
    print(f"\n🚀 Average Speedup Summary:")
    print(f"   Triton vs Torchvision:")
    print(f"     - Sequential: {avg_speedup_sequential_vs_tv:.2f}x")
    print(f"     - Fused: {avg_speedup_fused_vs_tv:.2f}x")
    
    if KORNIA_AVAILABLE and 'speedup_kornia_vs_tv' in results[0]:
        avg_speedup_kornia_vs_tv = sum(r['speedup_kornia_vs_tv'] for r in results) / len(results)
        avg_speedup_sequential_vs_kornia = sum(r['speedup_sequential_vs_kornia'] for r in results) / len(results)
        avg_speedup_fused_vs_kornia = sum(r['speedup_fused_vs_kornia'] for r in results) / len(results)
        print(f"   Kornia vs Torchvision: {avg_speedup_kornia_vs_tv:.2f}x (Kornia is slower)")
        print(f"   Triton vs Kornia:")
        print(f"     - Sequential: {avg_speedup_sequential_vs_kornia:.2f}x")
        print(f"     - Fused: {avg_speedup_fused_vs_kornia:.2f}x")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark Video (5D) Augmentation: Triton-Augment vs Torchvision vs Kornia'
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


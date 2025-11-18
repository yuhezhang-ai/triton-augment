"""
Simple benchmark for video (5D tensor) augmentation.

This script compares four approaches for video augmentation:
1. Torchvision Compose (baseline - processes frames independently)
2. Kornia VideoSequential (video-aware augmentation)
3. Triton-Augment with same_on_frame=True (consistent across frames)
4. Triton-Augment with same_on_frame=False (independent per frame)

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
        # Torchvision doesn't natively support 5D, so we process frame-by-frame
        N, T = img.shape[0], img.shape[1]
        # Reshape to [N*T, C, H, W]
        frames = img.reshape(N * T, *img.shape[2:])
        # Apply transform
        result = torchvision_transform(frames)
        # Reshape back to [N, T, C, H', W']
        return result.reshape(N, T, *result.shape[1:])
    
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
    # 3. Triton-Augment with same_on_frame=True (consistent across frames)
    # ========================================================================
    triton_same_frame_transform = ta.TritonFusedAugment(
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
    
    def triton_same_frame_fn():
        return triton_same_frame_transform(img)
    
    triton_same_frame_time = do_bench(triton_same_frame_fn, warmup=25, rep=100)
    
    # ========================================================================
    # 4. Triton-Augment with same_on_frame=False (independent per frame)
    # ========================================================================
    triton_diff_frame_transform = ta.TritonFusedAugment(
        crop_size=crop_size,
        horizontal_flip_p=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        grayscale_p=0.1,
        mean=mean,
        std=std,
        same_on_frame=False,  # Different augmentation for each frame
    )
    
    def triton_diff_frame_fn():
        return triton_diff_frame_transform(img)
    
    triton_diff_frame_time = do_bench(triton_diff_frame_fn, warmup=25, rep=100)
    
    # ========================================================================
    # Calculate speedups
    # ========================================================================
    speedup_same_frame = torchvision_time / triton_same_frame_time
    speedup_diff_frame = torchvision_time / triton_diff_frame_time
    speedup_kornia = torchvision_time / kornia_time if kornia_time else None
    
    result = {
        'batch_size': batch_size,
        'num_frames': num_frames,
        'image_size': f"{image_size}x{image_size}",
        'crop_size': f"{crop_size}x{crop_size}",
        'torchvision_time': torchvision_time,
        'triton_same_frame_time': triton_same_frame_time,
        'triton_diff_frame_time': triton_diff_frame_time,
        'speedup_same_frame': speedup_same_frame,
        'speedup_diff_frame': speedup_diff_frame,
    }
    
    if kornia_time:
        result['kornia_time'] = kornia_time
        result['speedup_kornia'] = speedup_kornia
    
    return result


def print_table(results):
    """Print results as a markdown table."""
    print("\n" + "="*100)
    print("VIDEO (5D TENSOR) AUGMENTATION BENCHMARK - REAL TRAINING SCENARIO")
    print("="*100)
    print("\nRandom Augmentations:")
    print("  - RandomCrop + RandomHorizontalFlip(p=0.5)")
    print("  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)")
    print("  - RandomGrayscale(p=0.1) + Normalize(ImageNet)")
    print("\nInput Shape: [N, T, C, H, W]")
    print("Device:", torch.cuda.get_device_name(0))
    print("\n")
    
    # Check if Kornia results are available
    has_kornia = 'kornia_time' in results[0]
    
    # Header
    if has_kornia:
        print("| Batch | Frames | Image Size |  Crop Size  | Torchvision (ms) | Kornia VideoSeq (ms) | Triton same_on_frame=True (ms) | Triton same_on_frame=False (ms) | Speedup vs TV (Kornia) | Speedup vs TV (same) | Speedup vs TV (diff) |")
        print("|-------|--------|------------|-------------|------------------|----------------------|--------------------------------|---------------------------------|------------------------|----------------------|----------------------|")
    else:
        print("| Batch | Frames | Image Size |  Crop Size  | Torchvision (ms) | Triton same_on_frame=True (ms) | Triton same_on_frame=False (ms) | Speedup vs TV (same) | Speedup vs TV (diff) |")
        print("|-------|--------|------------|-------------|------------------|--------------------------------|---------------------------------|----------------------|----------------------|")
    
    # Rows
    for r in results:
        if has_kornia:
            print(f"| {r['batch_size']:5d} | {r['num_frames']:6d} | {r['image_size']:10s} | {r['crop_size']:10s} | "
                  f"{r['torchvision_time']:16.3f} | {r['kornia_time']:20.3f} | "
                  f"{r['triton_same_frame_time']:30.3f} | {r['triton_diff_frame_time']:31.3f} | "
                  f"{r['speedup_kornia']:22.2f}x | {r['speedup_same_frame']:20.2f}x | "
                  f"{r['speedup_diff_frame']:20.2f}x |")
        else:
            print(f"| {r['batch_size']:5d} | {r['num_frames']:6d} | {r['image_size']:10s} | {r['crop_size']:10s} | "
                  f"{r['torchvision_time']:16.3f} | "
                  f"{r['triton_same_frame_time']:30.3f} | {r['triton_diff_frame_time']:31.3f} | "
                  f"{r['speedup_same_frame']:20.2f}x | {r['speedup_diff_frame']:20.2f}x |")
    
    print("\n")
    print("**Notes**:")
    print("  - Torchvision: Processes frames independently (no native 5D support)")
    if has_kornia:
        print("  - Kornia VideoSequential: Native 5D support with same_on_frame=True")
    print("  - Triton same_on_frame=True: All frames in a video get same augmentation (like Kornia)")
    print("  - Triton same_on_frame=False: Each frame gets independent augmentation (like Torchvision)")
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
    avg_speedup_same = sum(r['speedup_same_frame'] for r in results) / len(results)
    avg_speedup_diff = sum(r['speedup_diff_frame'] for r in results) / len(results)
    
    print(f"\n🚀 Average Speedup Summary:")
    print(f"   - Triton (same_on_frame=True) vs Torchvision: {avg_speedup_same:.2f}x")
    print(f"   - Triton (same_on_frame=False) vs Torchvision: {avg_speedup_diff:.2f}x")
    
    if KORNIA_AVAILABLE and 'speedup_kornia' in results[0]:
        avg_speedup_kornia = sum(r['speedup_kornia'] for r in results) / len(results)
        print(f"   - Kornia VideoSequential vs Torchvision: {avg_speedup_kornia:.2f}x")
    
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


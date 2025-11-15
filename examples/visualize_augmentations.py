"""
Visualization script to compare Triton-Augment with torchvision.

This script demonstrates that Triton-Augment produces identical results to torchvision
(except for fast contrast, which is intentionally different for performance).

Requires: pillow, matplotlib, torchvision

Usage:
    python examples/visualize_augmentations.py
    python examples/visualize_augmentations.py --image path/to/image.jpg
"""

import torch
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Error: matplotlib not installed. Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.transforms.v2.functional as tvF
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("Error: torchvision not installed. Install with: pip install torchvision pillow")
    TORCHVISION_AVAILABLE = False

try:
    import triton_augment as ta
    import triton_augment.functional as F
    TRITON_AVAILABLE = True
except ImportError:
    print("Error: triton_augment not installed. Install with: pip install -e .")
    TRITON_AVAILABLE = False


def create_sample_image(size=224):
    """Create a sample colorful image for demonstration."""
    # Create a gradient image with different colors in each region
    img = torch.zeros(1, 3, size, size)
    
    # Red gradient (top-left)
    img[0, 0, :size//2, :size//2] = torch.linspace(0, 1, size//2).unsqueeze(0).repeat(size//2, 1)
    
    # Green gradient (top-right)
    img[0, 1, :size//2, size//2:] = torch.linspace(0, 1, size//2).unsqueeze(1).repeat(1, size//2)
    
    # Blue gradient (bottom-left)
    img[0, 2, size//2:, :size//2] = torch.linspace(1, 0, size//2).unsqueeze(0).repeat(size//2, 1)
    
    # Combined (bottom-right)
    img[0, :, size//2:, size//2:] = 0.5
    img[0, 0, size//2:, size//2:] = torch.linspace(0, 1, size//2).unsqueeze(0).repeat(size//2, 1) * 0.7
    img[0, 1, size//2:, size//2:] = torch.linspace(0, 1, size//2).unsqueeze(1).repeat(1, size//2) * 0.7
    img[0, 2, size//2:, size//2:] = 0.8
    
    return img


def load_image(image_path, size=224):
    """Load an image from file."""
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    img = tensor[0].permute(1, 2, 0).cpu().numpy()
    img = img.clip(0, 1)
    return img


def compare_brightness(img, output_path='compare_brightness.png'):
    """Compare brightness adjustment: Torchvision vs Triton-Augment."""
    img_gpu = img.cuda()
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, len(factors), hspace=0.15, wspace=0.05)
    
    for i, factor in enumerate(factors):
        # Torchvision
        tv_result = tvF.adjust_brightness(img_gpu, factor)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(tensor_to_numpy(tv_result))
        ax.set_title(f'Factor: {factor:.2f}', fontsize=10)
        ax.set_ylabel('Torchvision', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        # Triton-Augment
        ta_result = F.adjust_brightness(img_gpu, factor)
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(tensor_to_numpy(ta_result))
        ax.set_ylabel('Triton-Augment', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        # Check if they match
        max_diff = torch.abs(tv_result - ta_result).max().item()
        if max_diff < 1e-5:
            ax.text(0.5, -0.1, '✓ Match', ha='center', va='top', transform=ax.transAxes,
                   fontsize=9, color='green', weight='bold')
    
    plt.suptitle('Brightness Comparison: Torchvision vs Triton-Augment', fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def compare_contrast(img, output_path='compare_contrast.png'):
    """Compare contrast: Torchvision vs Triton (exact) vs Triton (fast)."""
    img_gpu = img.cuda()
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(3, len(factors), hspace=0.15, wspace=0.05)
    
    for i, factor in enumerate(factors):
        # Torchvision
        tv_result = tvF.adjust_contrast(img_gpu, factor)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(tensor_to_numpy(tv_result))
        ax.set_title(f'Factor: {factor:.2f}', fontsize=10)
        ax.set_ylabel('Torchvision', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        # Triton-Augment (exact match)
        ta_exact = F.adjust_contrast(img_gpu, factor)
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(tensor_to_numpy(ta_exact))
        ax.set_ylabel('Triton (Exact)', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        max_diff = torch.abs(tv_result - ta_exact).max().item()
        if max_diff < 1e-5:
            ax.text(0.5, -0.1, '✓ Match', ha='center', va='top', transform=ax.transAxes,
                   fontsize=9, color='green', weight='bold')
        
        # Triton-Augment (fast - different!)
        ta_fast = F.adjust_contrast_fast(img_gpu, factor)
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(tensor_to_numpy(ta_fast))
        ax.set_ylabel('Triton (Fast)', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        max_diff_fast = torch.abs(tv_result - ta_fast).max().item()
        if max_diff_fast > 0.01:
            ax.text(0.5, -0.1, '✗ Different', ha='center', va='top', transform=ax.transAxes,
                   fontsize=9, color='orange', weight='bold')
    
    plt.suptitle('Contrast Comparison: Fast contrast uses different algorithm for speed', 
                 fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def compare_saturation(img, output_path='compare_saturation.png'):
    """Compare saturation adjustment: Torchvision vs Triton-Augment."""
    img_gpu = img.cuda()
    factors = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, len(factors), hspace=0.15, wspace=0.05)
    
    for i, factor in enumerate(factors):
        # Torchvision
        tv_result = tvF.adjust_saturation(img_gpu, factor)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(tensor_to_numpy(tv_result))
        label = 'Grayscale' if factor == 0.0 else f'Factor: {factor:.1f}'
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Torchvision', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        # Triton-Augment
        ta_result = F.adjust_saturation(img_gpu, factor)
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(tensor_to_numpy(ta_result))
        ax.set_ylabel('Triton-Augment', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        # Check if they match
        max_diff = torch.abs(tv_result - ta_result).max().item()
        if max_diff < 1e-4:
            ax.text(0.5, -0.1, '✓ Match', ha='center', va='top', transform=ax.transAxes,
                   fontsize=9, color='green', weight='bold')
    
    plt.suptitle('Saturation Comparison: Torchvision vs Triton-Augment', fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def compare_crop(img, output_path='compare_crop.png'):
    """Compare crop: Torchvision vs Triton-Augment."""
    img_gpu = img.cuda()
    _, _, h, w = img_gpu.shape
    crop_size = h // 2
    
    # Different crop positions
    crops = [
        ('Top-Left', 0, 0),
        ('Center', (h - crop_size) // 2, (w - crop_size) // 2),
        ('Bottom-Right', h - crop_size, w - crop_size),
    ]
    
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, len(crops), hspace=0.15, wspace=0.05)
    
    for i, (name, top, left) in enumerate(crops):
        # Torchvision
        tv_result = tvF.crop(img_gpu, top, left, crop_size, crop_size)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(tensor_to_numpy(tv_result))
        ax.set_title(name, fontsize=10)
        ax.set_ylabel('Torchvision', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        # Triton-Augment
        ta_result = F.crop(img_gpu, top, left, crop_size, crop_size)
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(tensor_to_numpy(ta_result))
        ax.set_ylabel('Triton-Augment', fontsize=10, rotation=0, ha='right', va='center')
        ax.axis('off')
        
        # Check if they match
        if torch.allclose(tv_result, ta_result):
            ax.text(0.5, -0.1, '✓ Identical', ha='center', va='top', transform=ax.transAxes,
                   fontsize=9, color='green', weight='bold')
    
    plt.suptitle(f'Crop Comparison: Torchvision vs Triton-Augment ({crop_size}×{crop_size})', 
                 fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def compare_flip(img, output_path='compare_flip.png'):
    """Compare horizontal flip: Torchvision vs Triton-Augment."""
    img_gpu = img.cuda()
    
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, hspace=0.15, wspace=0.1)
    
    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tensor_to_numpy(img_gpu))
    ax.set_title('Original', fontsize=11)
    ax.set_ylabel('Torchvision', fontsize=10, rotation=0, ha='right', va='center')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(tensor_to_numpy(img_gpu))
    ax.set_ylabel('Triton-Augment', fontsize=10, rotation=0, ha='right', va='center')
    ax.axis('off')
    
    # Flipped
    tv_result = tvF.horizontal_flip(img_gpu)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tensor_to_numpy(tv_result))
    ax.set_title('Horizontal Flip', fontsize=11)
    ax.axis('off')
    
    ta_result = F.horizontal_flip(img_gpu)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(tensor_to_numpy(ta_result))
    ax.axis('off')
    
    if torch.allclose(tv_result, ta_result):
        ax.text(0.5, -0.1, '✓ Identical', ha='center', va='top', transform=ax.transAxes,
               fontsize=9, color='green', weight='bold')
    
    plt.suptitle('Horizontal Flip Comparison: Torchvision vs Triton-Augment', fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def compare_normalize(img, output_path='compare_normalize.png'):
    """Compare normalization: Torchvision vs Triton-Augment."""
    img_gpu = img.cuda()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Apply normalization
    tv_result = tvF.normalize(img_gpu, mean, std)
    ta_result = F.normalize(img_gpu, mean, std)
    
    # Denormalize for visualization
    mean_t = torch.tensor(mean, device='cuda').view(1, 3, 1, 1)
    std_t = torch.tensor(std, device='cuda').view(1, 3, 1, 1)
    
    tv_denorm = tv_result * std_t + mean_t
    ta_denorm = ta_result * std_t + mean_t
    
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, hspace=0.15, wspace=0.1)
    
    # Before normalization
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tensor_to_numpy(img_gpu))
    ax.set_title('Before Normalize', fontsize=11)
    ax.set_ylabel('Torchvision', fontsize=10, rotation=0, ha='right', va='center')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(tensor_to_numpy(img_gpu))
    ax.set_ylabel('Triton-Augment', fontsize=10, rotation=0, ha='right', va='center')
    ax.axis('off')
    
    # After normalization (denormalized for visualization)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tensor_to_numpy(tv_denorm))
    ax.set_title('After Normalize\n(denormalized for display)', fontsize=11)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(tensor_to_numpy(ta_denorm))
    ax.axis('off')
    
    max_diff = torch.abs(tv_result - ta_result).max().item()
    if max_diff < 1e-5:
        ax.text(0.5, -0.1, f'✓ Match (max diff: {max_diff:.2e})', ha='center', va='top', 
               transform=ax.transAxes, fontsize=9, color='green', weight='bold')
    
    plt.suptitle('Normalize Comparison: Torchvision vs Triton-Augment', fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def compare_fused_pipeline(img, output_path='compare_fused_pipeline.png'):
    """Compare full pipeline: Torchvision Compose vs Triton Fused."""
    img_gpu = img.cuda()
    _, _, h, w = img_gpu.shape
    crop_size = h // 2
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Torchvision Compose (no contrast for exact match)
    tv_result = tvF.crop(img_gpu, h//4, w//4, crop_size, crop_size)
    tv_result = tvF.horizontal_flip(tv_result)
    tv_result = tvF.adjust_brightness(tv_result, 1.2)
    tv_result = tvF.adjust_saturation(tv_result, 1.3)
    tv_result = tvF.normalize(tv_result, mean, std)
    
    # Triton-Augment Ultimate Fusion (1 kernel!)
    ta_result = F.ultimate_fused_augment(
        img_gpu,
        top=h//4,
        left=w//4,
        height=crop_size,
        width=crop_size,
        flip_horizontal=True,
        brightness_factor=1.2,
        contrast_factor=1.0,  # Skip contrast for exact match
        saturation_factor=1.3,
        mean=mean,
        std=std,
    )
    
    # Denormalize for visualization
    mean_t = torch.tensor(mean, device='cuda').view(1, 3, 1, 1)
    std_t = torch.tensor(std, device='cuda').view(1, 3, 1, 1)
    
    tv_denorm = tv_result * std_t + mean_t
    ta_denorm = ta_result * std_t + mean_t
    
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 3, hspace=0.15, wspace=0.1)
    
    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tensor_to_numpy(img_gpu))
    ax.set_title('Original', fontsize=11)
    ax.set_ylabel('Torchvision\nCompose\n(5 kernels)', fontsize=9, rotation=0, ha='right', va='center')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(tensor_to_numpy(img_gpu))
    ax.set_ylabel('Triton-Augment\nUltimate Fusion\n(1 kernel!)', fontsize=9, rotation=0, ha='right', va='center')
    ax.axis('off')
    
    # After augmentation (denormalized)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tensor_to_numpy(tv_denorm))
    ax.set_title('After Pipeline\n(denormalized)', fontsize=11)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(tensor_to_numpy(ta_denorm))
    ax.axis('off')
    
    # Difference heatmap
    diff = torch.abs(tv_result - ta_result).mean(dim=1, keepdim=True)
    ax = fig.add_subplot(gs[:, 2])
    im = ax.imshow(diff[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=0.01)
    ax.set_title('Absolute Difference\n(magnified 100x)', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    max_diff = torch.abs(tv_result - ta_result).max().item()
    ax.text(0.5, -0.1, f'Max diff: {max_diff:.2e}', ha='center', va='top', 
           transform=ax.transAxes, fontsize=9)
    
    if max_diff < 1e-5:
        fig.text(0.5, 0.02, '✓ Results are identical! Triton-Augment is faster with 1 kernel vs 5 kernels', 
                ha='center', fontsize=12, color='green', weight='bold')
    
    plt.suptitle('Full Pipeline: Crop → Flip → Brightness → Saturation → Normalize', 
                 fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all comparison visualizations."""
    parser = argparse.ArgumentParser(description='Compare Triton-Augment with torchvision')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (if not provided, uses generated image)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--size', type=int, default=224,
                        help='Image size')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TRITON_AVAILABLE or not TORCHVISION_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("\n❌ Missing dependencies!")
        print("Install with: pip install matplotlib pillow torchvision")
        return
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA is not available")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load or create image
    if args.image:
        print(f"Loading image: {args.image}")
        img = load_image(args.image, size=args.size)
    else:
        print("Creating sample image...")
        img = create_sample_image(size=args.size)
    
    print(f"\nGenerating comparison visualizations in: {output_dir}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    # Generate all comparisons
    print("\n📊 Comparing Operations:")
    compare_brightness(img, output_dir / 'compare_brightness.png')
    compare_contrast(img, output_dir / 'compare_contrast.png')
    compare_saturation(img, output_dir / 'compare_saturation.png')
    compare_crop(img, output_dir / 'compare_crop.png')
    compare_flip(img, output_dir / 'compare_flip.png')
    compare_normalize(img, output_dir / 'compare_normalize.png')
    
    print("\n🚀 Comparing Full Pipeline:")
    compare_fused_pipeline(img, output_dir / 'compare_fused_pipeline.png')
    
    print("\n" + "="*70)
    print(f"✓ All comparisons saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('compare_*.png')):
        print(f"  - {file.name}")
    print("\n💡 Open the images to see that Triton-Augment matches torchvision exactly!")
    print("   (Except fast contrast, which is intentionally different for performance)")


if __name__ == '__main__':
    main()

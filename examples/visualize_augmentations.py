"""
Visualization script to demonstrate the effects of different augmentations.

This script applies various augmentation parameters and saves comparison images.
Requires: pillow and matplotlib
"""

import torch
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    import torchvision.transforms as T
    PILLOW_AVAILABLE = True
except ImportError:
    print("Warning: pillow/torchvision not installed. Install with: pip install pillow torchvision")
    PILLOW_AVAILABLE = False

try:
    import triton_augment as ta
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
    if not PILLOW_AVAILABLE:
        return None
    
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    # Remove batch dimension and convert to HWC format
    img = tensor[0].permute(1, 2, 0).cpu().numpy()
    # Clip to valid range
    img = img.clip(0, 1)
    return img


def visualize_brightness_effects(img, output_path='brightness_comparison.png'):
    """Visualize different brightness levels."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create visualization without matplotlib")
        return
    
    img = img.cuda()
    brightness_values = [-0.3, -0.15, 0.0, 0.15, 0.3]
    
    fig = plt.figure(figsize=(15, 3))
    gs = gridspec.GridSpec(1, len(brightness_values), hspace=0.05, wspace=0.05)
    
    for i, brightness in enumerate(brightness_values):
        result = ta.apply_brightness(img, brightness_factor=brightness)
        
        ax = fig.add_subplot(gs[i])
        ax.imshow(tensor_to_numpy(result))
        ax.set_title(f'Brightness: {brightness:+.2f}')
        ax.axis('off')
    
    plt.suptitle('Brightness Adjustment Effects', fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_contrast_effects(img, output_path='contrast_comparison.png'):
    """Visualize different contrast levels."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create visualization without matplotlib")
        return
    
    img = img.cuda()
    contrast_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    fig = plt.figure(figsize=(15, 3))
    gs = gridspec.GridSpec(1, len(contrast_values), hspace=0.05, wspace=0.05)
    
    for i, contrast in enumerate(contrast_values):
        result = ta.apply_contrast(img, contrast_factor=contrast)
        
        ax = fig.add_subplot(gs[i])
        ax.imshow(tensor_to_numpy(result))
        ax.set_title(f'Contrast: {contrast:.2f}')
        ax.axis('off')
    
    plt.suptitle('Contrast Adjustment Effects', fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_saturation_effects(img, output_path='saturation_comparison.png'):
    """Visualize different saturation levels."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create visualization without matplotlib")
        return
    
    img = img.cuda()
    saturation_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    fig = plt.figure(figsize=(15, 3))
    gs = gridspec.GridSpec(1, len(saturation_values), hspace=0.05, wspace=0.05)
    
    for i, saturation in enumerate(saturation_values):
        result = ta.fused_color_jitter(
            img,
            brightness_factor=0.0,
            contrast_factor=1.0,
            saturation_factor=saturation
        )
        
        ax = fig.add_subplot(gs[i])
        ax.imshow(tensor_to_numpy(result))
        label = 'Grayscale' if saturation == 0.0 else f'Saturation: {saturation:.1f}'
        ax.set_title(label)
        ax.axis('off')
    
    plt.suptitle('Saturation Adjustment Effects', fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_combined_effects(img, output_path='combined_comparison.png'):
    """Visualize combined color jitter effects."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create visualization without matplotlib")
        return
    
    img = img.cuda()
    
    # Different augmentation configurations
    configs = [
        {'name': 'Original', 'b': 0.0, 'c': 1.0, 's': 1.0},
        {'name': 'Bright & Saturated', 'b': 0.2, 'c': 1.1, 's': 1.3},
        {'name': 'Dark & Low Contrast', 'b': -0.2, 'c': 0.7, 's': 0.8},
        {'name': 'High Contrast', 'b': 0.0, 'c': 1.5, 's': 1.2},
        {'name': 'Desaturated', 'b': 0.0, 'c': 1.0, 's': 0.3},
    ]
    
    fig = plt.figure(figsize=(15, 3))
    gs = gridspec.GridSpec(1, len(configs), hspace=0.05, wspace=0.05)
    
    for i, config in enumerate(configs):
        if config['name'] == 'Original':
            result = img
        else:
            result = ta.fused_color_jitter(
                img,
                brightness_factor=config['b'],
                contrast_factor=config['c'],
                saturation_factor=config['s']
            )
        
        ax = fig.add_subplot(gs[i])
        ax.imshow(tensor_to_numpy(result))
        ax.set_title(config['name'], fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Combined Color Jitter Effects', fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_random_augmentations(img, output_path='random_augmentations.png', n=6):
    """Visualize multiple random augmentations."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create visualization without matplotlib")
        return
    
    img = img.cuda()
    
    transform = ta.TritonColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    )
    
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(2, n // 2, hspace=0.1, wspace=0.05)
    
    for i in range(n):
        result = transform(img)
        
        row = i // (n // 2)
        col = i % (n // 2)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(tensor_to_numpy(result))
        ax.set_title(f'Random Sample {i+1}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Random Color Jitter Samples (brightness=0.3, contrast=0.3, saturation=0.3)', 
                 fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize Triton-Augment effects')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (if not provided, uses generated image)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--size', type=int, default=224,
                        help='Image size')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TRITON_AVAILABLE:
        print("Error: triton_augment is required")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for visualizations")
        return
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load or create image
    if args.image:
        if not PILLOW_AVAILABLE:
            print("Error: pillow is required to load images")
            return
        print(f"Loading image: {args.image}")
        img = load_image(args.image, size=args.size)
        if img is None:
            print("Failed to load image")
            return
    else:
        print("Creating sample image...")
        img = create_sample_image(size=args.size)
    
    print(f"\nGenerating visualizations in: {output_dir}")
    print("="*60)
    
    # Generate visualizations
    visualize_brightness_effects(img, output_dir / 'brightness_comparison.png')
    visualize_contrast_effects(img, output_dir / 'contrast_comparison.png')
    visualize_saturation_effects(img, output_dir / 'saturation_comparison.png')
    visualize_combined_effects(img, output_dir / 'combined_comparison.png')
    visualize_random_augmentations(img, output_dir / 'random_augmentations.png')
    
    print("="*60)
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("\nVisualization files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()


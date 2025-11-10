"""
Basic usage examples for Triton-Augment.

This script demonstrates the main features of the library with simple examples.
"""

import torch
import triton_augment as ta


def example_1_basic_color_jitter():
    """Example 1: Basic color jitter without normalization."""
    print("\n" + "="*60)
    print("Example 1: Basic Color Jitter")
    print("="*60)
    
    # Create a batch of random images on GPU
    images = torch.rand(4, 3, 224, 224, device='cuda')
    print(f"Input shape: {images.shape}")
    print(f"Input range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Create the transform
    transform = ta.TritonColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    )
    
    # Apply transformation
    augmented = transform(images)
    print(f"Output shape: {augmented.shape}")
    print(f"Output range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    print("✓ Color jitter applied successfully!")


def example_2_normalization():
    """Example 2: Normalization only."""
    print("\n" + "="*60)
    print("Example 2: Normalization")
    print("="*60)
    
    # Create images
    images = torch.rand(2, 3, 256, 256, device='cuda')
    print(f"Input shape: {images.shape}")
    print(f"Input mean: {images.mean():.3f}")
    print(f"Input std: {images.std():.3f}")
    
    # Create normalization transform (ImageNet stats)
    normalize = ta.TritonNormalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    # Apply normalization
    normalized = normalize(images)
    print(f"Output shape: {normalized.shape}")
    print(f"Output mean: {normalized.mean():.3f}")
    print(f"Output std: {normalized.std():.3f}")
    print("✓ Normalization applied successfully!")


def example_3_fused_transform():
    """Example 3: Fused color jitter + normalization (recommended)."""
    print("\n" + "="*60)
    print("Example 3: Fused Color Jitter + Normalization")
    print("="*60)
    
    # Create images
    images = torch.rand(8, 3, 224, 224, device='cuda')
    print(f"Input shape: {images.shape}")
    
    # Create fused transform (most efficient!)
    transform = ta.TritonColorJitterNormalize(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    print(f"Transform: {transform}")
    
    # Apply transformation
    augmented = transform(images)
    print(f"Output shape: {augmented.shape}")
    print("✓ Fused transformation applied successfully!")
    print("\n💡 This is the recommended approach for best performance!")


def example_4_functional_api():
    """Example 4: Using the functional API directly."""
    print("\n" + "="*60)
    print("Example 4: Functional API")
    print("="*60)
    
    # Create images
    images = torch.rand(4, 3, 224, 224, device='cuda')
    print(f"Input shape: {images.shape}")
    
    # Apply individual operations
    print("\nApplying brightness adjustment...")
    bright = ta.apply_brightness(images, brightness_factor=0.1)
    print(f"  Output range: [{bright.min():.3f}, {bright.max():.3f}]")
    
    print("\nApplying contrast adjustment...")
    contrasted = ta.apply_contrast(bright, contrast_factor=1.2)
    print(f"  Output range: [{contrasted.min():.3f}, {contrasted.max():.3f}]")
    
    print("\nApplying normalization...")
    normalized = ta.apply_normalize(
        contrasted,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    print(f"  Output mean: {normalized.mean():.3f}")
    
    # Or apply everything at once with fused kernel
    print("\nApplying fused operation...")
    fused = ta.fused_color_normalize(
        images,
        brightness_factor=0.1,
        contrast_factor=1.2,
        saturation_factor=1.0,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    print(f"  Output mean: {fused.mean():.3f}")
    print("✓ Functional API demonstrated successfully!")


def example_5_training_pipeline():
    """Example 5: Integration with training pipeline."""
    print("\n" + "="*60)
    print("Example 5: Training Pipeline Integration")
    print("="*60)
    
    # Simulate a training loop
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create augmentation transform
    augment = ta.TritonColorJitterNormalize(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    print("Running simulated training loop...")
    
    # Simulate training loop
    for iteration in range(3):
        # Get batch (simulated)
        images = torch.rand(16, 3, 224, 224, device='cuda')
        labels = torch.randint(0, 10, (16,), device='cuda')
        
        # Apply augmentation on GPU
        images = augment(images)
        
        # Forward pass
        outputs = model(images)
        
        # Dummy loss
        loss = outputs.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Iteration {iteration+1}: Loss = {loss.item():.4f}")
    
    print("✓ Training pipeline integration demonstrated!")


def example_6_custom_parameters():
    """Example 6: Using custom parameter ranges."""
    print("\n" + "="*60)
    print("Example 6: Custom Parameter Ranges")
    print("="*60)
    
    # Create images
    images = torch.rand(4, 3, 224, 224, device='cuda')
    
    # Use custom ranges instead of symmetric ranges
    transform = ta.TritonColorJitter(
        brightness=(-0.3, 0.3),  # Specific range
        contrast=(0.7, 1.3),     # Specific range
        saturation=(0.5, 1.5)    # Specific range
    )
    
    print(f"Transform: {transform}")
    
    # Apply multiple times to show randomness
    print("\nApplying transform 3 times to show randomness:")
    for i in range(3):
        augmented = transform(images)
        print(f"  Run {i+1}: mean={augmented.mean():.3f}, std={augmented.std():.3f}")
    
    print("✓ Custom parameter ranges demonstrated!")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Triton-Augment Usage Examples")
    print("="*60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Triton-Augment version: {ta.__version__}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\n❌ Error: CUDA is not available. These examples require a GPU.")
        return
    
    # Run all examples
    example_1_basic_color_jitter()
    example_2_normalization()
    example_3_fused_transform()
    example_4_functional_api()
    example_5_training_pipeline()
    example_6_custom_parameters()
    
    print("\n" + "="*60)
    print("All examples completed successfully! 🎉")
    print("="*60)
    print("\n💡 Tip: Use TritonColorJitterNormalize for best performance!")
    print("📖 See README.md for more information.")


if __name__ == '__main__':
    main()


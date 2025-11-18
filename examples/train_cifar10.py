"""
Simple CIFAR-10 Training Example with Triton-Augment

This example demonstrates how to use Triton-Augment for maximum performance.

Key Points:
1. Use torchvision for: data loading, ToTensor (CPU with num_workers)
2. Use Triton-Augment in training loop (GPU batch processing)
3. Fast asynchronous data loading + fast GPU augmentation = best performance!

Note: This example uses same random parameters for all images in a batch.
      Per-image randomness within batches is planned for a future release.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import triton_augment as ta
import time


# ============================================================================
# Data Loading - Standard PyTorch Pattern with Workers
# ============================================================================

def get_dataloaders(batch_size=128):
    """
    Load CIFAR-10 dataset with standard PyTorch pattern.
    
    Use num_workers for fast async data loading!
    """
    # Minimal transform: only ToTensor on CPU
    basic_transform = transforms.ToTensor()
    
    train_dataset = datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=basic_transform  # Only ToTensor
    )
    
    test_dataset = datasets.CIFAR10(
        './data',
        train=False,
        download=True,
        transform=basic_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # ✅ Use workers for async data loading!
        pin_memory=True  # Faster CPU-to-GPU transfer
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, train_transform, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move to GPU
        images = images.cuda()
        labels = labels.cuda()
        
        # Apply Triton-Augment on GPU batch
        # All ops fused in 1 kernel! 🚀
        images = train_transform(images)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_time = time.time() - start_time
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy, epoch_time


def test(model, test_loader, test_transform):
    """Evaluate on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            
            # Apply normalization
            images = test_transform(images)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, accuracy


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Configuration
    batch_size = 128
    epochs = 10
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("CIFAR-10 Training with Triton-Augment")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print()
    
    if device.type != 'cuda':
        print("ERROR: CUDA device required for Triton-Augment!")
        return
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_dataloaders(batch_size)
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Using {train_loader.num_workers} workers for async data loading")
    print()
    
    # Create Triton-Augment transforms
    # Applied in training loop on GPU batches!
    train_transform = ta.TritonFusedAugment(
        crop_size=28,                    # Random crop from 32×32 to 28×28
        horizontal_flip_p=0.5,           # 50% chance of horizontal flip
        brightness=0.2,                  # Brightness jitter ±20%
        contrast=0.2,                    # Contrast jitter ±20%
        saturation=0.2,                  # Saturation jitter ±20%
        grayscale_p=0.1,          # 10% chance of grayscale
        mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 mean
        std=(0.2470, 0.2435, 0.2616),    # CIFAR-10 std
        same_on_batch=False        # Each image gets different random params
    )
    
    test_transform = ta.TritonNormalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
    
    print("✓ Augmentation pipeline:")
    print("  - Data loading: CPU with 4 workers (async, fast!)")
    print("  - Augmentation: GPU in training loop (batched, fused!)")
    print("  - All Triton ops fused in 1 kernel per batch! 🚀")
    print("  - Per-image randomness: Each image gets different random params!")
    print()
    
    # Create model
    print("Creating model...")
    model = models.resnet18(num_classes=10)
    model = model.cuda()
    print("✓ Model: ResNet-18")
    print()
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()
    
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, train_transform, epoch
        )
        
        # Test
        test_loss, test_acc = test(model, test_loader, test_transform)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("  ✓ Saved new best model!")
        
        # Print epoch summary
        print(f"\n  Epoch Summary:")
        print(f"    Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"    Test Loss:  {test_loss:.4f}  Test Acc:  {test_acc:.2f}%")
        print(f"    Time: {epoch_time:.2f}s")
        print(f"    Best Test Acc: {best_acc:.2f}%")
    
    # Final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"✓ Best Test Accuracy: {best_acc:.2f}%")
    print(f"✓ Model saved to: best_model.pth")
    print()
    print("💡 Key Insights:")
    print("   1. Fast async data loading (num_workers=4)")
    print("   2. Fast GPU batch augmentation (all ops fused in 1 kernel)")
    print("   3. Best of both worlds: CPU for I/O, GPU for compute!")
    print("   4. Speedup comes from batch processing + kernel fusion")
    print("=" * 80)


if __name__ == '__main__':
    main()

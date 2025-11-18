"""
Simple MNIST Training Example with Triton-Augment

This example demonstrates how to use Triton-Augment for maximum performance.

Key Points:
1. Use torchvision for: data loading, ToTensor (CPU with num_workers)
2. Use Triton-Augment in training loop (GPU batch processing)
3. Fast asynchronous data loading + fast GPU augmentation = best performance!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import triton_augment as ta
import time


# ============================================================================
# Model Definition
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# ============================================================================
# Data Loading - Standard PyTorch Pattern with Workers
# ============================================================================

def get_dataloaders(batch_size=128):
    """
    Load MNIST dataset with standard PyTorch pattern.
    """
    basic_transform = transforms.ToTensor()
    
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=basic_transform
    )
    
    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=basic_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
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
        images = images.cuda()
        labels = labels.cuda()
        
        # Apply Triton-Augment on GPU batch
        images = train_transform(images)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        
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
    criterion = nn.NLLLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            images = test_transform(images)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, accuracy


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Configuration
    batch_size = 128
    epochs = 5
    lr = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("MNIST Training with Triton-Augment")
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
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_dataloaders(batch_size)
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Using {train_loader.num_workers} workers for async data loading")
    print()
    
    # Create Triton-Augment transforms
    train_transform = ta.TritonFusedAugment(
        crop_size=24,
        horizontal_flip_p=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.0,
        grayscale_p=0.0,
        mean=(0.1307,),
        std=(0.3081,),
        same_on_batch=False  # Each image gets different random params
    )
    
    test_transform = ta.TritonNormalize(
        mean=(0.1307,),
        std=(0.3081,)
    )
    
    print("✓ Augmentation pipeline:")
    print("  - Data loading: CPU with 4 workers (async, fast!)")
    print("  - Augmentation: GPU in training loop (batched, fused!)")
    print("  - All Triton ops fused in 1 kernel per batch! 🚀")
    print()
    
    # Create model
    print("Creating model...")
    model = SimpleCNN().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    print("✓ Model: SimpleCNN")
    print()
    
    # Training loop
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()
    
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 80)
        
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, train_transform, epoch
        )
        
        test_loss, test_acc = test(model, test_loader, test_transform)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_mnist_model.pth')
            print("  ✓ Saved new best model!")
        
        print(f"\n  Epoch Summary:")
        print(f"    Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"    Test Loss:  {test_loss:.4f}  Test Acc:  {test_acc:.2f}%")
        print(f"    Time: {epoch_time:.2f}s")
        print(f"    Best Test Acc: {best_acc:.2f}%")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"✓ Best Test Accuracy: {best_acc:.2f}%")
    print(f"✓ Model saved to: best_mnist_model.pth")
    print("=" * 80)


if __name__ == '__main__':
    main()

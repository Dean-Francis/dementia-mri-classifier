"""
Test script for the training function
This creates a minimal synthetic dataset and runs a training test
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model import DementiaCNN
from train import train_model


def create_synthetic_data(num_samples=40, image_size=224):
    """
    Create synthetic MRI-like data for testing

    Args:
        num_samples: Total samples to create (split 80/20 train/val)
        image_size: Size of square images

    Returns:
        train_loader, val_loader
    """
    print(f"Creating synthetic dataset with {num_samples} samples...")

    # Generate random grayscale images (simulate MRI slices)
    # Shape: [num_samples, 1, 224, 224]
    images = torch.randn(num_samples, 1, image_size, image_size)

    # Generate random binary labels (0=CN, 1=AD)
    labels = torch.randint(0, 2, (num_samples,))

    # Create a TensorDataset
    dataset = TensorDataset(images, labels)

    # 80/20 split
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader


def main():
    print("=" * 60)
    print("Testing Training Function")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create synthetic data
    print("\n--- Creating Synthetic Data ---")
    train_loader, val_loader = create_synthetic_data(num_samples=40, image_size=224)

    # Initialize model
    print("\n--- Initializing Model ---")
    model = DementiaCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n--- Training Model ---")
    print("(Running 2 epochs for quick test)\n")

    try:
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            epochs=2,  # Just 2 epochs for quick test
            device=device
        )

        print("\n--- Training Complete ---")
        print(f"Final Training Loss: {train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {val_losses[-1]:.4f}")

        # Print loss history
        print("\nLoss History:")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            print(f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        print("\n[SUCCESS] Training function works correctly!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

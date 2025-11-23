"""
End-to-end training test with real OASIS MRI data
Uses existing modules: data_extraction, dataset, model, train
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from pathlib import Path
import sys

from src.data_extraction import extract_slices
from src.dataset import MRIDataset
from src.model import DementiaCNN
from src.train import train_model
from src.integrated_gradients import compute_integrated_gradients


def main():
    print("=" * 70)
    print("TRAINING TEST WITH REAL OASIS MRI DATA")
    print("=" * 70)

    # Paths
    root_dir = Path("dataset")
    data_root = root_dir / "disc1"
    output_dir = root_dir / "processed" / "oasis_slices"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")

    # Step 1: Extract slices using data_extraction module
    print("\n" + "=" * 70)
    print("STEP 1: EXTRACTING MRI SLICES")
    print("=" * 70)

    slice_count = extract_slices(data_root, output_dir, cdr_threshold=0)

    # Verify we have data
    total_slices = slice_count['CN'] + slice_count['AD']
    if total_slices < 20:
        print(f"\nERROR: Only {total_slices} slices extracted. Need at least 20 samples.")
        return False

    # Step 2: Load dataset using MRIDataset
    print("\n" + "=" * 70)
    print("STEP 2: LOADING DATASET")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MRIDataset(output_dir, transform=transform)
    print(f"\nTotal images loaded: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No images found!")
        return False

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")

    # Step 3: Initialize model using DementiaCNN
    print("\n" + "=" * 70)
    print("STEP 3: INITIALIZING MODEL")
    print("=" * 70)

    model = DementiaCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: DementiaCNN")
    print(f"Parameters: {total_params:,}")

    # Step 4: Train using train_model function
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING")
    print("=" * 70)
    print("\nRunning training...\n")

    try:
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            epochs=5,  # 5 epochs for testing
            device=device
        )

        # Results
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        print(f"\nFinal Training Loss: {train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {val_losses[-1]:.4f}")

        print("\nLoss History:")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            print(f"  Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        print("\n[SUCCESS] Training with real data completed successfully!")
        
        

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(success)

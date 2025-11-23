"""
Complete pipeline: Extract data -> Train model
Run this script to extract MRI data and train the dementia detection model
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from data_extraction import extract_slices
from dataset import MRIDataset
from model import DementiaCNN
from train import train_model


def main(disc_max=None):
    print("=" * 70)
    print("DEMENTIA DETECTION PIPELINE")
    print("=" * 70)

    # Configuration
    data_root = r'../dataset'  # Will automatically find all disc1, disc2, etc.
    output_dir = '../oasis_slices'
    batch_size = 32  # Increased for larger dataset (disc1-disc12)
    epochs = 50  # With more data, can train longer without overfitting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    if disc_max is not None:
        print(f"Training discs: disc1-disc{disc_max} (disc{disc_max+1}+ will be excluded)")
    print()

    # Step 1: Extract slices
    print("=" * 70)
    print("STEP 1: EXTRACTING MRI SLICES")
    print("=" * 70)

    slice_count = extract_slices(data_root, output_dir, cdr_threshold=0, disc_max=disc_max)

    # Check if we have data
    total_slices = slice_count['CN'] + slice_count['AD']
    if total_slices == 0:
        print("\nERROR: No data extracted! Check your data_root path.")
        return

    # Step 2: Load dataset
    print("\n" + "=" * 70)
    print("STEP 2: LOADING DATASET")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MRIDataset(output_dir, transform=transform, augment=True)
    print(f"\nTotal images: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No images loaded!")
        return

    # Check class distribution
    import numpy as np
    labels = np.array(dataset.labels)
    unique, counts = np.unique(labels, return_counts=True)

    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        label_name = 'CN' if u == 0 else 'AD'
        percentage = (c / len(dataset)) * 100
        print(f"  {label_name}: {c} samples ({percentage:.1f}%)")

    # Get class weights
    class_weights = dataset.get_class_weights()
    print(f"\nClass weights: CN={class_weights[0]:.4f}, AD={class_weights[1]:.4f}")

    # Split data: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTrain set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Step 3: Initialize model
    print("\n" + "=" * 70)
    print("STEP 3: INITIALIZING MODEL")
    print("=" * 70)

    model = DementiaCNN().to(device)
    total_params, trainable_params = model.get_model_size()
    print(f"\nModel: DementiaCNN")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Step 4: Train
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING MODEL")
    print("=" * 70)
    print()

    metrics = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        device=device,
        use_class_weights=True,
        class_weights=class_weights
    )

    # Step 5: Evaluate on test set
    print("\n" + "=" * 70)
    print("STEP 5: TEST SET EVALUATION")
    print("=" * 70)

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    test_acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(test_labels, test_preds)

    print(f"\nTest Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print(f"\nBest epoch: {metrics['best_epoch']}")
    print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
    print(f"Final validation accuracy: {metrics['val_accs'][-1]:.4f}")
    print(f"Final validation F1: {metrics['val_f1s'][-1]:.4f}")
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test F1 score: {f1:.4f}")

    print("\nModel saved to: best_model.pth")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    disc_max = None
    if len(sys.argv) > 1:
        try:
            disc_max = int(sys.argv[1])
            print(f"Command-line argument: disc_max={disc_max}\n")
        except ValueError:
            print(f"Invalid disc_max value: {sys.argv[1]}")
            print("Usage: python run_pipeline.py [disc_max]")
            print("Example: python run_pipeline.py 11  # Train on disc1-disc11")
            sys.exit(1)

    main(disc_max=disc_max)
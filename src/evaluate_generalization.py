"""
Evaluate model generalization on disc3-disc12 data
Tests the model trained on disc1-disc2 against unseen discs
"""

import torch
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
from model import DementiaCNN
import os

def load_model(model_path='best_model.pth'):
    """Load trained model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DementiaCNN()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[OK] Model loaded from {model_path}")
    else:
        print(f"[ERROR] Model not found at {model_path}")
        return None, None

    model.to(device)
    model.eval()
    return model, device

def predict_image(model, device, image_path, transform):
    """Predict on a single image"""
    try:
        image = Image.open(image_path).convert('L')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred_class].item()

        return pred_class, confidence
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def main():
    print("=" * 80)
    print("GENERALIZATION EVALUATION: disc3-disc12 (Unseen Subjects)")
    print("=" * 80)

    # Setup
    device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_root = Path('../dataset')

    print(f"\nDevice: {device_obj}")
    print(f"Dataset root: {dataset_root}\n")

    # Load model
    print("Loading trained model...")
    model, device = load_model('best_model.pth')
    if model is None:
        print("Cannot proceed without model")
        return

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Find disc3-disc12 folders
    disc_folders = sorted([d for d in dataset_root.glob('disc*') if d.is_dir()])
    disc_folders = [d for d in disc_folders if int(d.name[4:]) >= 3]  # Filter for disc3+

    if not disc_folders:
        print("No disc3-disc12 folders found!")
        print(f"Please add disc3, disc4, ... disc12 folders to: {dataset_root}")
        return

    print(f"Found {len(disc_folders)} disc folder(s): {', '.join([d.name for d in disc_folders])}\n")

    # Extract all slices from disc3-disc12
    print("Extracting slices from disc3-disc12...")
    cn_slices = []
    ad_slices = []

    for disc_folder in disc_folders:
        # Look for CN and AD subdirectories
        cn_dir = disc_folder / 'CN'
        ad_dir = disc_folder / 'AD'

        if cn_dir.exists():
            cn_files = list(cn_dir.glob('*.png'))
            cn_slices.extend(cn_files)
            print(f"  {disc_folder.name}/CN: {len(cn_files)} slices")

        if ad_dir.exists():
            ad_files = list(ad_dir.glob('*.png'))
            ad_slices.extend(ad_files)
            print(f"  {disc_folder.name}/AD: {len(ad_files)} slices")

    print(f"\nTotal CN slices (disc3-12): {len(cn_slices)}")
    print(f"Total AD slices (disc3-12): {len(ad_slices)}")
    print(f"Total slices: {len(cn_slices) + len(ad_slices)}\n")

    if len(cn_slices) == 0 and len(ad_slices) == 0:
        print("[ERROR] No slices found in disc3-disc12!")
        print("Make sure you have CN/ and AD/ subdirectories in each disc folder")
        return

    # Evaluate on CN slices
    print("=" * 80)
    print("EVALUATING CN SLICES (Ground Truth: CN)")
    print("=" * 80)

    cn_correct = 0
    cn_predictions = []

    for i, image_path in enumerate(cn_slices):
        pred_class, confidence = predict_image(model, device, image_path, transform)

        if pred_class is not None:
            cn_predictions.append((pred_class, confidence))
            if pred_class == 0:  # Correct (predicted CN)
                cn_correct += 1

    cn_accuracy = 100 * cn_correct / len(cn_slices) if len(cn_slices) > 0 else 0
    print(f"\nCN Accuracy: {cn_correct}/{len(cn_slices)} = {cn_accuracy:.1f}%")
    if len(cn_predictions) > 0:
        avg_cn_conf = np.mean([c for p, c in cn_predictions if p == 0])
        print(f"Average confidence on correct CN predictions: {avg_cn_conf:.3f}")

    # Evaluate on AD slices
    print("\n" + "=" * 80)
    print("EVALUATING AD SLICES (Ground Truth: AD)")
    print("=" * 80)

    ad_correct = 0
    ad_predictions = []

    for i, image_path in enumerate(ad_slices):
        pred_class, confidence = predict_image(model, device, image_path, transform)

        if pred_class is not None:
            ad_predictions.append((pred_class, confidence))
            if pred_class == 1:  # Correct (predicted AD)
                ad_correct += 1

    ad_accuracy = 100 * ad_correct / len(ad_slices) if len(ad_slices) > 0 else 0
    print(f"\nAD Accuracy: {ad_correct}/{len(ad_slices)} = {ad_accuracy:.1f}%")
    if len(ad_predictions) > 0:
        avg_ad_conf = np.mean([c for p, c in ad_predictions if p == 1])
        print(f"Average confidence on correct AD predictions: {avg_ad_conf:.3f}")

    # Overall results
    print("\n" + "=" * 80)
    print("OVERALL RESULTS: disc3-disc12 (Out-of-Distribution)")
    print("=" * 80)

    total_correct = cn_correct + ad_correct
    total_slices = len(cn_slices) + len(ad_slices)
    overall_accuracy = 100 * total_correct / total_slices if total_slices > 0 else 0

    print(f"\nTotal Accuracy: {total_correct}/{total_slices} = {overall_accuracy:.1f}%")
    print(f"CN Accuracy:   {cn_accuracy:.1f}%")
    print(f"AD Accuracy:   {ad_accuracy:.1f}%")

    # Confusion matrix
    print("\n" + "Confusion Matrix (disc3-disc12):")
    tn = sum(1 for p, _ in cn_predictions if p == 0)  # CN predicted as CN
    fp = sum(1 for p, _ in cn_predictions if p == 1)  # CN predicted as AD
    fn = sum(1 for p, _ in ad_predictions if p == 0)  # AD predicted as CN
    tp = sum(1 for p, _ in ad_predictions if p == 1)  # AD predicted as AD

    print(f"  True Negatives:  {tn} (CN -> CN)")
    print(f"  False Positives: {fp} (CN -> AD)")
    print(f"  False Negatives: {fn} (AD -> CN)")
    print(f"  True Positives:  {tp} (AD -> AD)")

    # Calculate precision and recall
    if tp + fp > 0:
        precision = 100 * tp / (tp + fp)
        print(f"\nPrecision (AD): {precision:.1f}%")

    if tp + fn > 0:
        recall = 100 * tp / (tp + fn)
        print(f"Recall (AD):    {recall:.1f}%")

    # Compare with training set
    print("\n" + "=" * 80)
    print("COMPARISON: Training vs Out-of-Distribution")
    print("=" * 80)
    print(f"Training (disc1-2) Accuracy: 92.5%")
    print(f"Test (disc3-12)    Accuracy: {overall_accuracy:.1f}%")
    print(f"Difference:                  {overall_accuracy - 92.5:.1f}%")

    if overall_accuracy >= 80:
        print("\n[GOOD] Model generalizes well to unseen subjects!")
    elif overall_accuracy >= 70:
        print("\n[FAIR] Model shows reasonable generalization")
    else:
        print("\n[POOR] Model does not generalize well to unseen subjects")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

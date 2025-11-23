"""
Evaluate model generalization on disc12 (held-out test set)
Tests the model trained on disc1-disc11 against completely unseen subjects
"""

import torch
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
from model import DementiaCNN
from data_extraction import extract_slices
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
    print("GENERALIZATION EVALUATION: disc12 (Out-of-Distribution Test)")
    print("=" * 80)

    # Setup
    device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_root = '../dataset'
    output_dir = '../oasis_slices_disc12'  # Separate directory for disc12 slices

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

    # Extract disc12 slices
    print("=" * 80)
    print("STEP 1: EXTRACTING disc12 SLICES")
    print("=" * 80)
    print()

    # Extract only disc12 (disc_min=12 means start from disc 12)
    extract_slices(dataset_root, output_dir, cdr_threshold=0, disc_min=12)
    print()

    # Now load the extracted slices
    print("=" * 80)
    print("STEP 2: EVALUATING ON disc12 SLICES")
    print("=" * 80)
    print()

    cn_slices = []
    ad_slices = []

    # Look for CN and AD slices in output directory
    cn_dir = Path(output_dir) / 'CN'
    ad_dir = Path(output_dir) / 'AD'

    if cn_dir.exists():
        cn_files = list(cn_dir.glob('*.png'))
        cn_slices.extend(sorted(cn_files))
        print(f"Found {len(cn_files)} CN slices")

    if ad_dir.exists():
        ad_files = list(ad_dir.glob('*.png'))
        ad_slices.extend(sorted(ad_files))
        print(f"Found {len(ad_files)} AD slices")

    print(f"\nTotal CN slices (disc12): {len(cn_slices)}")
    print(f"Total AD slices (disc12): {len(ad_slices)}")
    print(f"Total slices: {len(cn_slices) + len(ad_slices)}\n")

    if len(cn_slices) == 0 and len(ad_slices) == 0:
        print("[ERROR] No slices found in disc12!")
        print("Make sure you have CN/ and AD/ subdirectories in disc12")
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
    print("OVERALL RESULTS: disc12 (Out-of-Distribution)")
    print("=" * 80)

    total_correct = cn_correct + ad_correct
    total_slices = len(cn_slices) + len(ad_slices)
    overall_accuracy = 100 * total_correct / total_slices if total_slices > 0 else 0

    print(f"\nTotal Accuracy: {total_correct}/{total_slices} = {overall_accuracy:.1f}%")
    print(f"CN Accuracy:   {cn_accuracy:.1f}%")
    print(f"AD Accuracy:   {ad_accuracy:.1f}%")

    # Confusion matrix
    print("\n" + "Confusion Matrix (disc12):")
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

    # Generalization assessment
    print("\n" + "=" * 80)
    print("GENERALIZATION ASSESSMENT")
    print("=" * 80)
    print(f"Model trained on: disc1-disc11")
    print(f"Model tested on:  disc12 (completely held-out subjects)")
    print(f"\nDisc12 Accuracy: {overall_accuracy:.1f}%")

    if overall_accuracy >= 80:
        print("\n[GOOD] Model generalizes well to completely unseen subjects!")
    elif overall_accuracy >= 70:
        print("\n[FAIR] Model shows reasonable generalization to new subjects")
    elif overall_accuracy >= 60:
        print("\n[MODERATE] Model shows some generalization but with noticeable performance drop")
    else:
        print("\n[POOR] Model does not generalize well to completely unseen subjects")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

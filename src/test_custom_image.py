"""
Test the trained model on custom MRI slice images
Place your test images in: ../test_images/
Supported formats: .png, .jpg, .jpeg
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from model import DementiaCNN
import os

def predict_image(image_path, model, device, transform):
    """
    Predict whether an MRI slice shows CN (Cognitively Normal) or AD (Alzheimer's Disease)

    Args:
        image_path: Path to the MRI slice image
        model: Trained DementiaCNN model
        device: torch device (cpu or cuda)
        transform: Image transforms

    Returns:
        prediction: 'CN' or 'AD'
        confidence: Confidence score (0-1)
    """
    # Load and transform image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probs[0, prediction].item()

    # Map to label
    label = 'CN' if prediction == 0 else 'AD'
    return label, confidence


def main():
    print("=" * 70)
    print("CUSTOM IMAGE PREDICTION")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_images_dir = '../test_images'

    print(f"\nDevice: {device}")
    print(f"Looking for images in: {test_images_dir}\n")

    # Check if directory exists and has images
    if not os.path.exists(test_images_dir):
        print(f"ERROR: {test_images_dir} directory not found!")
        print("Please create the directory and place your MRI slice images there.")
        return

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(Path(test_images_dir).glob(ext)))

    if not image_files:
        print(f"No images found in {test_images_dir}")
        print("Please add .png, .jpg, or .jpeg files to the folder.")
        return

    # Load trained model
    print("Loading trained model...")
    model = DementiaCNN().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    print("Model loaded successfully!\n")

    # Transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Make predictions
    print("=" * 70)
    print(f"Found {len(image_files)} image(s) to test")
    print("=" * 70)

    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {image_path.name}")
        print("-" * 70)

        try:
            label, confidence = predict_image(image_path, model, device, transform)
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

            # Explain prediction
            if label == 'CN':
                print("Interpretation: Model indicates COGNITIVELY NORMAL brain")
            else:
                print("Interpretation: Model indicates ALZHEIMER'S DISEASE signs detected")

        except Exception as e:
            print(f"ERROR processing {image_path.name}: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

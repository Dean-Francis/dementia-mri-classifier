"""
Integrated Gradients for model interpretability and deletion metric analysis
Measures feature importance by computing gradients along a path from baseline to input
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from model import DementiaCNN
import os


class IntegratedGradients:
    """
    Compute integrated gradients for model interpretability.

    Integrated Gradients explains predictions by computing gradients along a straight line
    path from a baseline (zero image) to the input image.
    """

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: PyTorch model
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def integrated_gradients(self, input_image, target_class, baseline=None, steps=50):
        """
        Compute integrated gradients for an input image.

        Args:
            input_image: Input image tensor [1, C, H, W]
            target_class: Target class index (0=CN, 1=AD)
            baseline: Baseline image (default: black image)
            steps: Number of integration steps

        Returns:
            integrated_grads: Attribution map [1, C, H, W]
            path_logits: Model outputs along the path (for analysis)
        """
        if baseline is None:
            baseline = torch.zeros_like(input_image)

        # Generate interpolation path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(self.device)
        path_images = []
        for alpha in alphas:
            path_img = baseline + alpha * (input_image - baseline)
            path_images.append(path_img)

        path_images = torch.cat(path_images, dim=0)  # [steps, C, H, W]

        # Compute gradients for each step
        path_images.requires_grad_(True)
        outputs = self.model(path_images)  # [steps, num_classes]

        # Get target class logits
        target_logits = outputs[:, target_class]

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=target_logits.sum(),
            inputs=path_images,
            create_graph=False
        )[0]  # [steps, C, H, W]

        # Integrate gradients using trapezoidal rule
        # Compute integral manually for compatibility
        integrated_grads = torch.zeros_like(grads[0]).unsqueeze(0)
        for i in range(steps - 1):
            integrated_grads += (grads[i] + grads[i + 1]) / 2 * (alphas[i + 1] - alphas[i])
        # Result shape: [1, C, H, W]

        # Multiply by (input - baseline)
        integrated_grads *= (input_image - baseline)

        return integrated_grads, outputs.detach()

    def compute_attribution_map(self, input_image, target_class, steps=50):
        """
        Compute attribution map showing which pixels influence the prediction.

        Args:
            input_image: Input image tensor [1, C, H, W]
            target_class: Target class index
            steps: Number of integration steps

        Returns:
            attribution_map: Absolute value of integrated gradients [1, H, W]
        """
        integrated_grads, _ = self.integrated_gradients(
            input_image, target_class, steps=steps
        )

        # Convert to attribution map (take absolute value, average over channels)
        attribution = torch.abs(integrated_grads).mean(dim=1)  # [1, H, W]
        attribution = attribution.squeeze(0)  # [H, W]

        return attribution

    def deletion_metric(self, input_image, target_class, mask, steps=50):
        """
        Deletion metric: measure how much prediction changes when removing important features.

        High score = model relies heavily on these features = model makes correct decisions
        based on important regions

        Args:
            input_image: Input image [1, C, H, W]
            target_class: Predicted class
            mask: Binary mask of important regions [H, W]
            steps: Integration steps (higher = more accurate)

        Returns:
            deletion_score: How much confidence drops when removing masked region
        """
        # Get baseline confidence and logit
        with torch.no_grad():
            baseline_logits = self.model(input_image)
            baseline_probs = F.softmax(baseline_logits, dim=1)
            baseline_confidence = baseline_probs[0, target_class].item()
            baseline_logit = baseline_logits[0, target_class].item()

        # Expand mask to match input dimensions
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]

        # Create perturbed image with masked region set to black (normalized value ~-1)
        perturbed_image = input_image.clone()
        perturbed_image = perturbed_image * (1 - mask_expanded)  # Zero out masked region

        # Get confidence on perturbed image
        with torch.no_grad():
            perturbed_logits = self.model(perturbed_image)
            perturbed_probs = F.softmax(perturbed_logits, dim=1)
            perturbed_confidence = perturbed_probs[0, target_class].item()
            perturbed_logit = perturbed_logits[0, target_class].item()

        # Deletion score: how much confidence drops (can use either confidence or logits)
        deletion_score = baseline_confidence - perturbed_confidence

        return deletion_score, baseline_confidence, perturbed_confidence

    def insertion_metric(self, input_image, target_class, mask, steps=50):
        """
        Insertion metric: measure how much prediction improves when adding important features.

        High score = these features are important for the prediction

        Args:
            input_image: Input image [1, C, H, W]
            target_class: Target class
            mask: Binary mask of important regions [H, W]

        Returns:
            insertion_score: How much confidence increases by keeping only masked region
        """
        # Get baseline confidence (black image)
        black_image = torch.zeros_like(input_image)
        with torch.no_grad():
            black_logits = self.model(black_image)
            black_probs = F.softmax(black_logits, dim=1)
            baseline_confidence = black_probs[0, target_class].item()

        # Expand mask
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]

        # Create image with only masked region visible
        masked_image = input_image.clone() * mask_expanded

        # Get confidence on masked image
        with torch.no_grad():
            masked_logits = self.model(masked_image)
            masked_probs = F.softmax(masked_logits, dim=1)
            masked_confidence = masked_probs[0, target_class].item()

        # Insertion score: how much improvement
        insertion_score = masked_confidence - baseline_confidence

        return insertion_score, baseline_confidence, masked_confidence


def load_image(image_path, device='cpu'):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('L')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def load_model(model_path='best_model.pth', device='cpu'):
    """Load trained model"""
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
        return None

    return model.to(device)


def visualize_attribution(attribution_map, output_path='attribution.png'):
    """Save attribution map as image (for visualization)"""
    # Normalize to [0, 1]
    attr_normalized = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)
    attr_np = (attr_normalized.cpu().numpy() * 255).astype(np.uint8)

    img = Image.fromarray(attr_np, mode='L')
    img.save(output_path)
    print(f"Attribution map saved to {output_path}")


def main():
    print("=" * 80)
    print("INTEGRATED GRADIENTS: MODEL INTERPRETABILITY & DELETION METRIC")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")

    # Load model
    print("Loading model...")
    model = load_model('best_model.pth', device=device)
    if model is None:
        return

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model, device=device)
    print("Integrated Gradients initialized\n")

    # Test on sample image
    test_image_path = r'../test_images'
    image_files = list(Path(test_image_path).glob('*.png'))

    if not image_files:
        print(f"No test images found in {test_image_path}")
        return

    # Process first image
    image_path = image_files[0]
    print(f"Processing: {image_path.name}")
    print("-" * 80)

    # Load image
    input_image = load_image(str(image_path), device=device)

    # Get model prediction
    with torch.no_grad():
        logits = model(input_image)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_class].item()

    class_names = ['CN', 'AD']
    print(f"Prediction: {class_names[pred_class]}")
    print(f"Confidence: {confidence:.4f}\n")

    # Compute Integrated Gradients
    print("Computing Integrated Gradients...")
    attribution_map = ig.compute_attribution_map(input_image, pred_class, steps=50)
    print(f"Attribution map shape: {attribution_map.shape}")
    print(f"Attribution map range: [{attribution_map.min():.4f}, {attribution_map.max():.4f}]\n")

    # Create importance mask (top 25% of pixels by attribution)
    threshold = torch.quantile(attribution_map, 0.75)
    importance_mask = (attribution_map > threshold).float()
    important_pixels = importance_mask.sum().item()
    total_pixels = importance_mask.numel()
    print(f"Important pixels (top 25%): {int(important_pixels)}/{total_pixels} ({100*important_pixels/total_pixels:.1f}%)\n")

    # Deletion Metric: How much does confidence drop when removing important regions?
    print("=" * 80)
    print("DELETION METRIC")
    print("=" * 80)
    print("Removes important features and measures confidence drop\n")

    deletion_score, baseline_conf, deletion_conf = ig.deletion_metric(
        input_image, pred_class, importance_mask, steps=50
    )

    print(f"Baseline confidence (full image):     {baseline_conf:.4f}")
    print(f"Confidence after deletion:            {deletion_conf:.4f}")
    print(f"Deletion score (confidence drop):     {deletion_score:.4f}")
    print(f"Deletion rate:                        {100*deletion_score/baseline_conf:.1f}%\n")

    if deletion_score > 0.3:
        print("[HIGH] Model heavily relies on these important regions")
        print("       This indicates the model is using relevant features for prediction")
    elif deletion_score > 0.1:
        print("[MODERATE] Model moderately relies on these regions")
    else:
        print("[LOW] Model does not heavily rely on these regions")
        print("      Model may be learning other patterns\n")

    # Insertion Metric: How much does confidence increase when only keeping important regions?
    print("\n" + "=" * 80)
    print("INSERTION METRIC")
    print("=" * 80)
    print("Keeps only important features and measures confidence gain\n")

    insertion_score, black_conf, masked_conf = ig.insertion_metric(
        input_image, pred_class, importance_mask
    )

    print(f"Confidence on black image:            {black_conf:.4f}")
    print(f"Confidence with only important regions: {masked_conf:.4f}")
    print(f"Insertion score (confidence gain):   {insertion_score:.4f}")
    print(f"Insertion rate:                      {100*insertion_score/(confidence-black_conf) if (confidence-black_conf) > 0 else 0:.1f}%\n")

    if insertion_score > 0.3:
        print("[HIGH] Important regions are sufficient for prediction")
        print("       Model makes good decisions based on these features alone")
    elif insertion_score > 0.1:
        print("[MODERATE] Important regions contribute significantly")
    else:
        print("[LOW] Important regions alone are not sufficient")
        print("      Model may need other context\n")

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION SUMMARY")
    print("=" * 80)
    print(f"Image: {image_path.name}")
    print(f"Prediction: {class_names[pred_class]} ({confidence:.1%})")
    print(f"\nDeletion Score: {deletion_score:.4f} ({100*deletion_score/baseline_conf:.1f}% drop)")
    print(f"Insertion Score: {insertion_score:.4f}")
    print(f"\nModel Behavior: {'Focused' if deletion_score > 0.2 else 'Distributed'}")
    print(f"Feature Importance: {'High' if deletion_score > 0.3 else 'Moderate' if deletion_score > 0.1 else 'Low'}")
    print("=" * 80)

    # Save attribution map
    visualize_attribution(attribution_map, 'attribution_map.png')


if __name__ == "__main__":
    main()

"""
Master Pipeline: Complete workflow for Dementia Detection Model
1. Extract data from dataset
2. Train model
3. Compute Integrated Gradients
4. Analyze deletion metrics
5. Generate visualizations with statistics
6. Test on single images
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import sys

# Import custom modules
from data_extraction import extract_slices
from dataset import MRIDataset
from model import DementiaCNN
from train import train_model
from integrated_gradients import IntegratedGradients, load_image, load_model

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def phase_1_extract_data(data_root='../dataset', output_dir='../oasis_slices', disc_max=None):
    """Phase 1: Extract MRI slices from dataset"""
    print("\n" + "=" * 80)
    print("PHASE 1: EXTRACTING MRI SLICES")
    print("=" * 80)
    print()

    slice_count = extract_slices(data_root, output_dir, cdr_threshold=0, disc_max=disc_max)

    total_slices = slice_count['CN'] + slice_count['AD']
    if total_slices == 0:
        print("\nERROR: No data extracted!")
        return None

    return slice_count


def phase_2_train_model(output_dir='../oasis_slices', batch_size=32, epochs=50):
    """Phase 2: Train the model"""
    print("\n" + "=" * 80)
    print("PHASE 2: TRAINING MODEL")
    print("=" * 80)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load dataset
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MRIDataset(output_dir, transform=transform, augment=True)
    print(f"Total images: {len(dataset)}\n")

    if len(dataset) == 0:
        print("ERROR: No images loaded!")
        return None, None

    # Check class distribution
    labels = np.array(dataset.labels)
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for u, c in zip(unique, counts):
        label_name = 'CN' if u == 0 else 'AD'
        percentage = (c / len(dataset)) * 100
        print(f"  {label_name}: {c} samples ({percentage:.1f}%)")

    # Split data with fixed random seed for reproducibility
    from torch.utils.data import DataLoader, random_split

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Use a generator with fixed seed to ensure reproducible splits
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTrain set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples\n")

    # Initialize and train model
    print("Initializing model...")
    model = DementiaCNN().to(device)
    total_params, trainable_params = model.get_model_size()
    print(f"Total parameters: {total_params:,}\n")

    # Get class weights
    class_weights = dataset.get_class_weights()

    # Train
    print("Training...")
    metrics = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        device=device,
        use_class_weights=True,
        class_weights=class_weights
    )

    # Test
    print("\nEvaluating on test set...")
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

    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}\n")

    return model, device


def phase_3_integrated_gradients(model, device, dataset_path='../oasis_slices', num_samples=5, steps=30):
    """Phase 3: Compute Integrated Gradients and deletion metrics"""
    print("\n" + "=" * 80)
    print("PHASE 3: INTEGRATED GRADIENTS & DELETION METRICS")
    print("=" * 80)
    print()

    model.eval()
    ig = IntegratedGradients(model, device=device)

    results = {
        'CN': {'deletion_scores': [], 'insertion_scores': [], 'confidences': [], 'paths': []},
        'AD': {'deletion_scores': [], 'insertion_scores': [], 'confidences': [], 'paths': []},
    }

    # Analyze CN samples
    print("Analyzing CN samples...")
    cn_dir = Path(dataset_path) / 'CN'
    cn_files = sorted(list(cn_dir.glob('*.png')))[:num_samples]

    for image_path in cn_files:
        input_image = load_image(str(image_path), device=device)

        with torch.no_grad():
            logits = model(input_image)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_class].item()

        attribution_map = ig.compute_attribution_map(input_image, pred_class, steps=steps)
        threshold = torch.quantile(attribution_map, 0.80)
        importance_mask = (attribution_map > threshold).float()

        deletion_score, _, _ = ig.deletion_metric(input_image, pred_class, importance_mask)
        insertion_score, _, _ = ig.insertion_metric(input_image, pred_class, importance_mask)

        results['CN']['deletion_scores'].append(deletion_score)
        results['CN']['insertion_scores'].append(insertion_score)
        results['CN']['confidences'].append(confidence)
        results['CN']['paths'].append(str(image_path))

    # Analyze AD samples
    print("Analyzing AD samples...")
    ad_dir = Path(dataset_path) / 'AD'
    ad_files = sorted(list(ad_dir.glob('*.png')))[:num_samples]

    for image_path in ad_files:
        input_image = load_image(str(image_path), device=device)

        with torch.no_grad():
            logits = model(input_image)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_class].item()

        attribution_map = ig.compute_attribution_map(input_image, pred_class, steps=steps)
        threshold = torch.quantile(attribution_map, 0.80)
        importance_mask = (attribution_map > threshold).float()

        deletion_score, _, _ = ig.deletion_metric(input_image, pred_class, importance_mask)
        insertion_score, _, _ = ig.insertion_metric(input_image, pred_class, importance_mask)

        results['AD']['deletion_scores'].append(deletion_score)
        results['AD']['insertion_scores'].append(insertion_score)
        results['AD']['confidences'].append(confidence)
        results['AD']['paths'].append(str(image_path))

    # Print summary
    print("\nFeature Importance Summary:")
    print("-" * 80)
    for class_name in ['CN', 'AD']:
        deletion_scores = np.array(results[class_name]['deletion_scores'])
        print(f"\n{class_name} Samples ({len(deletion_scores)} images):")
        print(f"  Avg Deletion Score: {deletion_scores.mean():+.4f} ± {deletion_scores.std():.4f}")
        print(f"  Positive Deletions: {(deletion_scores > 0).sum()}/{len(deletion_scores)} ({100*(deletion_scores > 0).sum()/len(deletion_scores):.1f}%)")

    return results


def phase_4_test_single_image(model, device, image_path, dataset_path='../oasis_slices'):
    """Phase 4: Test on single image and generate visualization"""
    print("\n" + "=" * 80)
    print("PHASE 4: SINGLE IMAGE ANALYSIS & VISUALIZATION")
    print("=" * 80)
    print()

    if not Path(image_path).exists():
        print(f"ERROR: Image not found at {image_path}")
        return None

    print(f"Testing image: {Path(image_path).name}\n")

    # Load and predict
    model.eval()
    input_image = load_image(str(image_path), device=device)

    with torch.no_grad():
        logits = model(input_image)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_class].item()
        cn_prob = probs[0, 0].item()
        ad_prob = probs[0, 1].item()

    class_names = ['CN (Cognitively Normal)', 'AD (Alzheimer\'s Disease)']
    print(f"Prediction: {class_names[pred_class]}")
    print(f"Confidence: {confidence:.1%}")
    print(f"  CN Probability: {cn_prob:.1%}")
    print(f"  AD Probability: {ad_prob:.1%}\n")

    # Compute Integrated Gradients
    print("Computing Integrated Gradients...")
    ig = IntegratedGradients(model, device=device)
    attribution_map = ig.compute_attribution_map(input_image, pred_class, steps=50)

    # Compute deletion metric
    threshold = torch.quantile(attribution_map, 0.75)
    importance_mask = (attribution_map > threshold).float()

    deletion_score, baseline_conf, deletion_conf = ig.deletion_metric(
        input_image, pred_class, importance_mask
    )
    insertion_score, black_conf, masked_conf = ig.insertion_metric(
        input_image, pred_class, importance_mask
    )

    print(f"Deletion Score: {deletion_score:+.4f}")
    print(f"Insertion Score: {insertion_score:+.4f}\n")

    # Create visualization
    print("Generating visualization...")
    visualization_path = create_single_image_visualization(
        image_path, input_image, attribution_map, pred_class, confidence,
        cn_prob, ad_prob, deletion_score, insertion_score
    )

    return {
        'image_path': image_path,
        'prediction': pred_class,
        'confidence': confidence,
        'cn_prob': cn_prob,
        'ad_prob': ad_prob,
        'attribution_map': attribution_map,
        'deletion_score': deletion_score,
        'insertion_score': insertion_score,
        'visualization_path': visualization_path
    }


def create_single_image_visualization(image_path, input_image, attribution_map, pred_class,
                                      confidence, cn_prob, ad_prob, deletion_score, insertion_score):
    """Create comprehensive visualization for single image"""
    import matplotlib.pyplot as plt

    # Load original image
    original_image = Image.open(image_path).convert('L')
    original_image = original_image.resize((224, 224), Image.BILINEAR)
    original_array = np.array(original_image, dtype=np.float32) / 255.0

    attribution_np = attribution_map.cpu().numpy()

    # Create figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f'Dementia Detection Analysis: {Path(image_path).name}',
                 fontsize=14, fontweight='bold')

    class_names = ['CN', 'AD']

    # 1. Original Image
    ax = axes[0, 0]
    ax.imshow(original_array, cmap='gray')
    ax.set_title('Original MRI Slice', fontweight='bold')
    ax.axis('off')

    # 2. Attribution Map
    ax = axes[0, 1]
    im = ax.imshow(attribution_np, cmap='hot')
    ax.set_title('Attribution Map', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 3. Overlay
    ax = axes[0, 2]
    attr_normalized = (attribution_np - attribution_np.min()) / (attribution_np.max() - attribution_np.min() + 1e-8)
    overlay = np.zeros((*original_array.shape, 3))
    overlay[:, :, 0] = attr_normalized * 0.7
    overlay[:, :, 1] = original_array * 0.5
    overlay[:, :, 2] = original_array * 0.5
    ax.imshow(overlay)
    ax.set_title('Overlay (Red=Important)', fontweight='bold')
    ax.axis('off')

    # 4. Most Important Parts Only (NEW)
    ax = axes[0, 3]
    threshold = torch.quantile(attribution_map, 0.75)
    mask = (attribution_map > threshold).float().cpu().numpy()
    important_only = original_array.copy()
    important_only[mask == 0] = 1.0  # Set non-important regions to white
    ax.imshow(important_only, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Top 25% Important Parts', fontweight='bold')
    ax.axis('off')

    # 5. Top 25% Important Regions (Binary mask)
    ax = axes[1, 0]
    mask_colored = np.zeros((*mask.shape, 3))
    mask_colored[:, :, 0] = original_array
    mask_colored[:, :, 1] = original_array * (1 - mask)
    mask_colored[:, :, 2] = original_array
    ax.imshow(mask_colored)
    ax.set_title('Top 25% Regions Highlighted', fontweight='bold')
    ax.axis('off')

    # 6. Statistics Text
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
PREDICTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Class: {class_names[pred_class]}
Confidence: {confidence:.1%}
CN Probability: {cn_prob:.1%}
AD Probability: {ad_prob:.1%}

FEATURE IMPORTANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deletion Score: {deletion_score:+.4f}
Insertion Score: {insertion_score:+.4f}

INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Positive Deletion: Model relies on
                   these regions (GOOD)
Negative Deletion: Model works better
                   without these (BAD)
"""
    ax.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 7. Feature Assessment
    ax = axes[1, 2]
    ax.axis('off')

    if deletion_score > 0.3:
        focus = "FOCUSED"
        color = "green"
        msg = "Model relies on specific\nregions for prediction"
    elif deletion_score > 0.1:
        focus = "MODERATE"
        color = "orange"
        msg = "Model uses some specific\nfeatures"
    else:
        focus = "DISTRIBUTED"
        color = "red"
        msg = "Model uses scattered\nfeatures (less interpretable)"

    assessment_text = f"""
FEATURE ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━
Focus Type: {focus}

{msg}

Confidence Level: {"HIGH" if abs(deletion_score) > 0.5 else "MEDIUM" if abs(deletion_score) > 0.2 else "LOW"}

Generalization: {"GOOD" if deletion_score > 0.3 else "FAIR" if deletion_score > 0.1 else "POOR"}
"""
    ax.text(0.5, 0.5, assessment_text, fontsize=9, verticalalignment='center',
            horizontalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    # 8. Empty space (for layout)
    ax = axes[1, 3]
    ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    os.makedirs('results', exist_ok=True)
    output_path = f"results/{Path(image_path).stem}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_path}")
    plt.close()

    return output_path


def main():
    import sys

    print("\n" + "=" * 80)
    print("DEMENTIA DETECTION MODEL - COMPLETE PIPELINE")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Configuration
    data_root = '../dataset'
    output_dir = '../oasis_slices'
    batch_size = 32
    epochs = 50
    disc_max = None  # Set to 11 for disc1-disc11 only

    # Check for --dev flag
    dev_mode = '--dev' in sys.argv
    if dev_mode:
        print("Running in DEV MODE (using existing best_model.pth)\n")

    # Phase 1: Extract data (skip in dev mode if data already extracted)
    if not dev_mode:
        slice_count = phase_1_extract_data(data_root, output_dir, disc_max=disc_max)
        if slice_count is None:
            return
    else:
        print("\n" + "=" * 80)
        print("PHASE 1: EXTRACT DATA (SKIPPED - DEV MODE)")
        print("=" * 80)
        print("Using existing extracted data from ../oasis_slices/\n")

    # Phase 2: Train model (skip in dev mode, load existing model)
    if not dev_mode:
        model, device = phase_2_train_model(output_dir, batch_size, epochs)
        if model is None:
            return
    else:
        print("\n" + "=" * 80)
        print("PHASE 2: TRAINING MODEL (SKIPPED - DEV MODE)")
        print("=" * 80)
        print("Loading existing model from best_model.pth...\n")

        # Load existing model
        model = DementiaCNN().to(device)
        model_path = 'best_model.pth'

        if not Path(model_path).exists():
            print(f"ERROR: Model file not found at {model_path}")
            return

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from: {model_path}")
        total_params, _ = model.get_model_size()
        print(f"Total parameters: {total_params:,}\n")

    # Phase 3: Integrated Gradients
    ig_results = phase_3_integrated_gradients(model, device, output_dir, num_samples=5, steps=30)

    # Phase 4: Test on single images
    print("\n" + "=" * 80)
    print("TESTING ON SINGLE IMAGES FROM test_images/")
    print("=" * 80)
    print()

    test_dir = Path('../test_images')
    test_images = sorted(list(test_dir.glob('*.png')))

    if test_images:
        for test_image in test_images:
            result = phase_4_test_single_image(model, device, str(test_image), output_dir)
            if result:
                print()
    else:
        print(f"No test images found in {test_dir}")

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nGenerated outputs:")
    if not dev_mode:
        print("  - Trained model: best_model.pth")
    else:
        print("  - Model used: best_model.pth")
    print("  - Visualizations: results/")
    print("  - Attribution maps: attributions/")
    print()


if __name__ == "__main__":
    main()

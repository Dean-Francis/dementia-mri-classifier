import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from PIL import Image
import seaborn as sns
import nibabel as nib
from pathlib import Path
import pandas as pd

# ============================================================================
# STEP 1: EXTRACT 2D SLICES FROM OASIS DATA
# ============================================================================

def extract_slices_from_oasis(data_root, output_dir, slices_per_subject=5, 
                               handle_mci='skip'):
    """
    Load OASIS data, extract LIMITED 2D slices from middle region, 
    and organize into CN/AD folders with metadata tracking
    
    Parameters:
    -----------
    data_root : str
        Path to OASIS disc1 folder
    output_dir : str
        Where to save extracted slices
    slices_per_subject : int
        Number of slices to extract per subject (default: 5)
    handle_mci : str
        How to handle CDR=0.5 cases:
        - 'skip': Exclude MCI subjects (default)
        - 'ad': Treat as AD
        - 'separate': Create separate MCI folder
    
    Returns:
    --------
    slice_count : dict
        Number of slices per class
    metadata : list
        List of dicts with subject_id, slice_idx, CDR, label info
    """
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'CN'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'AD'), exist_ok=True)
    if handle_mci == 'separate':
        os.makedirs(os.path.join(output_dir, 'MCI'), exist_ok=True)
    
    slice_count = {'CN': 0, 'AD': 0}
    if handle_mci == 'separate':
        slice_count['MCI'] = 0
    
    metadata = []
    subject_stats = {'total': 0, 'processed': 0, 'skipped_mci': 0, 
                     'failed': 0, 'skipped_no_cdr': 0}
    
    # Find all subject directories (OAS1_XXXX_MR1 folders)
    subject_dirs = sorted([d for d in Path(data_root).iterdir() 
                          if d.is_dir() and 'OAS1_' in d.name])
    
    print(f"Found {len(subject_dirs)} subject directories")
    print(f"Extracting {slices_per_subject} slices per subject\n")
    
    for subject_dir in subject_dirs:
        subject_stats['total'] += 1
        
        # Look for the txt file inside the subject directory
        txt_files = list(subject_dir.glob('*_MR1.txt'))
        
        if not txt_files:
            print(f"✗ {subject_dir.name}: No txt file found")
            subject_stats['failed'] += 1
            continue
        
        txt_file = txt_files[0]
        subject_id = subject_dir.name
        
        # Parse txt file to get CDR
        cdr = None
        
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    if 'CDR:' in line:
                        cdr_str = line.split(':')[1].strip()
                        if cdr_str and cdr_str != '':
                            cdr = float(cdr_str)
                            break
        except Exception as e:
            print(f"✗ {subject_id}: Error reading txt file - {e}")
            subject_stats['failed'] += 1
            continue
        
        if cdr is None:
            print(f"✗ {subject_id}: No CDR value found")
            subject_stats['skipped_no_cdr'] += 1
            continue
        
        # Determine label based on CDR
        if cdr == 0:
            label = 'CN'
        elif cdr == 0.5:
            if handle_mci == 'skip':
                print(f"⊘ {subject_id}: Skipping MCI (CDR=0.5)")
                subject_stats['skipped_mci'] += 1
                continue
            elif handle_mci == 'ad':
                label = 'AD'
            elif handle_mci == 'separate':
                label = 'MCI'
        else:  # cdr >= 1.0
            label = 'AD'
        
        # Find MPRAGE directory
        processed_dir = subject_dir / 'PROCESSED' / 'MPRAGE'
        
        if not processed_dir.exists():
            print(f"✗ {subject_id}: PROCESSED/MPRAGE directory not found")
            subject_stats['failed'] += 1
            continue
        
        # Look for SUBJ_* or T88_* subdirectories
        mprage_subdirs = list(processed_dir.glob('SUBJ_*')) + list(processed_dir.glob('T88_*'))
        
        if not mprage_subdirs:
            print(f"✗ {subject_id}: No SUBJ_* or T88_* directories found")
            subject_stats['failed'] += 1
            continue
        
        # Process the first valid subdirectory
        volume_loaded = False
        
        for mprage_subdir in mprage_subdirs:
            # Find img/hdr files
            img_files = list(mprage_subdir.glob('*.img'))
            hdr_files = list(mprage_subdir.glob('*.hdr'))
            
            if not img_files or not hdr_files:
                continue
            
            hdr_path = str(hdr_files[0])
            
            try:
                # Load img/hdr using nibabel
                img_data = nib.load(hdr_path)
                volume = img_data.get_fdata()
                
                # Remove singleton dimensions
                volume = np.squeeze(volume)
                
                # Ensure it's 3D
                if len(volume.shape) != 3:
                    print(f"✗ {subject_id}: Expected 3D, got shape {volume.shape}")
                    continue
                
                # Skip if volume is too small
                if volume.shape[0] < 50 or volume.shape[1] < 50 or volume.shape[2] < 20:
                    print(f"✗ {subject_id}: Volume too small {volume.shape}")
                    continue
                
                # Normalize to 0-255
                volume_min = volume.min()
                volume_max = volume.max()
                
                if volume_max - volume_min < 1e-8:
                    print(f"✗ {subject_id}: Volume has no contrast")
                    continue
                
                volume = (volume - volume_min) / (volume_max - volume_min)
                volume = (volume * 255).astype(np.uint8)
                
                # Select middle slices only (avoid edges with skull/air)
                total_slices = volume.shape[2]
                middle_start = total_slices // 3  # Start at 33% (more restrictive)
                middle_end = 2 * total_slices // 3  # End at 67% (more restrictive)
                
                # Select evenly spaced slices from middle region
                selected_indices = np.linspace(
                    middle_start, 
                    middle_end - 1,  # -1 to stay within bounds
                    slices_per_subject, 
                    dtype=int
                )
                
                slices_saved = 0
                
                for slice_idx in selected_indices:
                    slice_2d = volume[:, :, slice_idx]
                    
                    # Quality checks - MORE STRICT
                    mean_intensity = np.mean(slice_2d)
                    std_intensity = np.std(slice_2d)

                    # Skip if too dark, too bright, or no contrast
                    if mean_intensity < 30 or mean_intensity > 180 or std_intensity < 20:
                        continue
                    
                    # Save as PNG
                    filename = f"{subject_id}_slice{slice_idx:03d}.png"
                    filepath = os.path.join(output_dir, label, filename)
                    
                    img_pil = Image.fromarray(slice_2d, mode='L')
                    img_pil.save(filepath)
                    
                    # Save metadata
                    metadata.append({
                        'subject_id': subject_id,
                        'slice_idx': slice_idx,
                        'cdr': cdr,
                        'label': label,
                        'filename': filename,
                        'mean_intensity': mean_intensity,
                        'std_intensity': std_intensity
                    })
                    
                    slice_count[label] += 1
                    slices_saved += 1
                
                print(f"✓ {subject_id} ({label}): CDR={cdr}, "
                      f"volume={volume.shape}, saved {slices_saved} slices")
                
                subject_stats['processed'] += 1
                volume_loaded = True
                break  # Successfully processed, no need to check other subdirs
                
            except Exception as e:
                print(f"✗ {subject_id}: Error loading volume - {e}")
                continue
        
        if not volume_loaded:
            subject_stats['failed'] += 1
    
    # Save metadata to CSV
    df = pd.DataFrame(metadata)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Total subjects found: {subject_stats['total']}")
    print(f"Successfully processed: {subject_stats['processed']}")
    print(f"Skipped (MCI): {subject_stats['skipped_mci']}")
    print(f"Skipped (no CDR): {subject_stats['skipped_no_cdr']}")
    print(f"Failed: {subject_stats['failed']}")
    print("\nSlices extracted:")
    for label, count in slice_count.items():
        print(f"  {label}: {count} slices")
    print(f"\nMetadata saved to: {metadata_path}")
    print("="*60)


# ============================================================================
# STEP 2: DATASET SETUP
# ============================================================================

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load CN (label 0)
        cn_dir = os.path.join(root_dir, 'CN')
        if os.path.exists(cn_dir):
            for img_name in os.listdir(cn_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cn_dir, img_name))
                    self.labels.append(0)
        
        # Load AD (label 1)
        ad_dir = os.path.join(root_dir, 'AD')
        if os.path.exists(ad_dir):
            for img_name in os.listdir(ad_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(ad_dir, img_name))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label


# ============================================================================
# STEP 3: CNN MODEL
# ============================================================================

class DementiaCNN(nn.Module):
    def __init__(self):
        super(DementiaCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# STEP 4: TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=20, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=3)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses


# ============================================================================
# STEP 5: INTEGRATED GRADIENTS
# ============================================================================

def compute_integrated_gradients(model, images, labels, device='cpu', steps=200):
    """Compute IG attributions for a batch of images"""
    print(f"Computing IG on device: {device}")  # Add this line
    print(f"Number of images: {images.shape[0]}, Steps: {steps}")  # Add this too
    ig = IntegratedGradients(model)
    attributions_list = []
    
    for i in range(images.shape[0]):
        img = images[i:i+1].to(device)
        label = labels[i].item()
        
        attr = ig.attribute(img, target=label, n_steps=steps)
        attributions_list.append(attr.squeeze().detach().cpu().numpy())
    
    return np.array(attributions_list)


# ============================================================================
# STEP 6: DELETION METRIC
# ============================================================================

def deletion_metric(model, images, labels, attributions, device='cpu', 
                    steps=20, percentile=90):
    """
    Deletion metric: Remove top-k important pixels and measure performance drop
    """
    model.eval()
    batch_size = images.shape[0]
    
    # Baseline accuracy
    with torch.no_grad():
        outputs = model(images.to(device))
        baseline_preds = torch.argmax(outputs, dim=1).cpu().numpy()
    baseline_acc = accuracy_score(labels.numpy(), baseline_preds)
    
    deletion_scores = []
    
    for step in range(1, steps + 1):
        perturbed_images = images.clone()
        
        for i in range(batch_size):
            attr = attributions[i]
            
            # Calculate how many pixels to remove at this step
            fraction_to_remove = (step / steps) * (percentile / 100.0)
            threshold = np.percentile(np.abs(attr), 100 - (fraction_to_remove * 100))
            
            # Create mask: 0 where we delete, 1 where we keep
            mask = (np.abs(attr) < threshold).astype(np.float32)
            
            # Apply mask to the image (keeping normalized values)
            perturbed_images[i, 0] *= torch.tensor(mask, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(perturbed_images.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        acc = accuracy_score(labels.numpy(), preds)
        deletion_scores.append(baseline_acc - acc)
    
    return deletion_scores, baseline_acc


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    epochs = 20
    data_root = r'C:\Users\user\Desktop\Convolutional_Neural_Netowork\Dr. Firuz Kamalov\images\disc1'  
    output_dir = 'oasis_slices'
    
    print(f"Using device: {device}")
    
    # Step 1: Extract slices from OASIS
    print("\n=== Extracting 2D Slices from OASIS ===")
    extract_slices_from_oasis(data_root, output_dir, slices_per_subject=20, handle_mci='ad')
    
    # Step 2: Load dataset
    print("\n=== Loading Dataset ===")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = MRIDataset(output_dir, transform=transform)
    print(f"Total images: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: No images found! Check data path and folder structure.")
        return
    
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train set: {len(train_dataset)}, Test set: {len(test_dataset)}")
    
    # Step 3: Initialize and train model
    print("\n=== Training Model ===")
    model = DementiaCNN().to(device)
    train_losses, val_losses = train_model(model, train_loader, test_loader, 
                                           epochs=epochs, device=device)
    
    # Step 4: Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(all_labels, all_preds, zero_division=0):.4f}")
    
    # Step 5: Compute Integrated Gradients
    print("\n=== Computing Integrated Gradients ===")
    test_images, test_labels = next(iter(test_loader))

    test_images_subset = test_images[:4]
    test_labels_subset = test_labels[:4]
    attributions = compute_integrated_gradients(model, test_images, 
                                               test_labels, device=device, steps=200)
    print(f"Computed attributions for {len(attributions)} images")
    
    # Step 6: Deletion Metric
    print("\n=== Evaluating with Deletion Metric ===")
    deletion_scores, baseline = deletion_metric(model, test_images, test_labels,
                                               attributions, device=device, steps=20)
    
    print(f"Baseline Accuracy: {baseline:.4f}")
    print(f"Max Deletion Score: {max(deletion_scores):.4f}")
    print(f"Mean Deletion Score: {np.mean(deletion_scores):.4f}")
    
    # Step 7: Visualizations
    print("\n=== Generating Visualizations ===")
    
    # Create a larger figure for multiple examples
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Loss curves (top left)
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Plot 2: Deletion curve (top right)
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(deletion_scores)
    ax2.set_title('Deletion Metric (IG)')
    ax2.set_xlabel('Deletion Steps')
    ax2.set_ylabel('Accuracy Drop')
    ax2.grid(True, alpha=0.3)
    
    # Plots 3-10: Show 4 MRI samples with their attributions (2 rows of 4)
    num_samples = min(4, len(test_images))
    
    for i in range(num_samples):
        # Original MRI
        ax_img = plt.subplot(3, 4, 5 + i)
        sample_img = test_images[i].squeeze().cpu().numpy()
        ax_img.imshow(sample_img, cmap='gray')
        ax_img.set_title(f'MRI {i+1} (Label: {test_labels[i].item()})')
        ax_img.axis('off')
        
        # IG Attribution
        ax_attr = plt.subplot(3, 4, 9 + i)
        sample_attr = attributions[i]
        im = ax_attr.imshow(np.abs(sample_attr), cmap='hot')
        ax_attr.set_title(f'IG Attribution {i+1}')
        ax_attr.axis('off')
        plt.colorbar(im, ax=ax_attr, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('dementia_pipeline_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to dementia_pipeline_results.png")
    plt.show()


if __name__ == "__main__":
    main()
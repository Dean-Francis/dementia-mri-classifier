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

# ============================================================================
# STEP 1: EXTRACT 2D SLICES FROM OASIS DATA
# ============================================================================

def extract_slices_from_oasis(data_root, output_dir, cdr_threshold=0.5):
    """
    Load OASIS data, extract 2D slices, and organize into CN/AD folders
    
    CDR = 0 -> CN (Control)
    CDR > 0 -> AD (Alzheimer's)
    """
    os.makedirs(os.path.join(output_dir, 'CN'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'AD'), exist_ok=True)
    
    slice_count = {'CN': 0, 'AD': 0}
    
    # Find all subject txt files
    subject_files = sorted(Path(data_root).glob('*_MR1.txt'))
    
    print(f"Found {len(subject_files)} subjects")
    
    for txt_file in subject_files:
        # Parse txt file to get CDR and subject ID
        subject_id = None
        cdr = None
        
        with open(txt_file, 'r') as f:
            for line in f:
                if 'SESSION ID:' in line:
                    subject_id = line.split(':')[1].strip()
                elif 'CDR:' in line:
                    cdr = float(line.split(':')[1].strip())
        
        if subject_id is None or cdr is None:
            continue
        
        # Determine label
        label = 'AD' if cdr > 0 else 'CN'
        
        # Find MPRAGE img/hdr files
        subject_dir = txt_file.parent / subject_id / 'PROCESSED' / 'MPRAGE'
        
        # Look for SUBJ_111 or similar subdirectories
        mprage_dirs = list(subject_dir.glob('SUBJ_*')) + list(subject_dir.glob('T88_*'))
        
        for mprage_subdir in mprage_dirs:
            # Find img file
            img_files = list(mprage_subdir.glob('*.img'))
            hdr_files = list(mprage_subdir.glob('*.hdr'))
            
            if not img_files or not hdr_files:
                continue
            
            img_path = str(img_files[0])
            
            try:
                # Load img/hdr as NIfTI using nibabel
                img_data = nib.load(img_path)
                volume = img_data.get_fdata()
                
                # Normalize to 0-255
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
                volume = (volume * 255).astype(np.uint8)
                
                # Extract all 2D slices
                for slice_idx in range(volume.shape[2]):
                    slice_2d = volume[:, :, slice_idx]
                    
                    # Skip mostly black slices
                    if np.mean(slice_2d) < 10:
                        continue
                    
                    # Save as PNG
                    filename = f"{subject_id}_{mprage_subdir.name}_slice{slice_idx:03d}.png"
                    filepath = os.path.join(output_dir, label, filename)
                    
                    img_pil = Image.fromarray(slice_2d, mode='L')
                    img_pil.save(filepath)
                    slice_count[label] += 1
                
                print(f"✓ {subject_id} ({label}): CDR={cdr}")
                
            except Exception as e:
                print(f"✗ Error loading {subject_id}: {e}")
                continue
    
    print(f"\nExtraction complete!")
    print(f"CN slices: {slice_count['CN']}")
    print(f"AD slices: {slice_count['AD']}")
    
    return slice_count


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

def compute_integrated_gradients(model, images, labels, device='cpu', steps=50):
    """Compute IG attributions for a batch of images"""
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
            threshold = np.percentile(np.abs(attr), 100 - (step / steps) * percentile)
            mask = np.abs(attr) >= threshold
            perturbed_images[i] *= torch.tensor(mask, dtype=torch.float32)
        
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
    data_root = '/Users/deanfrancistolero/Desktop/Firuz Kamalov/data/disc1'  # Change this
    output_dir = 'oasis_slices'
    
    print(f"Using device: {device}")
    
    # Step 1: Extract slices from OASIS
    print("\n=== Extracting 2D Slices from OASIS ===")
    extract_slices_from_oasis(data_root, output_dir)
    
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
    attributions = compute_integrated_gradients(model, test_images, 
                                               test_labels, device=device)
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
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train')
    axes[0, 0].plot(val_losses, label='Val')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # Deletion curve
    axes[0, 1].plot(deletion_scores)
    axes[0, 1].set_title('Deletion Metric (IG)')
    axes[0, 1].set_xlabel('Deletion Steps')
    axes[0, 1].set_ylabel('Accuracy Drop')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample MRI with IG
    sample_img = test_images[0].squeeze().numpy()
    sample_attr = attributions[0]
    
    axes[1, 0].imshow(sample_img, cmap='gray')
    axes[1, 0].set_title('Original MRI')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.abs(sample_attr), cmap='hot')
    axes[1, 1].set_title('Integrated Gradients Attribution')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dementia_pipeline_results.png', dpi=150)
    print("Saved visualization to dementia_pipeline_results.png")
    plt.show()


if __name__ == "__main__":
    main()
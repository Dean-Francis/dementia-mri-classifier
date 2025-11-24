# Dementia MRI Classifier

A deep learning system for detecting Alzheimer's Disease (AD) from brain MRI scans using Convolutional Neural Networks with integrated explainability through Integrated Gradients.

## Overview

This project implements a complete pipeline for:
1. **Data extraction** from OASIS MRI dataset
2. **CNN-based binary classification** (Cognitively Normal vs Alzheimer's Disease)
3. **Model interpretability** using Integrated Gradients
4. **Feature importance analysis** with deletion and insertion metrics

## Project Structure

```
dementia-mri-classifier/
├── src/
│   ├── config.py              # Configuration management
│   ├── data_extraction.py     # Extract MRI slices from OASIS dataset
│   ├── dataset.py             # PyTorch Dataset class with augmentation
│   ├── model.py               # CNN architecture (DementiaCNN)
│   ├── train.py               # Training loop with early stopping
│   ├── integrated_gradients.py # Model interpretability & metrics
│   ├── master_pipeline.py     # Complete end-to-end pipeline
│   └── evaluate_disc12.py     # Evaluation scripts
├── dataset/                   # OASIS dataset (disc1, disc2, etc.)
├── oasis_slices/             # Extracted 2D slices
│   ├── CN/                   # Cognitively Normal slices
│   └── AD/                   # Alzheimer's Disease slices
├── test_images/              # Sample images for testing
├── results/                  # Visualization outputs
├── requirements.txt          # Python dependencies
└── best_model.pth           # Trained model weights
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- PyTorch >= 2.9.0
- torchvision >= 0.15.0
- nibabel >= 5.1.0 (for NIfTI file reading)
- captum >= 0.6.0 (for Integrated Gradients)
- scikit-learn, pandas, matplotlib, seaborn

## Dataset

The project uses the **OASIS (Open Access Series of Imaging Studies)** dataset:
- **Format**: NIfTI (.img/.hdr files)
- **Structure**: Multiple discs (disc1, disc2, ...)
- **Labels**: Based on Clinical Dementia Rating (CDR)
  - CDR = 0.0 → CN (Cognitively Normal)
  - CDR > 0.0 → AD (Alzheimer's Disease)

### Dataset Structure
```
dataset/
├── disc1/
│   ├── OAS1_0001_MR1/
│   │   ├── OAS1_0001_MR1.txt          # Patient metadata (CDR score)
│   │   └── PROCESSED/MPRAGE/SUBJ_111/
│   │       └── *.img                   # 3D MRI volume
│   ├── OAS1_0002_MR1/
│   └── ...
├── disc2/
└── ...
```

## How It Works

### 1. Data Extraction ([data_extraction.py](src/data_extraction.py))

Converts 3D MRI volumes into 2D slices for training:

```python
extract_slices(data_root='dataset', output_dir='oasis_slices')
```

**Process:**
1. Reads NIfTI format 3D brain volumes using `nibabel`
2. Extracts Clinical Dementia Rating (CDR) from patient metadata files
3. Slices each 3D volume into 2D axial slices
4. **Quality filtering**: Removes low-quality slices
   - Dark slices (mean intensity < 30)
   - Overexposed slices (mean intensity > 180)
   - Low contrast slices (std < 20)
5. Normalizes pixel values to [0, 255]
6. Saves as PNG images in CN/ or AD/ folders

**Caching**: Already-processed subjects are skipped to speed up re-runs.

### 2. Dataset Loader ([dataset.py](src/dataset.py))

`MRIDataset` class handles data loading and augmentation:

**Features:**
- Loads grayscale PNG images from CN/ and AD/ folders
- Resizes images to 224×224 pixels
- Normalizes to [-1, 1] range (mean=0.5, std=0.5)
- **Data augmentation** (training only):
  - Random rotation (±10 degrees)
  - Random translation (10%)
  - Random scaling (0.9-1.1×)
  - Random horizontal flip
- Computes class weights for handling imbalanced datasets

### 3. CNN Architecture ([model.py](src/model.py))

**DementiaCNN** - Custom CNN for binary classification:

```
Input: [1, 224, 224] (grayscale MRI slice)
    ↓
Conv Block 1: 1 → 16 channels + BatchNorm + ReLU + MaxPool (112×112)
    ↓
Conv Block 2: 16 → 32 channels + BatchNorm + ReLU + MaxPool (56×56)
    ↓
Conv Block 3: 32 → 64 channels + BatchNorm + ReLU + MaxPool (28×28)
    ↓
Conv Block 4: 64 → 128 channels + BatchNorm + ReLU + AdaptiveAvgPool (1×1)
    ↓
Flatten: [128]
    ↓
FC1: 128 → 64 + ReLU + Dropout(0.5)
    ↓
FC2: 64 → 32 + ReLU + Dropout(0.5)
    ↓
FC3: 32 → 2 (logits for CN and AD)
    ↓
Output: [2] (class scores)
```

**Key Features:**
- Lightweight architecture (~100K parameters)
- Batch normalization for stable training
- Dropout (50%) to prevent overfitting
- He/Kaiming initialization for ReLU networks

### 4. Training ([train.py](src/train.py))

**Training Configuration:**
- **Loss**: CrossEntropyLoss with class weights (handles imbalance)
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Batch size**: 32
- **Epochs**: Up to 50 (with early stopping)
- **Data split**: 70% train, 15% validation, 15% test

**Training Features:**
- **Learning rate scheduling**: ReduceLROnPlateau
  - Reduces LR by 50% if validation loss plateaus for 3 epochs
  - Minimum LR: 1e-6
- **Early stopping**: Stops if validation loss doesn't improve for 10 epochs
- **Gradient clipping**: Max norm = 1.0 (prevents exploding gradients)
- **Metrics tracking**: Loss, accuracy, precision, recall, F1-score

**Model Checkpointing:**
- Best model saved to `best_model.pth` based on validation loss

### 5. Integrated Gradients ([integrated_gradients.py](src/integrated_gradients.py))

**Explainable AI** for understanding model predictions.

#### What are Integrated Gradients?

A method to attribute the prediction of a neural network to its input features:

1. **Baseline**: Start from a black image (zero input)
2. **Path**: Create a straight-line interpolation from baseline to actual input
3. **Gradients**: Compute gradients at each step along the path
4. **Integration**: Sum gradients to get feature importance

**Mathematical Formula:**
```
IG(x) = (x - baseline) × ∫[0,1] ∂F(baseline + α(x - baseline))/∂x dα
```

#### Attribution Map
- Shows which pixels are most important for the prediction
- Brighter regions = higher importance
- Helps understand what the model "looks at"

#### Deletion Metric

**Purpose**: Measure how much the model relies on important features.

**Process:**
1. Compute attribution map using Integrated Gradients
2. Identify top 25% most important pixels
3. **Remove** those pixels (set to black)
4. Measure confidence drop

**Interpretation:**
- **High deletion score (>0.3)**: Model heavily relies on these features (GOOD)
- **Low deletion score (<0.1)**: Model uses distributed/scattered features (BAD)

**Formula:**
```
Deletion Score = Confidence(full image) - Confidence(masked image)
```

#### Insertion Metric

**Purpose**: Measure how informative important features are.

**Process:**
1. Start with black image
2. **Add only** the top 25% important pixels
3. Measure confidence gain

**Interpretation:**
- **High insertion score**: Important features alone are sufficient
- **Low insertion score**: Model needs broader context

### 6. Master Pipeline ([master_pipeline.py](src/master_pipeline.py))

**Complete end-to-end workflow:**

#### Phase 1: Data Extraction
```bash
python src/master_pipeline.py
```
- Extracts MRI slices from `dataset/` to `oasis_slices/`
- Filters discs if specified (e.g., `disc_max=11` uses only disc1-disc11)

#### Phase 2: Model Training
- Loads extracted slices
- Splits into train/val/test sets (70/15/15)
- Trains CNN with early stopping
- Saves best model to `best_model.pth`

#### Phase 3: Integrated Gradients Analysis
- Analyzes 5 sample images from each class (CN and AD)
- Computes attribution maps and deletion metrics
- Prints summary statistics

#### Phase 4: Single Image Visualization
- Tests on images in `test_images/` folder
- Generates comprehensive visualizations:
  - Original MRI slice
  - Attribution map (heatmap)
  - Overlay (important regions in red)
  - Top 25% important parts only
  - Statistics and interpretation
- Saves to `results/` folder

#### Dev Mode
Skip data extraction and training, use existing model:
```bash
python src/master_pipeline.py --dev
```

## Usage

### Quick Start Guide

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Prepare Dataset
Ensure your dataset is structured as:
```
dataset/
├── disc1/
│   ├── OAS1_0001_MR1/
│   │   ├── OAS1_0001_MR1.txt
│   │   └── PROCESSED/MPRAGE/SUBJ_111/*.img
│   └── ...
├── disc2/
└── ...
```

#### Step 3: Run Complete Pipeline
```bash
cd src
python master_pipeline.py
```

This executes all 4 phases automatically (extraction → training → analysis → visualization).

---

### Option 1: Complete Pipeline (Train from Scratch)

**Command:**
```bash
cd src
python master_pipeline.py
```

**What happens:**
1. **Phase 1**: Extracts MRI slices from `dataset/` to `oasis_slices/`
2. **Phase 2**: Trains CNN model for ~50 epochs (saves to `best_model.pth`)
3. **Phase 3**: Analyzes 5 CN and 5 AD samples with Integrated Gradients
4. **Phase 4**: Generates visualizations for images in `test_images/`

**Expected runtime**: 30-60 minutes (depends on dataset size and GPU availability)

**Outputs:**
- `best_model.pth` - Trained model weights
- `oasis_slices/CN/` - Cognitively normal slices
- `oasis_slices/AD/` - Alzheimer's disease slices
- `results/*_analysis.png` - Visualization outputs

---

### Option 2: Dev Mode (Use Existing Model)

**Command:**
```bash
cd src
python master_pipeline.py --dev
```

**What this does:**
- Skips Phase 1 (data extraction)
- Skips Phase 2 (training)
- Loads existing `best_model.pth`
- Runs Phase 3 & 4 (analysis and visualization)

**Use when:**
- You already have a trained model
- You want to test on new images quickly
- You're debugging visualization code

**Expected runtime**: 1-2 minutes

---

### Option 3: Run Individual Components

#### 3a. Extract Data Only

```bash
cd src
python -c "
from data_extraction import extract_slices
extract_slices('../dataset', '../oasis_slices', disc_max=11)
"
```

Or create a script `extract_only.py`:
```python
from data_extraction import extract_slices

# Extract from disc1 to disc11 only
slice_count = extract_slices(
    data_root='../dataset',
    output_dir='../oasis_slices',
    disc_max=11  # Optional: limit to specific discs
)

print(f"Extracted {slice_count['CN']} CN slices")
print(f"Extracted {slice_count['AD']} AD slices")
```

Run: `python extract_only.py`

---

#### 3b. Train Model Only

Create `train_only.py`:
```python
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from model import DementiaCNN
from dataset import MRIDataset
from train import train_model

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = MRIDataset('../oasis_slices', transform=transform, augment=True)
print(f"Total samples: {len(dataset)}")

# Split data
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = DementiaCNN().to(device)

# Train
class_weights = dataset.get_class_weights()
metrics = train_model(
    model, train_loader, val_loader,
    epochs=50, device=device,
    use_class_weights=True,
    class_weights=class_weights
)

print("Training complete! Model saved to best_model.pth")
```

Run: `python train_only.py`

---

#### 3c. Analyze Single Image

```bash
cd src
python integrated_gradients.py
```

This will:
- Load `best_model.pth`
- Analyze first image in `test_images/`
- Print deletion and insertion metrics
- Save attribution map to `attribution_map.png`

**Or create custom analysis script:**
```python
import torch
from integrated_gradients import IntegratedGradients, load_model, load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = load_model('best_model.pth', device)

# Load image
image_path = 'test_images/sample.png'
input_image = load_image(image_path, device)

# Get prediction
with torch.no_grad():
    logits = model(input_image)
    pred_class = torch.argmax(logits, dim=1).item()
    print(f"Prediction: {'CN' if pred_class == 0 else 'AD'}")

# Compute Integrated Gradients
ig = IntegratedGradients(model, device)
attribution = ig.compute_attribution_map(input_image, pred_class, steps=50)

# Compute deletion metric
threshold = torch.quantile(attribution, 0.75)
mask = (attribution > threshold).float()
deletion_score, base_conf, del_conf = ig.deletion_metric(input_image, pred_class, mask)

print(f"Deletion Score: {deletion_score:.4f}")
```

---

### Option 4: Test on Custom Images

1. **Place your test images** in `test_images/` folder (PNG format, grayscale)

2. **Run analysis:**
```bash
cd src
python master_pipeline.py --dev
```

3. **Check results** in `results/` folder

**Image requirements:**
- Format: PNG (grayscale preferred)
- Will be automatically resized to 224×224
- Should be axial MRI brain slices

## Configuration

Edit [src/config.py](src/config.py) to customize:

- **Data settings**: Image size, normalization, quality filters
- **Model architecture**: Channels, dropout rate, batch norm
- **Training hyperparameters**: Learning rate, batch size, epochs
- **Augmentation**: Rotation, translation, brightness

## Output Files

After running the pipeline:

```
├── best_model.pth              # Trained model weights
├── results/                    # Visualization outputs
│   └── *_analysis.png         # Single image analysis
└── oasis_slices/              # Extracted slices
    ├── CN/                    # ~X,XXX slices
    └── AD/                    # ~X,XXX slices
```

## Model Performance

Expected metrics (varies by dataset):
- **Test Accuracy**: 70-85%
- **Precision**: 0.70-0.85
- **Recall**: 0.70-0.85
- **F1 Score**: 0.70-0.85

Performance depends on:
- Number of discs used
- Class balance (CN vs AD ratio)
- Quality of MRI scans
- Augmentation settings

## Interpretability Results

### Deletion Score Analysis
- **CN samples**: Typically lower deletion scores (0.1-0.3)
  - Model looks at distributed features across brain
- **AD samples**: Often higher deletion scores (0.3-0.5)
  - Model focuses on specific atrophy regions (hippocampus, ventricles)

### What the Model Learns

The CNN learns to identify:
1. **Structural changes**: Enlarged ventricles, cortical thinning
2. **Hippocampal atrophy**: Key indicator of Alzheimer's
3. **Gray matter reduction**: In temporal and parietal lobes
4. **White matter changes**: Periventricular lesions

## Reproducibility

The project uses fixed random seeds for reproducibility:
- `np.random.seed(42)`
- `torch.manual_seed(42)`
- Deterministic CUDA operations
- Fixed data splits using `torch.Generator()`

Running the same pipeline multiple times produces identical results.

## Troubleshooting

### "No data extracted"
- Check dataset path points to folder containing disc folders
- Verify .txt files contain CDR scores
- Ensure .img files exist in PROCESSED/MPRAGE/SUBJ_111/

### "Model not found"
- Run full pipeline first (without `--dev` flag)
- Or download pretrained weights to `best_model.pth`

### Low accuracy
- Increase number of discs (use more data)
- Adjust augmentation parameters
- Try longer training (increase epochs)
- Check class balance

### CUDA out of memory
- Reduce batch size in config
- Use CPU instead: `device='cpu'`

## Scientific Background

### Clinical Dementia Rating (CDR)
- 0.0 = No dementia (Cognitively Normal)
- 0.5 = Very mild dementia
- 1.0 = Mild dementia
- 2.0 = Moderate dementia
- 3.0 = Severe dementia

## License

This is an academic project for educational purposes.

## Acknowledgments

- **OASIS** (Open Access Series of Imaging Studies) for providing the dataset
- **PyTorch** and **Captum** teams for deep learning frameworks
- Research on Alzheimer's Disease neuroimaging

---

**Note**: This tool is for research purposes only and should not be used for clinical diagnosis. Always consult medical professionals for health-related decisions.

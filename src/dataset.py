import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        """
        MRI Dataset for binary classification (CN vs AD)

        Args:
            root_dir: Path to folder containing 'CN' and 'AD' subdirectories
            transform: Optional transforms to apply to images
            augment: If True, apply additional augmentation for training
        """
        self.root_dir = root_dir
        self.augment = augment
        self.images = []  # Image paths
        self.labels = []  # Labels (0 = CN, 1 = AD)

        # Load CN Images (label 0)
        cn_dir = os.path.join(root_dir, 'CN')
        if os.path.exists(cn_dir):
            for img_name in os.listdir(cn_dir):
                if img_name.endswith('.png'):
                    self.images.append(os.path.join(cn_dir, img_name))
                    self.labels.append(0)

        # Load AD Images (label 1)
        ad_dir = os.path.join(root_dir, 'AD')
        if os.path.exists(ad_dir):
            for img_name in os.listdir(ad_dir):
                if img_name.endswith('.png'):
                    self.images.append(os.path.join(ad_dir, img_name))
                    self.labels.append(1)

        # Set transform: use provided or default to basic normalization
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        self.transform = transform

        # Additional augmentation for training robustness
        if self.augment:
            self.augmentation = transforms.Compose([
                transforms.RandomAffine(
                    degrees=10,           # Random rotation ±10 degrees
                    translate=(0.1, 0.1), # Random translation
                    scale=(0.9, 1.1)      # Random scaling
                ),
                transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            ])
        else:
            self.augmentation = None

    def __len__(self):
        """Return total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """
        Load and return image-label pair

        Returns:
            (image, label): Tensor image and integer label
        """
        img_path = self.images[index]
        label = self.labels[index]

        try:
            # Load image as grayscale
            image = Image.open(img_path).convert('L')

            # Apply augmentation if enabled
            if self.augmentation is not None:
                image = self.augmentation(image)

            # Apply standard transforms (resize, normalize, etc.)
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced datasets

        Returns:
            Tensor: Weight for each class (lower count = higher weight)
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        return torch.tensor(weights, dtype=torch.float32)

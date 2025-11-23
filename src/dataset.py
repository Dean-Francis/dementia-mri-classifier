import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, subject_dirs: list[Path], transform=None):
        # subject_dirs: list of folder Paths, each containing PNG slices
        self.transform = transform
        self.images = []
        self.labels = []

        for subj_dir in subject_dirs:
            label = 0 if subj_dir.parent.name == "CN" else 1

            # Add all PNGs inside this subject’s folder
            for img_path in subj_dir.glob("*.png"):
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        
        return image, label

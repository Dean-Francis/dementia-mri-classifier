import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # root_dir: Path to 'oasis_slices' (the folder containing CN and AD slices)
        # transform: Optional transforms to apply to images

        self.root_dir = root_dir
        self.transform = transform
        self.images = [] # Will store image paths
        self.labels = [] # Will store labels (0 = CN, 1 = AD)

        # Load CN Images
        cn_dir = os.path.join(root_dir, 'CN')
        if os.path.exists(cn_dir):
            for img_name in os.listdir(cn_dir):
                if img_name.endswith('.png'):
                    self.images.append(os.path.join(cn_dir, img_name))
                    self.labels.append(0)
        
        # Load AD Images
        ad_dir = os.path.join(root_dir, 'AD')
        if os.path.exists(ad_dir):
            for img_name in os.listdir(ad_dir):
                if img_name.endswith('.png'):
                    self.images.append(os.path.join(ad_dir, img_name))
                    self.labels.append(1)
    # Returns the total number of paths
    def __len__(self):
        return len(self.images)
    
    # 
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        # Load the PNG image
        image = Image.open(img_path).convert('L') # Load it and ensure it's in grayscale

        if self.transform:
            image = self.transform(image)

        # returns the image and the label (0 = CN, 1 = AD)
        return image, label

import torch
import torch.nn as nn
import torch.nn.functional as F


class DementiaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------
        # Feature Extractor (CNN)
        # -------------------------
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),               # 112 × 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),               # 56 × 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),               # 28 × 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))     # → (256, 1, 1)
        )

        # -------------------------
        # Classifier (Fully Connected)
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # → (batch, 256)
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)                 # Binary output logit
        )

    def forward(self, x):
        x = self.features(x)   # → (batch, 256, 1, 1)
        x = self.classifier(x) # → (batch, 1)
        return x

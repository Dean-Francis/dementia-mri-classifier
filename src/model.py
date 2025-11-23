import torch
from torch import nn


class DementiaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv Block 1: 1 -> 16 channels (reduced from 32 to reduce params)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Light regularization: Drop 10% of feature maps
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112

            # Conv Block 2: 16 -> 32 channels (reduced from 64)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Light regularization: Drop 10% of feature maps
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56

            # Conv Block 3: 32 -> 64 channels (reduced from 128)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Light regularization: Drop 10% of feature maps
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28

            # Conv Block 4: 64 -> 128 channels (reduced from 256)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Light regularization: Drop 10% of feature maps
            nn.AdaptiveAvgPool2d((1, 1))  # 28x28 -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # Reduced from 256->128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),   # Reduced from 128->64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)  # Raw logits for CN and AD
        )

        # Initialize weights using He initialization (optimal for ReLU)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization for ReLU networks"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to: [batch, 256]
        x = self.classifier(x)  # Output: [batch, 2]
        return x  # logits

    def get_model_size(self):
        """Return total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
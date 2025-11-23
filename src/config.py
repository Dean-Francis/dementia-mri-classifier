"""
Configuration file for Dementia Detection project
Centralizes all hyperparameters and settings
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    data_root: str = r'dataset/disc1'
    output_dir: str = 'oasis_slices'
    cdr_threshold: float = 0.0  # 0=CN, >0=AD

    # Slice quality filtering
    min_intensity: float = 30  # Skip if mean < 30 (dark slices)
    max_intensity: float = 180  # Skip if mean > 180 (overexposed)
    min_std_intensity: float = 20  # Skip if std < 20 (no contrast)

    # Image normalization
    image_size: tuple = (224, 224)
    normalize_mean: float = 0.5
    normalize_std: float = 0.5

    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    num_classes: int = 2
    input_channels: int = 1

    # Architecture
    conv_channels: List[int] = None
    use_batch_norm: bool = True
    dropout_rate: float = 0.5

    # Initialization
    weight_init: str = 'kaiming'  # 'kaiming', 'normal', 'xavier'

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128, 256]


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Optimization
    optimizer: str = 'adam'  # 'adam', 'sgd'
    momentum: float = 0.9  # For SGD

    # Learning rate scheduling
    lr_scheduler: str = 'reduce_on_plateau'  # 'cosine', 'linear', 'reduce_on_plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Regularization
    use_class_weights: bool = True
    gradient_clip_norm: Optional[float] = 1.0

    # Validation
    val_frequency: int = 1  # Validate every N epochs

    # Checkpointing
    save_best_model: bool = True
    model_save_path: str = 'checkpoints/best_model.pth'

    # Device
    device: str = 'cuda'  # 'cuda' or 'cpu'

    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    use_augmentation: bool = True

    # Geometric transforms
    rotation_degrees: float = 10.0
    translation_ratio: float = 0.1
    scale_range: tuple = (0.9, 1.1)

    # Intensity transforms
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2

    # Noise
    gaussian_noise_std: float = 0.01


class Config:
    """Master configuration combining all sub-configs"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.augmentation = AugmentationConfig()

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'augmentation': self.augmentation.__dict__
        }

    def __repr__(self) -> str:
        """Pretty print configuration"""
        lines = []
        lines.append("=" * 60)
        lines.append("CONFIGURATION SUMMARY")
        lines.append("=" * 60)

        for section_name, section_config in [
            ("DATA", self.data),
            ("MODEL", self.model),
            ("TRAINING", self.training),
            ("AUGMENTATION", self.augmentation)
        ]:
            lines.append(f"\n{section_name}:")
            for key, value in section_config.__dict__.items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)


# Global config instance
CONFIG = Config()

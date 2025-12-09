# src/config.py

from dataclasses import dataclass
import os


@dataclass
class ASLConfig:
    # Image / model settings
    img_height: int = 224
    img_width: int = 224
    num_channels: int = 3

    # Number of classes. For ASL A–Y, skipping J and Z, that's 24.
    # (We still override this with len(class_names) when building models.)
    num_classes: int = 24

    # Root data directory (you will have data/A, data/B, ..., data/Y)
    data_dir: str = os.path.join("data")

    # Training hyperparameters
    batch_size: int = 8
    epochs: int = 15
    learning_rate: float = 1e-4

    # Output directories
    experiments_dir: str = os.path.join("experiments")
    model_dir: str = os.path.join("experiments", "models")
    logs_dir: str = os.path.join("experiments", "logs")


CONFIG = ASLConfig()

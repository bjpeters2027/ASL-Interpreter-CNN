from dataclasses import dataclass
import os


@dataclass
class ASLConfig:
    # Image / model settings
    img_height: int = 224
    img_width: int = 224
    num_channels: int = 3
    num_classes: int = 26  # A-Z

    # Data directories (relative to project root)
    train_dir: str = os.path.join("data", "train")
    val_dir: str = os.path.join("data", "val")

    # Training hyperparameters
    batch_size: int = 8
    epochs: int = 15
    learning_rate: float = 1e-4

    # Output directories
    experiments_dir: str = os.path.join("experiments")
    model_dir: str = os.path.join("experiments", "models")
    logs_dir: str = os.path.join("experiments", "logs")


CONFIG = ASLConfig()

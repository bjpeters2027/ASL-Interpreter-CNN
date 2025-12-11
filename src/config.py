# src/config.py

from dataclasses import dataclass
import os


@dataclass
class ASLConfig:
    img_height: int = 224
    img_width: int = 224
    num_channels: int = 3

    num_classes: int = 24

    data_dir: str = os.path.join("data")


    batch_size: int = 8
    epochs: int = 15
    learning_rate: float = 1e-4

    experiments_dir: str = os.path.join("experiments")
    model_dir: str = os.path.join("experiments", "models")
    logs_dir: str = os.path.join("experiments", "logs")


CONFIG = ASLConfig()

# scripts/train_transfer.py

import os

from src.dataset import get_datasets
from src.model_transfer import build_transfer_model
from src.train import compile_and_train
from src.config import CONFIG


RUN_NAME = "transfer_mobilenetv2"


def main():
    os.makedirs(CONFIG.experiments_dir, exist_ok=True)
    os.makedirs(CONFIG.model_dir, exist_ok=True)
    os.makedirs(CONFIG.logs_dir, exist_ok=True)

    print("[train_transfer] Loading datasets...")
    train_ds, val_ds, class_names = get_datasets()
    print(f"[train_transfer] Classes: {class_names}")

    print("[train_transfer] Building transfer-learning model (MobileNetV2)...")
    model = build_transfer_model(num_classes=len(class_names))
    model.summary()

    print("[train_transfer] Starting training...")
    history, model_path, weights_traj_path = compile_and_train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        run_name=RUN_NAME,
    )

    print(f"[train_transfer] Training complete.")
    print(f"[train_transfer] Best model saved to: {model_path}")
    print(f"[train_transfer] Weight trajectory saved to: {weights_traj_path}")


if __name__ == "__main__":
    main()

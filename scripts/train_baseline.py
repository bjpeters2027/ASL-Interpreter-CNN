# scripts/train_baseline.py

import os

from src.dataset import get_datasets
from src.model_baseline import build_baseline_cnn
from src.train import compile_and_train
from src.config import CONFIG


RUN_NAME = "baseline_cnn"


def main():
    os.makedirs(CONFIG.experiments_dir, exist_ok=True)
    os.makedirs(CONFIG.model_dir, exist_ok=True)
    os.makedirs(CONFIG.logs_dir, exist_ok=True)

    print("[train_baseline] Loading datasets...")
    train_ds, val_ds, class_names = get_datasets()
    print(f"[train_baseline] Classes: {class_names}")

    print("[train_baseline] Building baseline CNN...")
    model = build_baseline_cnn(num_classes=len(class_names))
    model.summary()

    print("[train_baseline] Starting training...")
    history, model_path, weights_traj_path = compile_and_train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        run_name=RUN_NAME,
    )

    print(f"[train_baseline] Training complete.")
    print(f"[train_baseline] Best model saved to: {model_path}")
    print(f"[train_baseline] Weight trajectory saved to: {weights_traj_path}")


if __name__ == "__main__":
    main()

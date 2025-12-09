# scripts/evaluate_model.py

import os

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import CONFIG
from src.dataset import get_datasets
from src.evaluate import evaluate_model, plot_confusion_matrix


# Change this to "baseline_cnn" if you want to evaluate that instead
RUN_NAME = "transfer_mobilenetv2"


def main():
    print("[evaluate_model] Loading datasets...")
    train_ds, val_ds, class_names = get_datasets()
    print(f"[evaluate_model] Classes: {class_names}")

    model_path = os.path.join(CONFIG.model_dir, f"{RUN_NAME}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Have you run the corresponding train script yet?"
        )

    print(f"[evaluate_model] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print("[evaluate_model] Evaluating on validation set...")
    y_true, y_pred, cm = evaluate_model(model, val_ds, class_names, print_report=True)

    print("[evaluate_model] Plotting confusion matrix...")
    fig, ax = plot_confusion_matrix(
        cm,
        class_names,
        title=f"Confusion Matrix - {RUN_NAME}",
    )

    # Optional: save the confusion matrix figure
    plots_dir = os.path.join(CONFIG.experiments_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"confusion_matrix_{RUN_NAME}.png")
    fig.savefig(out_path, dpi=300)
    print(f"[evaluate_model] Confusion matrix saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()

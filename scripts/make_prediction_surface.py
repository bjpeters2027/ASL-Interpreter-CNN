# scripts/make_prediction_surface.py

import os

from src.config import CONFIG
from src.prediction_surface import plot_prediction_surface


RUN_NAME = "transfer_mobilenetv2"


def main():
    print(f"[make_prediction_surface] Creating prediction surface for run: {RUN_NAME}...")

    plots_dir = os.path.join(CONFIG.experiments_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"prediction_surface_{RUN_NAME}.png")

    plot_prediction_surface(
        run_name=RUN_NAME,
        grid_size=25,
        alpha_scale=3.0,
        max_batches_for_pca=10,
        out_path=out_path,
    )

    print(f"[make_prediction_surface] Finished. Saved to {out_path}")


if __name__ == "__main__":
    main()

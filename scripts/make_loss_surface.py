# scripts/make_loss_surface.py

import os

from src.config import CONFIG
from src.loss_surface import plot_loss_surface


# Choose which trained model/run to visualize:
# "baseline_cnn" or "transfer_mobilenetv2"
RUN_NAME = "transfer_mobilenetv2"


def main():
    print(f"[make_loss_surface] Creating loss surface for run: {RUN_NAME}...")

    plots_dir = os.path.join(CONFIG.experiments_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"loss_surface_{RUN_NAME}.png")

    plot_loss_surface(
        run_name=RUN_NAME,
        grid_size=21,          # resolution of the grid
        alpha_scale=1.0,       # scale of PCA directions
        max_batches=3,         # speed vs accuracy of loss estimate
        out_path=out_path,
    )

    print(f"[make_loss_surface] Finished. Saved to {out_path}")


if __name__ == "__main__":
    main()

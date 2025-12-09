# src/loss_surface.py

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .config import CONFIG
from .dataset import get_datasets


def _build_model_for_run(run_name: str) -> tf.keras.Model:
    """
    Loads the saved best model for the given run_name.
    """
    model_path = os.path.join(CONFIG.model_dir, f"{run_name}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Make sure you have trained the run '{run_name}' first."
        )
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model


def _get_weight_shapes(model: tf.keras.Model):
    weights = model.get_weights()
    shapes = [w.shape for w in weights]
    sizes = [int(np.prod(s)) for s in shapes]
    total_params = sum(sizes)
    return shapes, sizes, total_params


def _flat_to_weights(flat: np.ndarray, shapes, sizes):
    weights = []
    idx = 0
    for shape, size in zip(shapes, sizes):
        segment = flat[idx: idx + size]
        weights.append(segment.reshape(shape))
        idx += size
    return weights


def _compute_loss(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    max_batches: int,
) -> float:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    total_loss = 0.0
    n_batches = 0

    for i, (x, y) in enumerate(val_ds):
        if i >= max_batches:
            break
        preds = model(x, training=False)
        batch_loss = loss_fn(y, preds).numpy()
        total_loss += batch_loss
        n_batches += 1

    if n_batches == 0:
        return 0.0
    return total_loss / n_batches


def plot_loss_surface(
    run_name: str,
    grid_size: int = 21,
    alpha_scale: float = 1.0,
    max_batches: int = 3,
    out_path: str = None,
):
    """
    Plots a 3D loss surface over two PCA directions in parameter space.

    Args:
        run_name: name used during training (e.g., "baseline_cnn" or "transfer_mobilenetv2")
        grid_size: resolution of the 2D grid in PCA space
        alpha_scale: how far to move along each PCA direction
        max_batches: how many validation batches to use per grid point
        out_path: where to save the figure (PNG). If None, just shows the plot.
    """

    # 1. Load weight trajectory
    weights_path = os.path.join(CONFIG.logs_dir, f"{run_name}_weights.npy")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weight trajectory not found at {weights_path}. "
            f"Make sure you have trained '{run_name}' with the new train.py."
        )

    W = np.load(weights_path)  # shape (T, D)
    print(f"[LossSurface] Loaded weight trajectory: T={W.shape[0]}, D={W.shape[1]}")

    # 2. Load model & weight shapes
    model = _build_model_for_run(run_name)
    shapes, sizes, total_params = _get_weight_shapes(model)

    if W.shape[1] != total_params:
        raise ValueError(
            f"Mismatch between logged weights (D={W.shape[1]}) "
            f"and model parameter count (D={total_params})."
        )

    # 3. Do PCA on the weight trajectory
    W_centered = W - W.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2)
    pca.fit(W_centered)
    d1, d2 = pca.components_  # each shape (D,)

    # Use final weight vector as center point
    w_center = W[-1]

    # 4. Prepare data & grid
    _, val_ds, _ = get_datasets()

    alphas = np.linspace(-alpha_scale, alpha_scale, grid_size)
    betas = np.linspace(-alpha_scale, alpha_scale, grid_size)
    Z = np.zeros((grid_size, grid_size), dtype=np.float32)

    print("[LossSurface] Evaluating loss on grid...")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            w_flat = w_center + a * d1 + b * d2
            new_weights = _flat_to_weights(w_flat.astype(np.float32), shapes, sizes)
            model.set_weights(new_weights)
            Z[i, j] = _compute_loss(model, val_ds, max_batches=max_batches)

    # 5. Plot
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(A, B, Z, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_xlabel("PCA direction 1 (α)")
    ax.set_ylabel("PCA direction 2 (β)")
    ax.set_zlabel("Loss")
    ax.set_title(f"Loss Surface for {run_name}")

    fig.colorbar(surf, shrink=0.5, aspect=10)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300)
        print(f"[LossSurface] Saved figure to {out_path}")
    else:
        plt.show()

    plt.close(fig)

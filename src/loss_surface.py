import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf

from .config import CONFIG
from .dataset import get_datasets


def plot_loss_surface(
    run_name: str,
    grid_size: int = 21,
    alpha_scale: float = 1.0,
    max_batches: int = 5,
    out_path: Optional[str] = None,
):
    """
    Constructs a 3D loss surface over a 2D PCA subspace in weight space.

    Steps:
      1. Load weight trajectory from training (epochs+1, num_params).
      2. Fit PCA(2) on these flattened weight vectors.
      3. Define a grid in (PC1, PC2) space.
      4. For each grid point, reconstruct a weight vector and set it to the model.
      5. Evaluate loss on a small subset of data.
      6. Plot the resulting surface; overlay the training trajectory (projected).
    """
    weights_traj_path = os.path.join(CONFIG.logs_dir, f"{run_name}_weights.npy")
    model_path = os.path.join(CONFIG.model_dir, f"{run_name}.keras")

    if not os.path.exists(weights_traj_path):
        raise FileNotFoundError(
            f"Weight trajectory file not found: {weights_traj_path}. "
            "Make sure you trained with this run_name first."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first."
        )

    # Load trajectory of weights: shape (T, num_params)
    weights_traj = np.load(weights_traj_path)  # (T, D)
    T, D = weights_traj.shape
    print(f"[LossSurface] Loaded weight trajectory: T={T}, D={D}")

    # Fit PCA(2) to the trajectory
    pca = PCA(n_components=2)
    traj_2d = pca.fit_transform(weights_traj)  # (T, 2)

    # We'll use the final weight vector as the center point in PCA space
    center_2d = traj_2d[-1]  # (2,)
    center_w = weights_traj[-1]  # (D,)

    # Build grid in PCA space around center
    # e.g. alpha,beta in [-scale, +scale]
    alphas = np.linspace(-alpha_scale, alpha_scale, grid_size)
    betas = np.linspace(-alpha_scale, alpha_scale, grid_size)
    A, B = np.meshgrid(alphas, betas)

    # Precompute directions in original weight space for basis e1, e2
    # PCA components_: shape (2, D)
    e1 = pca.components_[0]
    e2 = pca.components_[1]

    # Prepare dataset subset for efficient loss computation
    train_ds, val_ds, _ = get_datasets()
    subset = val_ds.take(max_batches)  # small subset to approximate loss

    # Load model
    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    Z = np.zeros_like(A)

    # Helper: set flattened weights into the model
    def set_flat_weights(flat_w: np.ndarray):
        offset = 0
        new_weights = []
        for w in model.trainable_weights:
            shape = w.shape
            size = np.prod(shape)
            chunk = flat_w[offset:offset + size]
            offset += size
            new_weights.append(chunk.reshape(shape))
        model.set_weights(new_weights + model.get_weights()[len(new_weights):])

    # Evaluate loss at each grid point
    for i in range(grid_size):
        for j in range(grid_size):
            a = A[i, j]
            b = B[i, j]
            w_flat = center_w + a * e1 + b * e2
            set_flat_weights(w_flat.astype(np.float32))

            # Compute loss on subset
            losses = []
            for images, labels in subset:
                loss_val = model.evaluate(
                    images, labels, verbose=0, return_dict=True
                )["loss"]
                losses.append(loss_val)
            Z[i, j] = np.mean(losses)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z, cmap="viridis", alpha=0.8)

    # Overlay trajectory in PCA coordinates (PC1, PC2, loss along path)
    # Approximate each point's loss using the nearest grid point in (A,B)
    traj_loss = []
    for t in range(T):
        x, y = traj_2d[t]
        # Find nearest grid index
        i = (np.abs(alphas - (x - center_2d[0])).argmin())
        j = (np.abs(betas - (y - center_2d[1])).argmin())
        traj_loss.append(Z[j, i])  # note A[i,j],B[i,j] -> Z[j,i]
    traj_loss = np.array(traj_loss)

    ax.plot(
        traj_2d[:, 0] - center_2d[0],
        traj_2d[:, 1] - center_2d[1],
        traj_loss,
        color="red",
        marker="o",
        label="Training trajectory",
    )

    ax.set_xlabel("PC1 (alpha)")
    ax.set_ylabel("PC2 (beta)")
    ax.set_zlabel("Loss")
    ax.set_title(f"Loss Surface in PCA Weight Space - {run_name}")
    ax.legend()

    plt.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"[LossSurface] Saved figure to {out_path}")
    else:
        plt.show()

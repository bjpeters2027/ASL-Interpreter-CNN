from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf

from .config import CONFIG
from .dataset import get_datasets


def plot_prediction_surface(
    run_name: str,
    grid_size: int = 25,
    alpha_scale: float = 3.0,
    max_batches_for_pca: int = 10,
    out_path: Optional[str] = None,
):
    """
    Builds a 3D prediction surface over a 2D PCA subspace in INPUT space.

    Steps:
      1. Collect a set of input images from the dataset, flatten each to a vector.
      2. Fit PCA(2) to these flattened images.
      3. Take the mean image (in PCA space) as x0.
      4. For a grid in (PC1, PC2), reconstruct an image via inverse transform.
      5. Feed reconstructed images into the model and record the predicted score
         for some class (here: the argmax class at center).
      6. Plot the resulting 3D surface.
    """
    from .config import CONFIG  # local import to avoid cycles

    model_path = f"experiments/models/{run_name}.keras"
    model = tf.keras.models.load_model(model_path)

    train_ds, val_ds, class_names = get_datasets()

    # --- Collect a bunch of images for PCA ---
    X_list = []
    for batch_idx, (images, _) in enumerate(train_ds):
        # images: (B, H, W, C)
        arr = images.numpy()  # 0..255
        arr = arr / 255.0
        B = arr.shape[0]
        X_list.append(arr.reshape(B, -1))  # (B, H*W*C)

        if batch_idx + 1 >= max_batches_for_pca:
            break

    X = np.concatenate(X_list, axis=0)  # (N, D)
    print(f"[PredictionSurface] Collected {X.shape[0]} images for PCA in input space.")

    # --- Fit PCA(2) to input images ---
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)  # (N, 2)

    # We'll take the mean in PCA space as center
    center_2d = X_2d.mean(axis=0)  # (2,)
    center_img_flat = pca.inverse_transform(center_2d)  # (D,)

    # Reshape to image
    H, W, C = CONFIG.img_height, CONFIG.img_width, CONFIG.num_channels
    center_img = center_img_flat.reshape(H, W, C)
    center_img = np.clip(center_img, 0.0, 1.0)

    # Determine which class we will visualize probability for
    center_pred = model.predict(center_img[None, ...], verbose=0)
    target_class = int(np.argmax(center_pred, axis=1)[0])
    target_letter = class_names[target_class]
    print(f"[PredictionSurface] Visualizing probability for class {target_letter}.")

    # Build grid in PCA space (PC1, PC2) around center
    alphas = np.linspace(-alpha_scale, alpha_scale, grid_size)
    betas = np.linspace(-alpha_scale, alpha_scale, grid_size)
    A, B = np.meshgrid(alphas, betas)

    Z = np.zeros_like(A)

    for i in range(grid_size):
        for j in range(grid_size):
            # Coordinates in PCA space
            coord_2d = center_2d + np.array([A[i, j], B[i, j]])
            img_flat = pca.inverse_transform(coord_2d)
            img = img_flat.reshape(H, W, C)
            img = np.clip(img, 0.0, 1.0)

            preds = model.predict(img[None, ...], verbose=0)
            Z[i, j] = preds[0, target_class]  # probability for target class

    # Plot surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z, cmap="plasma", alpha=0.9)

    ax.set_xlabel("PC1 direction in input space")
    ax.set_ylabel("PC2 direction in input space")
    ax.set_zlabel(f"P(class = {target_letter})")
    ax.set_title(f"Prediction Surface in PCA Input Space - {run_name}")

    plt.tight_layout()

    if out_path is not None:
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"[PredictionSurface] Saved figure to {out_path}")
    else:
        plt.show()

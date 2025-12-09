import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import CONFIG


def _flatten_model_weights(model: tf.keras.Model) -> np.ndarray:
    """
    Flattens all trainable weights into a single 1D numpy array.
    """
    flat_tensors = [tf.reshape(w, [-1]) for w in model.trainable_weights]
    flat = tf.concat(flat_tensors, axis=0)
    return flat.numpy()


class WeightLoggerCallback(tf.keras.callbacks.Callback):
    """
    Logs flattened weights at each epoch into a .npy file so that
    we can later do PCA over the training trajectory (for the loss surface).
    """

    def __init__(self, run_name: str):
        super().__init__()
        self.run_name = run_name
        self._weights_history = []

    def on_train_begin(self, logs=None):
        # Log initial weights at epoch 0
        if self.model is not None:
            self._weights_history.append(_flatten_model_weights(self.model))

    def on_epoch_end(self, epoch, logs=None):
        if self.model is not None:
            self._weights_history.append(_flatten_model_weights(self.model))

    def on_train_end(self, logs=None):
        if not self._weights_history:
            return
        os.makedirs(CONFIG.logs_dir, exist_ok=True)
        path = os.path.join(CONFIG.logs_dir, f"{self.run_name}_weights.npy")
        arr = np.stack(self._weights_history, axis=0)  # shape: (epochs+1, num_params)
        np.save(path, arr)
        print(f"[WeightLogger] Saved weight trajectory to {path}")


def compile_and_train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    run_name: str,
) -> Tuple[tf.keras.callbacks.History, str, str]:
    """
    Compiles and trains a model with standard settings.

    Returns:
      - history  (Keras History object)
      - model_path (where best weights are saved)
      - weights_traj_path (where flattened weights history is saved)
    """
    os.makedirs(CONFIG.model_dir, exist_ok=True)
    os.makedirs(CONFIG.logs_dir, exist_ok=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path = os.path.join(CONFIG.model_dir, f"{run_name}.keras")
    csv_log_path = os.path.join(CONFIG.logs_dir, f"{run_name}.csv")

    weight_logger = WeightLoggerCallback(run_name=run_name)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(csv_log_path),
        weight_logger,
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG.epochs,
        callbacks=callbacks,
    )

    weights_traj_path = os.path.join(CONFIG.logs_dir, f"{run_name}_weights.npy")
    return history, model_path, weights_traj_path

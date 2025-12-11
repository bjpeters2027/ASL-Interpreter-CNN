# src/train.py

import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import CONFIG


class WeightLoggerCallback(tf.keras.callbacks.Callback):

    def __init__(self, run_name: str):
        super().__init__()
        self.run_name = run_name
        self._weights_list = []

    def _flatten_weights(self) -> np.ndarray:
        weights = self.model.get_weights()
        flat = np.concatenate([w.reshape(-1) for w in weights])
        return flat

    def on_train_begin(self, logs=None):
        self._weights_list.append(self._flatten_weights())

    def on_epoch_end(self, epoch, logs=None):
        self._weights_list.append(self._flatten_weights())

    def on_train_end(self, logs=None):
        if not self._weights_list:
            return
        weights_arr = np.stack(self._weights_list, axis=0)  # (T, D)
        os.makedirs(CONFIG.logs_dir, exist_ok=True)
        out_path = os.path.join(CONFIG.logs_dir, f"{self.run_name}_weights.npy")
        np.save(out_path, weights_arr)
        print(f"[WeightLogger] Saved weight trajectory to {out_path}")


def compile_and_train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    run_name: str,
) -> Tuple[tf.keras.callbacks.History, str, str]:

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    os.makedirs(CONFIG.model_dir, exist_ok=True)
    best_model_path = os.path.join(CONFIG.model_dir, f"{run_name}.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=False,
            verbose=1,
        ),
        WeightLoggerCallback(run_name),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG.epochs,
        callbacks=callbacks,
    )

    weights_path = os.path.join(CONFIG.logs_dir, f"{run_name}_weights.npy")

    return history, best_model_path, weights_path

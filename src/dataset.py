# src/dataset.py

from typing import Tuple, List
import tensorflow as tf
from .config import CONFIG


def get_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Loads training and validation datasets from a single directory with
    subfolders per class, e.g.:

        data/
          A/
          B/
          ...
          Y/

    We let Keras internally split into train/validation using validation_split.
    """

    # First create the raw training dataset
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG.data_dir,                     # root = "data/"
        labels="inferred",
        label_mode="int",
        image_size=(CONFIG.img_height, CONFIG.img_width),
        batch_size=CONFIG.batch_size,
        shuffle=True,
        validation_split=0.20,              # 80% train
        subset="training",
        seed=1337,                          # same seed for train/val
    )

    # Grab the class names BEFORE caching/prefetching
    class_names: List[str] = raw_train_ds.class_names  # e.g., ['A', 'B', ..., 'Y']

    # Create the raw validation dataset
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG.data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(CONFIG.img_height, CONFIG.img_width),
        batch_size=CONFIG.batch_size,
        shuffle=False,
        validation_split=0.20,              # 20% validation
        subset="validation",
        seed=1337,
    )

    # Now apply cache/prefetch to both
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

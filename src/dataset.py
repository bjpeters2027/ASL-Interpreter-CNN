from typing import Tuple, List
import tensorflow as tf
from .config import CONFIG


def get_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Loads train and validation datasets from directory structure:

        data/train/A, ..., data/train/Z
        data/val/A,   ..., data/val/Z
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG.train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(CONFIG.img_height, CONFIG.img_width),
        batch_size=CONFIG.batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG.val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(CONFIG.img_height, CONFIG.img_width),
        batch_size=CONFIG.batch_size,
        shuffle=False
    )

    # Optional performance tweaks
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    class_names = train_ds.class_names  # ['A', 'B', ..., 'Z']

    return train_ds, val_ds, class_names

# src/dataset.py

from typing import Tuple, List
import tensorflow as tf
from .config import CONFIG


def get_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:


    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG.data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(CONFIG.img_height, CONFIG.img_width),
        batch_size=CONFIG.batch_size,
        shuffle=True,
        validation_split=0.20,
        subset="training",
        seed=1337,
    )

    class_names: List[str] = raw_train_ds.class_names

    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG.data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(CONFIG.img_height, CONFIG.img_width),
        batch_size=CONFIG.batch_size,
        shuffle=False,
        validation_split=0.20,
        subset="validation",
        seed=1337,
    )


    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

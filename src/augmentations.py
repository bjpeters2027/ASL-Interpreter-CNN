import tensorflow as tf


def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Returns a Keras Sequential of common augmentations.
    Adjust as needed depending on how varied your real data is.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation"
    )

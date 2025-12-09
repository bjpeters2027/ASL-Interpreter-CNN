import tensorflow as tf
from .config import CONFIG
from .augmentations import build_augmentation_layer


def build_baseline_cnn(num_classes: int = CONFIG.num_classes) -> tf.keras.Model:
    """
    A small custom CNN for 26-class ASL classification.
    """
    inputs = tf.keras.Input(
        shape=(CONFIG.img_height, CONFIG.img_width, CONFIG.num_channels)
    )

    x = build_augmentation_layer()(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="asl_baseline_cnn")
    return model

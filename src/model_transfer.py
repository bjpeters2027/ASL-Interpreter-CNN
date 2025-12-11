import tensorflow as tf
from .config import CONFIG
from .augmentations import build_augmentation_layer


def build_transfer_model(num_classes: int = CONFIG.num_classes) -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=(CONFIG.img_height, CONFIG.img_width, CONFIG.num_channels)
    )

    x = build_augmentation_layer()(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(CONFIG.img_height, CONFIG.img_width, CONFIG.num_channels),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="asl_transfer_mobilenetv2")
    return model

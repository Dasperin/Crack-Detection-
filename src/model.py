"""Model definitions.

Architecture mirrors the original project (MobileNetV2 transfer learning,
frozen ImageNet base + small classification head, Adam + binary
cross-entropy) with one deliberate improvement: GlobalAveragePooling2D is
used instead of Flatten before the dense head.

Why this matters: MobileNetV2's last conv block outputs a 7x7x1280 tensor.
Flatten() turns that into a 62,720-length vector, so a Dense(128) layer
right after it alone has ~8M parameters - on a 40k-image dataset that is a
significant overfitting risk and is why the original run was capped at very
few epochs. GlobalAveragePooling2D instead collapses each of the 1280
channels to a single average value (a 1280-length vector), cutting that
first dense layer down to ~164k parameters (~98% fewer) while keeping the
same channel-wise feature information. This is the approach used in
Keras/TensorFlow's own transfer-learning tutorials for exactly this reason.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

from config import IMG_SIZE, LEARNING_RATE


def build_head_model(input_dim=1280, dropout=0.3):
    """Small classifier head trained on cached bottleneck features."""
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu", name="dense_128"),
            layers.Dropout(dropout, name="dropout"),
            layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="crack_classifier_head",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_full_model(head_weights=None, dropout=0.3):
    """End-to-end image -> prediction model (frozen base + trained head).

    Used for evaluation on raw images, Grad-CAM, and single-image inference
    - anywhere we need the real MobileNetV2 activations rather than cached
    features.
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="surface_crack_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    if head_weights is not None:
        model.get_layer("dense_128").set_weights(head_weights["dense_128"])
        model.get_layer("output").set_weights(head_weights["output"])

    return model, base

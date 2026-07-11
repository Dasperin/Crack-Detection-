"""CLI: predict Crack / No Crack for a single image.

Usage:
    python predict.py path/to/image.jpg
"""

import sys

import numpy as np
import tensorflow as tf

from config import IMG_SIZE, MODELS_DIR
import os


def predict(image_path):
    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "surface_crack_mobilenetv2.keras")
    )
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(arr, axis=0))

    prob = float(model.predict(arr, verbose=0)[0, 0])
    label = "Crack" if prob >= 0.5 else "No Crack"
    confidence = prob if prob >= 0.5 else 1 - prob
    print(f"Prediction: {label}  (confidence: {confidence:.2%}, raw score: {prob:.4f})")
    return label, prob


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
    predict(sys.argv[1])

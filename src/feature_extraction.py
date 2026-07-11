"""Turn raw images into cached MobileNetV2 bottleneck features.

Since the MobileNetV2 base is frozen (`trainable=False`), its output for a
given image never changes during training. Instead of re-running the same
frozen forward pass through every image on every epoch, we run it ONCE per
image, cache the pooled 1280-d feature vector, and train the small
classifier head on those cached vectors. This is the classic Keras
"bottleneck features" transfer-learning trick and is what makes it feasible
to train on the full 40,000-image dataset on a laptop CPU instead of a GPU.
"""

import io
import os
import time

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import BATCH_SIZE, DATA_RAW_DIR, FEATURES_DIR, IMG_SIZE, SAMPLES_DIR


def build_feature_extractor():
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_feature_extractor")


def _decode_resize(raw_bytes):
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32)


def extract_split(split, extractor, save_n_samples=0, row_group_batch=4):
    parquet_path = os.path.join(DATA_RAW_DIR, "default", split, "0000.parquet")
    pf = pq.ParquetFile(parquet_path)

    all_features = []
    all_labels = []
    saved_positive, saved_negative = 0, 0
    t0 = time.time()
    n_seen = 0

    row_groups = list(range(pf.num_row_groups))
    for i in range(0, len(row_groups), row_group_batch):
        group_idxs = row_groups[i : i + row_group_batch]
        table = pf.read_row_groups(group_idxs, columns=["image", "label"])
        images_col = table.column("image").to_pylist()
        labels_col = table.column("label").to_pylist()

        batch_imgs = np.stack([_decode_resize(rec["bytes"]) for rec in images_col])
        batch_labels = np.array(labels_col, dtype=np.int64)

        if save_n_samples:
            for img_arr, lbl in zip(batch_imgs, batch_labels):
                if lbl == 1 and saved_positive < save_n_samples:
                    Image.fromarray(img_arr.astype(np.uint8)).save(
                        os.path.join(SAMPLES_DIR, "Positive", f"{split}_{saved_positive}.jpg")
                    )
                    saved_positive += 1
                elif lbl == 0 and saved_negative < save_n_samples:
                    Image.fromarray(img_arr.astype(np.uint8)).save(
                        os.path.join(SAMPLES_DIR, "Negative", f"{split}_{saved_negative}.jpg")
                    )
                    saved_negative += 1

        preprocessed = preprocess_input(batch_imgs.copy())
        features = extractor.predict(preprocessed, batch_size=BATCH_SIZE, verbose=0)

        all_features.append(features)
        all_labels.append(batch_labels)
        n_seen += len(batch_labels)
        elapsed = time.time() - t0
        print(
            f"[{split}] {n_seen} images processed "
            f"({elapsed:.1f}s elapsed, {n_seen/elapsed:.1f} img/s)",
            flush=True,
        )

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAMPLES_DIR, "Positive"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLES_DIR, "Negative"), exist_ok=True)

    extractor = build_feature_extractor()

    for split, n_samples in [("train", 0), ("validation", 0), ("test", 25)]:
        print(f"\n=== Extracting features for split: {split} ===")
        X, y = extract_split(split, extractor, save_n_samples=n_samples)
        out_path = os.path.join(FEATURES_DIR, f"{split}.npz")
        np.savez_compressed(out_path, X=X, y=y)
        pos = int(y.sum())
        print(f"[{split}] saved {X.shape[0]} feature vectors -> {out_path}")
        print(f"[{split}] class balance: {pos} positive / {len(y) - pos} negative")


if __name__ == "__main__":
    main()

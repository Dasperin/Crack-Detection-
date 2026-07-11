"""Train the classification head on cached MobileNetV2 bottleneck features,
then assemble and save the full end-to-end (image -> prediction) model.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import FEATURES_DIR, HEAD_EPOCHS, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from model import build_full_model, build_head_model

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_features(split):
    data = np.load(os.path.join(FEATURES_DIR, f"{split}.npz"))
    return data["X"], data["y"].astype(np.float32)


def plot_training_curves(history, out_path):
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs_range, hist["accuracy"], label="Training accuracy", color="tab:blue")
    axes[0].plot(
        epochs_range, hist["val_accuracy"], label="Validation accuracy", color="tab:orange"
    )
    axes[0].set_title("Model accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, hist["loss"], label="Training loss", color="tab:blue")
    axes[1].plot(epochs_range, hist["val_loss"], label="Validation loss", color="tab:orange")
    axes[1].set_title("Model loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (binary cross-entropy)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Surface Crack Detection - MobileNetV2 Transfer Learning")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[train] saved training curves -> {out_path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    X_train, y_train = load_features("train")
    X_val, y_val = load_features("validation")
    print(f"[train] train features: {X_train.shape}, val features: {X_val.shape}")

    head = build_head_model(input_dim=X_train.shape[1])
    head.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = head.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=HEAD_EPOCHS,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_curves(history, os.path.join(OUTPUTS_DIR, "training_curves.png"))

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    best_val_acc = max(history.history["val_accuracy"])
    epochs_ran = len(history.history["loss"])

    metrics = {
        "epochs_ran": epochs_ran,
        "final_train_accuracy": float(final_train_acc),
        "final_val_accuracy": float(final_val_acc),
        "best_val_accuracy": float(best_val_acc),
    }
    with open(os.path.join(OUTPUTS_DIR, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[train] metrics:", metrics)

    # Save the small head on its own (fast to reload for further experiments).
    head.save(os.path.join(MODELS_DIR, "head_model.keras"))

    # Assemble the full image -> prediction model and save it - this is the
    # artifact predict.py / gradcam.py / evaluate.py load.
    head_weights = {
        "dense_128": head.get_layer("dense_128").get_weights(),
        "output": head.get_layer("output").get_weights(),
    }
    full_model, _ = build_full_model(head_weights=head_weights)
    full_model.save(os.path.join(MODELS_DIR, "surface_crack_mobilenetv2.keras"))
    print(
        "[train] saved full end-to-end model -> "
        f"{os.path.join(MODELS_DIR, 'surface_crack_mobilenetv2.keras')}"
    )


if __name__ == "__main__":
    main()

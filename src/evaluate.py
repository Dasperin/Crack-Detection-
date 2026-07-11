"""Evaluate the trained model on the held-out test set (never seen during
training or validation) and produce a confusion matrix + ROC curve +
classification report - the metrics that should actually be quoted.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from config import CLASS_NAMES, FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR


def load_features(split):
    data = np.load(os.path.join(FEATURES_DIR, f"{split}.npz"))
    return data["X"], data["y"].astype(np.float32)


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    head = tf.keras.models.load_model(os.path.join(MODELS_DIR, "head_model.keras"))
    X_test, y_test = load_features("test")
    print(f"[evaluate] test features: {X_test.shape}")

    y_prob = head.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    test_loss, test_acc = head.evaluate(X_test, y_test, verbose=0)
    print(f"[evaluate] test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix (held-out test set, n=%d)" % len(y_test))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})", color="tab:blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Test Set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test_samples": int(len(y_test)),
    }
    with open(os.path.join(OUTPUTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[evaluate] ROC-AUC: {auc:.4f}")
    print("[evaluate] confusion matrix:\n", cm)
    print(f"[evaluate] saved plots + metrics to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()

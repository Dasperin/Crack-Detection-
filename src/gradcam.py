"""Grad-CAM interpretability for the trained crack detector.

This is the main "beyond the original slides" addition: rather than only
reporting a predicted class, we visualize *where* in the image the model
looked to make that decision, by backpropagating the predicted class score
to the last MobileNetV2 conv block and overlaying the resulting activation
map on the original image. For a crack detector this is a meaningful,
practically-useful upgrade - it turns a black-box classifier into something
that can visually justify itself, which is exactly what you'd want before
trusting it on a production inspection line.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, IMG_SIZE, MODELS_DIR, OUTPUTS_DIR, SAMPLES_DIR

LAST_CONV_LAYER = "Conv_1"  # last conv block inside MobileNetV2 (before pooling)


def load_model():
    return tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "surface_crack_mobilenetv2.keras")
    )


def make_gradcam_heatmap(model, img_array, base_layer_name="mobilenetv2_1.00_224"):
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Could not find nested MobileNetV2 base model.")

    conv_layer = base_model.get_layer(LAST_CONV_LAYER)

    grad_model = tf.keras.models.Model(base_model.inputs, [conv_layer.output, base_model.output])

    # Rebuild the head as a small functional stack we can differentiate through.
    gap = model.get_layer("gap")
    dense = model.get_layer("dense_128")
    dropout = model.get_layer("dropout")
    out = model.get_layer("output")

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_array)
        x = gap(base_output)
        x = dense(x)
        x = dropout(x, training=False)
        preds = out(x)
        class_score = preds[:, 0]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(preds[0, 0])


def overlay_heatmap(img, heatmap, alpha=0.45):
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = tf.keras.utils.array_to_img(
        tf.expand_dims(heatmap, -1), scale=False
    ).resize((IMG_SIZE, IMG_SIZE))
    heatmap_resized = np.asarray(heatmap_img)

    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_resized / 255.0)[:, :, :3] * 255
    overlaid = colored_heatmap * alpha + img * (1 - alpha)
    return np.clip(overlaid, 0, 255).astype(np.uint8)


def preprocess_for_display_and_model(path):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    display_img = img_array.copy()
    model_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(img_array, axis=0)
    )
    return display_img, model_input


def main(n_per_class=4):
    model = load_model()
    out_dir = os.path.join(OUTPUTS_DIR, "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(SAMPLES_DIR, cls)
        files = sorted(os.listdir(cls_dir))[:n_per_class]

        fig, axes = plt.subplots(2, n_per_class, figsize=(3.2 * n_per_class, 6.4))
        for i, fname in enumerate(files):
            path = os.path.join(cls_dir, fname)
            display_img, model_input = preprocess_for_display_and_model(path)
            heatmap, score = make_gradcam_heatmap(model, model_input)
            overlay = overlay_heatmap(display_img, heatmap)

            pred_label = "Positive" if score >= 0.5 else "Negative"
            axes[0, i].imshow(display_img.astype(np.uint8))
            axes[0, i].set_title(f"True: {cls}")
            axes[0, i].axis("off")

            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f"Pred: {pred_label} ({score:.3f})")
            axes[1, i].axis("off")

        fig.suptitle(f"Grad-CAM - {cls} samples (top: original, bottom: Grad-CAM overlay)")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"gradcam_{cls.lower()}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[gradcam] saved {out_path}")


if __name__ == "__main__":
    main()

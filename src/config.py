"""Central configuration for the surface crack detection project."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data source: HF mirror of Ozgenel's "Concrete Crack Images for Classification"
# (the same dataset referenced by the arthurflor23/surface-crack-detection repo
# used in the original project) - 40,000 images, 227x227, 2 balanced classes.
HF_DATASET_REPO = "mohammadnajeeb/concrete_crack_images"
HF_REVISION = "refs/convert/parquet"

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data_raw")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

IMG_SIZE = 224
CLASS_NAMES = ["Negative", "Positive"]  # label 0 = no crack, 1 = crack

BATCH_SIZE = 64
HEAD_EPOCHS = 40
LEARNING_RATE = 1e-3

RANDOM_SEED = 42

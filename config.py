"""
Waste Image Classification System
Skill Nexis Deep Learning Major Project - Configuration Module

This module defines central configuration constants, hyperparameters,
directory paths, class labels, and runtime environment settings.
"""

import os
from pathlib import Path

# -------------------------------------------------------------------------
# Project Base Paths
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
VAL_DATA_DIR = PROCESSED_DATA_DIR / "val"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.keras"
IMPROVED_MODEL_PATH = MODELS_DIR / "improved_model.keras"

REPORTS_DIR = BASE_DIR / "reports"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_PREDICTIONS_DIR = OUTPUTS_DIR / "sample_predictions"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure essential directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    OUTPUTS_DIR,
    SAMPLE_PREDICTIONS_DIR,
    NOTEBOOKS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Dataset & Class Settings
# -------------------------------------------------------------------------
# The five mandatory waste categories
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
NUM_CLASSES = len(CLASSES)

# Class index mappings
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# Friendly display names and UI icons
CLASS_METADATA = {
    "cardboard": {
        "display_name": "Cardboard",
        "icon": "📦",
        "bin_color": "#D2B48C",
        "recycling_tip": "Flatten boxes and remove plastic packing tape before placing in the recycling bin.",
    },
    "glass": {
        "display_name": "Glass",
        "icon": "🍾",
        "bin_color": "#2E8B57",
        "recycling_tip": "Rinse containers thoroughly. Remove metal or plastic lids and recycle separately.",
    },
    "metal": {
        "display_name": "Metal",
        "icon": "🥫",
        "bin_color": "#A9A9A9",
        "recycling_tip": "Clean aluminum/tin cans. Aluminum foil should be rolled into a clean ball.",
    },
    "paper": {
        "display_name": "Paper",
        "icon": "📰",
        "bin_color": "#4682B4",
        "recycling_tip": "Keep dry and clean. Do not recycle paper contaminated with food, oil, or grease.",
    },
    "plastic": {
        "display_name": "Plastic",
        "icon": "🧴",
        "bin_color": "#FFA500",
        "recycling_tip": "Check resin code (1-7). Rinse containers and crush bottles to save space.",
    },
}

# -------------------------------------------------------------------------
# Image & Hyperparameter Settings
# -------------------------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4

# Split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

# -------------------------------------------------------------------------
# Runtime Backend Configuration
# -------------------------------------------------------------------------
# Set matplotlib backend cache to avoid permission warnings on Windows
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mpl_cache"))

# Graceful backend selection for Keras 3
# If native TensorFlow runtime fails to initialize (e.g. Windows DLL AVX issues),
# Keras automatically falls back to Torch backend seamlessly.
def configure_keras_backend():
    """Configures the optimal Keras backend based on system capabilities."""
    if "KERAS_BACKEND" in os.environ:
        return os.environ["KERAS_BACKEND"]

    try:
        import tensorflow as tf
        # Validate that TensorFlow internal C++ runtime loads without DLL initialization error
        _ = tf.constant([1.0])
        os.environ["KERAS_BACKEND"] = "tensorflow"
        return "tensorflow"
    except Exception:
        # Fallback to PyTorch backend for Keras 3
        os.environ["KERAS_BACKEND"] = "torch"
        return "torch"

ACTIVE_BACKEND = configure_keras_backend()

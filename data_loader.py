"""
Waste Image Classification System
Skill Nexis Deep Learning Major Project - Data Loader Module

Handles dataset preparation, splitting, image preprocessing, data augmentation,
and class distribution analysis for the 5 waste categories:
Cardboard, Glass, Metal, Paper, and Plastic.
"""

import os
import shutil
import random
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter

import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure backend is configured
config.configure_keras_backend()
import keras
from keras import layers


def verify_image(image_path: Path) -> bool:
    """Verifies that an image file can be opened and decoded without corruption."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_dataset_splits(
    raw_dir: Path = config.RAW_DATA_DIR,
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio: float = config.VAL_SPLIT,
    test_ratio: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, Dict[str, int]]:
    """
    Splits raw dataset into train, validation, and test folders.
    Filters raw classes to include only the five required target categories.
    """
    random.seed(seed)
    np.random.seed(seed)

    splits = ["train", "val", "test"]
    split_counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in config.CLASSES} for s in splits}

    for split in splits:
        for cls in config.CLASSES:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    found_classes = [d.name.lower() for d in raw_dir.iterdir() if d.is_dir()]
    valid_classes = [c for c in config.CLASSES if c in found_classes]

    if not valid_classes:
        logger.warning(
            f"No valid category folders found in '{raw_dir}'. "
            f"Expected subdirectories: {config.CLASSES}"
        )
        return split_counts

    for cls in valid_classes:
        raw_class_dir = raw_dir / cls
        if not raw_class_dir.exists():
            for d in raw_dir.iterdir():
                if d.name.lower() == cls:
                    raw_class_dir = d
                    break

        all_files = [
            f for f in raw_class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]
        
        valid_files = [f for f in all_files if verify_image(f)]
        random.shuffle(valid_files)

        n_total = len(valid_files)
        if n_total == 0:
            continue

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_files = valid_files[:n_train]
        val_files = valid_files[n_train : n_train + n_val]
        test_files = valid_files[n_train + n_val :]

        file_splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        for split_name, files in file_splits.items():
            dest_dir = processed_dir / split_name / cls
            for file_path in files:
                dest_file = dest_dir / file_path.name
                if not dest_file.exists():
                    shutil.copy2(file_path, dest_file)
            split_counts[split_name][cls] = len(files)

    logger.info("Dataset splitting completed successfully.")
    return split_counts


class WasteImageDataset(keras.utils.PyDataset):
    """
    Robust, backend-agnostic dataset generator for waste images.
    Fully compatible with Keras 3 on TensorFlow and PyTorch backends without DLL dependencies.
    """
    def __init__(
        self,
        directory: Path,
        batch_size: int = config.BATCH_SIZE,
        img_size: Tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH),
        shuffle: bool = True,
        seed: int = config.RANDOM_SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.directory = Path(directory)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.filepaths: List[Path] = []
        self.labels: List[np.ndarray] = []

        for idx, cls in enumerate(config.CLASSES):
            cls_dir = self.directory / cls
            if cls_dir.exists():
                for f in sorted(cls_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                        self.filepaths.append(f)
                        one_hot = np.zeros(config.NUM_CLASSES, dtype=np.float32)
                        one_hot[idx] = 1.0
                        self.labels.append(one_hot)

        self.indices = np.arange(len(self.filepaths))
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for i in batch_idx:
            img_path = self.filepaths[i]
            label = self.labels[i]
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(self.img_size, Image.Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.float32)
                    batch_images.append(arr)
                    batch_labels.append(label)
            except Exception:
                batch_images.append(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32))
                batch_labels.append(label)

        return np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)


def load_datasets(
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    img_size: Tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH),
    seed: int = config.RANDOM_SEED,
) -> Tuple[WasteImageDataset, WasteImageDataset, WasteImageDataset]:
    """
    Loads train, validation, and test datasets using WasteImageDataset.
    Returns:
        (train_ds, val_ds, test_ds)
    """
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists() or not any(d.iterdir()):
            raise FileNotFoundError(
                f"Processed directory '{d}' is empty or missing. "
                "Ensure data has been split using prepare_dataset_splits()."
            )

    train_ds = WasteImageDataset(
        directory=train_dir,
        batch_size=batch_size,
        img_size=img_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = WasteImageDataset(
        directory=val_dir,
        batch_size=batch_size,
        img_size=img_size,
        shuffle=False,
    )

    test_ds = WasteImageDataset(
        directory=test_dir,
        batch_size=batch_size,
        img_size=img_size,
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


def generate_class_distribution_report(
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    save_path: Path = config.REPORTS_DIR / "class_distribution.png",
) -> pd.DataFrame:
    """
    Analyzes class distributions across train, val, and test splits,
    generates a visualization, and returns a DataFrame.
    """
    records = []
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = processed_dir / split
        for cls in config.CLASSES:
            cls_dir = split_dir / cls
            count = len(list(cls_dir.glob("*.*"))) if cls_dir.exists() else 0
            records.append({"Split": split.capitalize(), "Category": cls.capitalize(), "Count": count})

    df = pd.DataFrame(records)

    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("muted", n_colors=3)

    ax = sns.barplot(
        data=df,
        x="Category",
        y="Count",
        hue="Split",
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
    )

    plt.title("Waste Classification Dataset - Class Distribution Across Splits", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Waste Category", fontsize=12, fontweight="semibold")
    plt.ylabel("Number of Images", fontsize=12, fontweight="semibold")
    plt.legend(title="Dataset Split", frameon=True)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Class distribution chart saved to '{save_path}'.")

    return df


def create_sample_dataset(
    target_dir: Path = config.RAW_DATA_DIR,
    samples_per_class: int = 25,
) -> None:
    """
    Generates synthetic sample images for demonstration, testing, and dry-running
    the complete pipeline when offline or before the full real dataset is downloaded.
    """
    logger.info("Generating synthetic sample images for dry-run verification...")
    rng = np.random.default_rng(config.RANDOM_SEED)

    class_styles = {
        "cardboard": {"base_rgb": (180, 130, 80), "var": 25, "texture": "kraft"},
        "glass": {"base_rgb": (160, 210, 200), "var": 20, "texture": "translucent"},
        "metal": {"base_rgb": (190, 195, 205), "var": 30, "texture": "sheen"},
        "paper": {"base_rgb": (240, 240, 245), "var": 15, "texture": "fiber"},
        "plastic": {"base_rgb": (70, 160, 230), "var": 35, "texture": "molded"},
    }

    for cls in config.CLASSES:
        cls_dir = target_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        style = class_styles[cls]

        for i in range(samples_per_class):
            img_arr = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8)
            base = np.array(style["base_rgb"])
            noise = rng.integers(-style["var"], style["var"] + 1, size=img_arr.shape)
            color_mat = np.clip(base + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(color_mat)
            draw = ImageDraw.Draw(img)

            if style["texture"] == "kraft":
                for y in range(0, config.IMG_HEIGHT, 15):
                    draw.line([(0, y), (config.IMG_WIDTH, y)], fill=(140, 100, 60), width=2)
            elif style["texture"] == "translucent":
                draw.ellipse([40, 30, 180, 190], outline=(220, 255, 255), width=4)
                draw.line([(110, 30), (110, 190)], fill=(240, 255, 255), width=2)
            elif style["texture"] == "sheen":
                draw.rectangle([50, 40, 170, 180], outline=(230, 235, 245), width=5)
                draw.line([(50, 40), (170, 180)], fill=(255, 255, 255), width=3)
            elif style["texture"] == "fiber":
                for y in range(30, config.IMG_HEIGHT - 30, 20):
                    draw.line([(25, y), (config.IMG_WIDTH - 25, y)], fill=(180, 180, 190), width=1)
            elif style["texture"] == "molded":
                draw.arc([35, 35, 185, 185], start=0, end=270, fill=(255, 200, 50), width=6)
                draw.rectangle([95, 20, 125, 45], fill=(240, 100, 50))

            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            img.save(cls_dir / f"sample_{cls}_{i+1:03d}.jpg", quality=90)

    logger.info(f"Sample dataset generated in '{target_dir}' with {samples_per_class} images per class.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Waste Dataset Preparation and Analysis")
    parser.add_argument("--create-sample", action="store_true", help="Generate synthetic sample dataset for testing")
    parser.add_argument("--split", action="store_true", help="Split raw data into train/val/test splits")
    parser.add_argument("--report", action="store_true", help="Generate class distribution chart and report")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_dataset()
    if args.split:
        prepare_dataset_splits()
    if args.report:
        generate_class_distribution_report()

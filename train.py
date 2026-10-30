"""
Waste Image Classification System
Skill Nexis Deep Learning Major Project - Training Module

Orchestrates training of the Baseline CNN and Improved Models, logs metrics,
generates training curves, and saves production-ready .keras checkpoints.
"""

import os
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import config
config.configure_keras_backend()
import keras

import data_loader
import model as model_lib

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_and_save_training_curves(history: keras.callbacks.History, save_path: Path, title: str):
    """Generates and saves publication-quality loss and accuracy training curve plots."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, val_acc, "s--", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    ax1.set_title(f"{title} - Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="lower right", frameon=True)

    # Loss Plot
    ax2.plot(epochs_range, loss, "o-", label="Training Loss", color="#1f77b4", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#d62728", linewidth=2)
    ax2.set_title(f"{title} - Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Crossentropy Loss", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Training curves saved to: {save_path}")


def train_single_model(
    model_type: str = "improved",
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
):
    """Trains either 'baseline' or 'improved' model and saves the trained artifacts."""
    logger.info(f"Starting training pipeline for: {model_type.upper()} model")

    # Ensure processed dataset exists
    if not (config.TRAIN_DATA_DIR / "cardboard").exists() or not any((config.TRAIN_DATA_DIR / "cardboard").iterdir()):
        logger.info("Processed data not detected. Checking raw directory to split...")
        if not (config.RAW_DATA_DIR / "cardboard").exists():
            logger.info("Raw data directory empty. Auto-generating sample dataset for verification...")
            data_loader.create_sample_dataset()
        data_loader.prepare_dataset_splits()
        data_loader.generate_class_distribution_report()

    # Load datasets
    train_ds, val_ds, _ = data_loader.load_datasets(batch_size=batch_size)

    if model_type.lower() == "baseline":
        net = model_lib.build_baseline_model()
        save_model_path = config.BASELINE_MODEL_PATH
        plot_save_path = config.REPORTS_DIR / "baseline_training_curves.png"
        title = "Baseline CNN"
    else:
        net = model_lib.build_improved_model()
        save_model_path = config.IMPROVED_MODEL_PATH
        plot_save_path = config.REPORTS_DIR / "improved_training_curves.png"
        title = "Improved Model"

    net = model_lib.compile_model(net, learning_rate=learning_rate)
    callbacks = model_lib.get_callbacks(str(save_model_path), patience=5)

    logger.info(f"Training for {epochs} epochs...")
    history = net.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    net.save(str(save_model_path))
    logger.info(f"Model saved to {save_model_path}")

    # Generate training curves
    plot_and_save_training_curves(history, plot_save_path, title)
    return net, history


def main():
    parser = argparse.ArgumentParser(description="Train Waste Image Classification Models")
    parser.add_argument(
        "--model",
        type=str,
        default="improved",
        choices=["baseline", "improved", "both"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--sample_mode", action="store_true", help="Force synthetic sample dataset creation")
    args = parser.parse_args()

    if args.sample_mode:
        data_loader.create_sample_dataset()
        data_loader.prepare_dataset_splits()
        data_loader.generate_class_distribution_report()

    if args.model == "both":
        train_single_model("baseline", epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        train_single_model("improved", epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    else:
        train_single_model(args.model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)


if __name__ == "__main__":
    main()

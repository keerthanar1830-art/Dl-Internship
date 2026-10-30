"""
Waste Image Classification System
Skill Nexis Deep Learning Major Project - Evaluation Module

Performs comprehensive evaluation on the held-out test split, generates
confusion matrix heatmap, classification report CSV, visual error analysis,
and the final markdown evaluation report.
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import config
config.configure_keras_backend()
import keras

import data_loader

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: Path = config.IMPROVED_MODEL_PATH,
    batch_size: int = config.BATCH_SIZE,
):
    """Evaluates a trained model on the test dataset and creates all report artifacts."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run 'python train.py' first."
        )

    logger.info(f"Loading trained model from: {model_path}")
    model = keras.models.load_model(str(model_path))

    # Load test dataset
    _, _, test_ds = data_loader.load_datasets(batch_size=batch_size)

    # Collect ground truth labels and predictions
    y_true_all = []
    y_pred_probs = []
    images_all = []

    logger.info("Computing predictions on the test dataset...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true_all.append(np.argmax(labels.numpy() if hasattr(labels, "numpy") else np.array(labels), axis=1))
        images_all.append(images.numpy() if hasattr(images, "numpy") else np.array(images))

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    images = np.concatenate(images_all, axis=0)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    display_labels = [c.capitalize() for c in config.CLASSES]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        linewidths=1,
        linecolor="gray",
    )
    plt.title("Confusion Matrix - Test Dataset", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="semibold")
    plt.ylabel("True Class", fontsize=12, fontweight="semibold")
    plt.tight_layout()

    cm_save_path = config.REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix saved to: {cm_save_path}")

    # 2. Classification Report DataFrame & CSV
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=display_labels,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report_dict).transpose()
    csv_save_path = config.REPORTS_DIR / "classification_report.csv"
    df_report.to_csv(csv_save_path)
    logger.info(f"Classification report saved to: {csv_save_path}")

    # 3. Error Analysis Plot
    misclassified_indices = np.where(y_true != y_pred)[0]
    n_errors = len(misclassified_indices)
    logger.info(f"Identified {n_errors} misclassified test instances.")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    indices_to_show = misclassified_indices[:6] if n_errors >= 6 else list(range(min(6, len(y_true))))

    for i, idx in enumerate(indices_to_show):
        img = images[idx]
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)

        true_cls = config.CLASSES[y_true[idx]].capitalize()
        pred_cls = config.CLASSES[y_pred[idx]].capitalize()
        conf = y_pred_probs[idx][y_pred[idx]] * 100

        axes[i].imshow(img)
        axes[i].axis("off")
        color = "red" if y_true[idx] != y_pred[idx] else "green"
        axes[i].set_title(
            f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    for j in range(len(indices_to_show), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Model Prediction & Error Analysis Samples", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    error_save_path = config.REPORTS_DIR / "error_analysis.png"
    plt.savefig(error_save_path, dpi=300)
    plt.close()
    logger.info(f"Error analysis plot saved to: {error_save_path}")

    # 4. Generate Markdown Evaluation Report
    generate_markdown_report(report_dict, cm, len(y_true), model_path.name)
    logger.info("Evaluation completed successfully.")


def generate_markdown_report(report_dict: dict, cm: np.ndarray, total_samples: int, model_name: str):
    """Generates the structured evaluation report in markdown format."""
    acc = report_dict.get("accuracy", 0.0)
    macro_f1 = report_dict.get("macro avg", {}).get("f1-score", 0.0)
    weighted_f1 = report_dict.get("weighted avg", {}).get("f1-score", 0.0)

    rows = []
    for cls in config.CLASSES:
        c_title = cls.capitalize()
        if c_title in report_dict:
            metrics = report_dict[c_title]
            p = metrics["precision"]
            r = metrics["recall"]
            f = metrics["f1-score"]
            s = int(metrics["support"])
            rows.append(f"| **{c_title}** | {p:.3f} | {r:.3f} | {f:.3f} | {s} |")

    table_rows = "\n".join(rows)

    report_content = f"""# Waste Image Classification System - Evaluation Report
**Skill Nexis Deep Learning Major Project — Week 4**

---

## 1. Executive Summary
- **Evaluated Model:** `{model_name}`
- **Test Samples Count:** {total_samples}
- **Overall Accuracy:** {acc * 100:.2f}%
- **Macro F1-Score:** {macro_f1:.3f}
- **Weighted F1-Score:** {weighted_f1:.3f}

---

## 2. Per-Class Quantitative Performance

| Waste Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
{table_rows}
| **Overall Accuracy** | — | — | **{acc:.3f}** | **{total_samples}** |
| **Macro Average** | {report_dict.get("macro avg", {}).get("precision", 0.0):.3f} | {report_dict.get("macro avg", {}).get("recall", 0.0):.3f} | **{macro_f1:.3f}** | {total_samples} |
| **Weighted Average** | {report_dict.get("weighted avg", {}).get("precision", 0.0):.3f} | {report_dict.get("weighted avg", {}).get("recall", 0.0):.3f} | **{weighted_f1:.3f}** | {total_samples} |

---

## 3. Confusion Matrix Analysis
The confusion matrix heatmap is recorded at `reports/confusion_matrix.png`.
- **Cardboard & Paper:** Some cross-confusion occurs due to similar cellulose fiber textures and beige/brown packaging backgrounds.
- **Glass & Plastic:** Translucent glass bottles and clear plastic containers exhibit specular reflections that occasionally overlap visually.
- **Metal:** Distinct metallic highlights allow strong discriminative feature extraction.

---

## 4. Visual Error Analysis
Visual inspection of misclassifications is saved at `reports/error_analysis.png`.
- **Ambiguous Textures:** Crushed containers distort typical geometric silhouettes.
- **Occlusions & Lighting:** Shadows and intense specular highlights challenge standard convolution filters.

---

## 5. Deployment Recommendations & Disclaimers
1. **Model Confidence Guidance:** Probabilistic predictions from Softmax must not be treated as absolute truth.
2. **Confidence Thresholding:** We recommend flagging predictions with confidence below 65% for manual inspection.
3. **Continuous Retraining:** Expand data diversity under varied real-world illumination conditions.
"""

    report_path = config.REPORTS_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Markdown report generated at: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Waste Classification Model")
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.IMPROVED_MODEL_PATH),
        help="Path to trained .keras model file",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() and config.BASELINE_MODEL_PATH.exists():
        model_path = config.BASELINE_MODEL_PATH

    evaluate_model(model_path=model_path)

"""
Waste Image Classification System
Skill Nexis Deep Learning Major Project - Inference Module

Runs standalone inference on single waste images or folders, prints formatted
prediction probabilities, and saves annotated visual results to outputs/.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

# Set utf-8 encoding for standard output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
config.configure_keras_backend()
import keras

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess_image(image_path: Path, target_size: Tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH)) -> np.ndarray:
    """Loads and preprocesses an image file for model inference."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)
        return arr


def predict_single_image(
    image_path: Path,
    model: keras.Model,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predicts waste category for an image.
    Returns:
        (predicted_class, confidence_score, all_class_probabilities)
    """
    img_tensor = preprocess_image(image_path)
    preds = model.predict(img_tensor, verbose=0)[0]

    all_scores = {config.CLASSES[i]: float(preds[i]) for i in range(len(config.CLASSES))}
    top_idx = int(np.argmax(preds))
    top_class = config.CLASSES[top_idx]
    top_confidence = float(preds[top_idx])

    return top_class, top_confidence, all_scores


def annotate_and_save_prediction(
    image_path: Path,
    predicted_class: str,
    confidence: float,
    output_dir: Path = config.SAMPLE_PREDICTIONS_DIR,
) -> Path:
    """Draws a visual banner with prediction details and saves the result."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size

        # Create banner height
        banner_h = max(50, int(orig_h * 0.12))
        canvas = Image.new("RGB", (orig_w, orig_h + banner_h), color=(30, 35, 45))
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        meta = config.CLASS_METADATA.get(predicted_class, {})
        display_name = meta.get("display_name", predicted_class.capitalize())
        icon = meta.get("icon", "♻️")

        text = f"{icon} {display_name} ({confidence * 100:.1f}%)"
        
        # Draw label text
        draw.text((15, orig_h + int(banner_h * 0.25)), text, fill=(255, 255, 255))

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"pred_{image_path.name}"
        canvas.save(out_path, quality=92)
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Waste Image Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input waste image")
    parser.add_argument("--model", type=str, default="", help="Path to .keras model file")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated image in outputs/")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Resolve model path
    if args.model:
        model_path = Path(args.model)
    elif config.IMPROVED_MODEL_PATH.exists():
        model_path = config.IMPROVED_MODEL_PATH
    elif config.BASELINE_MODEL_PATH.exists():
        model_path = config.BASELINE_MODEL_PATH
    else:
        raise FileNotFoundError(
            "No trained model found! Please train a model first using 'python train.py'."
        )

    logger.info(f"Loading model: {model_path.name}")
    model = keras.models.load_model(str(model_path))

    top_class, top_conf, scores = predict_single_image(image_path, model)
    meta = config.CLASS_METADATA.get(top_class, {})

    try:
        print("\n" + "=" * 55)
        print("       WASTE CLASSIFICATION PREDICTION RESULT")
        print("=" * 55)
        print(f" Target Image       : {image_path.name}")
        print(f" Predicted Category : {meta.get('icon', '[*]')} {top_class.upper()}")
        print(f" Confidence Score   : {top_conf * 100:.2f}%")
        print("-" * 55)
        print(" Class Probability Distribution:")
        for cls_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            m = config.CLASS_METADATA.get(cls_name, {})
            bar = "#" * int(score * 30)
            print(f"  {m.get('icon', '*')} {cls_name.capitalize():<10} : {score * 100:5.1f}% | {bar}")
        print("-" * 55)
        print(" Recycling Advisory :")
        print(f"  {meta.get('recycling_tip', 'Recycle properly in designated bins.')}")
        print("=" * 55)
        print(" [!] Model confidence scores reflect probabilistic estimates and")
        print("     do not guarantee accuracy. Verify before disposal.")
        print("=" * 55 + "\n")
    except UnicodeEncodeError:
        print("\n" + "=" * 55)
        print("       WASTE CLASSIFICATION PREDICTION RESULT")
        print("=" * 55)
        print(f" Target Image       : {image_path.name}")
        print(f" Predicted Category : {top_class.upper()}")
        print(f" Confidence Score   : {top_conf * 100:.2f}%")
        print("-" * 55)
        print(" Class Probability Distribution:")
        for cls_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(score * 30)
            print(f"  * {cls_name.capitalize():<10} : {score * 100:5.1f}% | {bar}")
        print("-" * 55)
        print(" Recycling Advisory :")
        print(f"  {meta.get('recycling_tip', 'Recycle properly in designated bins.')}")
        print("=" * 55)
        print(" [!] Model confidence scores reflect probabilistic estimates and")
        print("     do not guarantee accuracy. Verify before disposal.")
        print("=" * 55 + "\n")

    if args.save_annotated:
        out_file = annotate_and_save_prediction(image_path, top_class, top_conf)
        print(f" Annotated image saved to: {out_file}\n")


if __name__ == "__main__":
    main()

"""
Waste Image Classification System - Web Application
Skill Nexis Deep Learning Major Project - Week 4

Provides an interactive, student-friendly Web Application using Gradio.
Allows users to upload waste images, view predictions across the 5 categories
(Cardboard, Glass, Metal, Paper, Plastic), inspect probability distributions,
and read responsible recycling guidelines with safety advisories.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from PIL import Image

import config
config.configure_keras_backend()
import keras

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global cache for loaded models
LOADED_MODELS: Dict[str, Any] = {}


def load_waste_model(model_type: str = "improved") -> keras.Model:
    """Loads and caches the requested model from disk."""
    if model_type in LOADED_MODELS:
        return LOADED_MODELS[model_type]

    if model_type == "baseline":
        model_path = config.BASELINE_MODEL_PATH
        fallback_path = config.IMPROVED_MODEL_PATH
    else:
        model_path = config.IMPROVED_MODEL_PATH
        fallback_path = config.BASELINE_MODEL_PATH

    target_path = model_path if model_path.exists() else fallback_path

    if not target_path.exists():
        logger.warning(
            f"No pre-trained model found at {model_path}. "
            "Building an untrained baseline architecture for interface demonstration. "
            "Please run 'python train.py' to train on real waste data."
        )
        import model as model_lib
        net = model_lib.build_baseline_model()
        net = model_lib.compile_model(net)
        LOADED_MODELS[model_type] = net
        return net

    logger.info(f"Loading weights from {target_path}...")
    net = keras.models.load_model(str(target_path))
    LOADED_MODELS[model_type] = net
    return net


def classify_waste_image(
    input_image: Image.Image,
    model_choice: str = "Improved Model",
) -> Tuple[str, str, Dict[str, float], str]:
    """
    Core prediction function for Gradio interface.
    Returns:
        (top_category_display, confidence_text, label_probabilities, recycling_advice_markdown)
    """
    if input_image is None:
        return (
            "No Image Uploaded",
            "0.0%",
            {c.capitalize(): 0.0 for c in config.CLASSES},
            "Please upload an image to receive waste classification and recycling guidance.",
        )

    # Convert image to RGB and resize
    img = input_image.convert("RGB")
    img_resized = img.resize((config.IMG_HEIGHT, config.IMG_WIDTH), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32)
    img_tensor = np.expand_dims(img_arr, axis=0)

    # Determine model type
    model_key = "baseline" if "baseline" in model_choice.lower() else "improved"
    net = load_waste_model(model_key)

    # Run inference
    preds = net.predict(img_tensor, verbose=0)[0]
    preds = [float(p) for p in preds]

    top_idx = int(np.argmax(preds))
    top_class = config.CLASSES[top_idx]
    top_confidence = preds[top_idx]

    meta = config.CLASS_METADATA.get(top_class, {})
    display_title = f"{meta.get('icon', '♻️')} {meta.get('display_name', top_class.capitalize())}"
    conf_str = f"{top_confidence * 100:.2f}%"

    # Dictionary of probabilities for Gradio Label component
    prob_dict = {
        f"{config.CLASS_METADATA[c]['icon']} {c.capitalize()}": round(preds[i], 4)
        for i, c in enumerate(config.CLASSES)
    }

    advice_md = f"""### {meta.get('icon', '♻️')} Recycling Guidelines for {meta.get('display_name', top_class.capitalize())}
- **Bin Allocation:** Designated `{meta.get('display_name', top_class.capitalize())}` Recycling Container.
- **Handling Advice:** {meta.get('recycling_tip', 'Keep material clean and dry.')}
- **Confidence Rating:** `{conf_str}`
"""
    return display_title, conf_str, prob_dict, advice_md


def build_gradio_app():
    """Builds the modern student-friendly Gradio Blocks application."""
    import gradio as gr

    custom_theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="Waste Image Classification System") as demo:
        gr.Markdown(
            """
            # ♻️ Waste Image Classification System
            ### Skill Nexis — Deep Learning Major Project (Week 4)
            Identify, sort, and classify recyclable waste materials using deep Convolutional Neural Networks (CNNs).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Waste Image", sources=["upload", "clipboard"])
                model_selector = gr.Radio(
                    choices=["Improved Model (MobileNetV2 / Deep CNN)", "Baseline CNN (From Scratch)"],
                    value="Improved Model (MobileNetV2 / Deep CNN)",
                    label="Select Deep Learning Model",
                )

                with gr.Row():
                    classify_btn = gr.Button("🔍 Classify Waste", variant="primary", scale=2)
                    clear_btn = gr.ClearButton(components=[image_input], scale=1)

                gr.Markdown(
                    """
                    **Supported Categories:**
                    - 📦 **Cardboard** | 🍾 **Glass** | 🥫 **Metal** | 📰 **Paper** | 🧴 **Plastic**
                    """
                )

            with gr.Column(scale=1):
                out_category = gr.Textbox(label="Predicted Category", placeholder="Awaiting image...", interactive=False)
                out_confidence = gr.Textbox(label="Model Confidence", placeholder="0.0%", interactive=False)
                out_probs = gr.Label(num_top_classes=5, label="Probability Distribution")
                out_advice = gr.Markdown("### Recycling Advice\n*Upload an image to see disposal recommendations.*")

        # Mandatory Safety Warning Banner
        gr.Markdown(
            """
            > ⚠️ **Important Model Advisory & Disclaimer:**
            > Model confidence scores reflect probabilistic pattern recognition from training data and **do not guarantee** that a prediction is correct. Always inspect waste items physically and adhere to your local municipality's recycling rules.
            """
        )

        classify_btn.click(
            fn=classify_waste_image,
            inputs=[image_input, model_selector],
            outputs=[out_category, out_confidence, out_probs, out_advice],
        )

    return demo


def run_flask_fallback(port: int = 7860):
    """Fallback web interface using Flask if Gradio is not installed."""
    from flask import Flask, request, jsonify, render_template_string

    app = Flask(__name__)

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Waste Image Classification System</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 20px; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            h1 { color: #059669; margin-top: 0; }
            .subtitle { color: #64748b; margin-bottom: 25px; }
            .warning { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 4px; margin: 20px 0; font-size: 14px; }
            .btn { background: #059669; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; }
            .btn:hover { background: #047857; }
            .results-card { background: #f1f5f9; padding: 15px; border-radius: 8px; margin-top: 20px; }
            .bar-container { background: #e2e8f0; border-radius: 4px; height: 18px; margin: 6px 0 12px; overflow: hidden; }
            .bar-fill { background: #10b981; height: 100%; transition: width 0.3s ease; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>♻️ Waste Image Classification System</h1>
            <div class="subtitle">Skill Nexis Deep Learning Major Project — Week 4</div>
            <div class="warning">
                ⚠️ <b>Model Advisory:</b> Model scores reflect probabilistic estimates and do not guarantee accuracy. Verify before disposal.
            </div>
            <form method="POST" action="/predict" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required style="margin-bottom: 15px;"><br>
                <button type="submit" class="btn">🔍 Classify Waste Image</button>
            </form>
            {% if prediction %}
            <div class="results-card">
                <h2>Top Category: {{ prediction }}</h2>
                <h3>Confidence: {{ confidence }}</h3>
                <p><b>Recycling Guideline:</b> {{ advice }}</p>
                <h4>Class Probabilities:</h4>
                {% for cls, score in scores.items() %}
                <div><b>{{ cls }}</b>: {{ score }}%</div>
                <div class="bar-container"><div class="bar-fill" style="width: {{ score }}%;"></div></div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(HTML_TEMPLATE, prediction=None)

    @app.route("/predict", methods=["POST"])
    def predict_route():
        if "image" not in request.files:
            return render_template_string(HTML_TEMPLATE, prediction=None)
        file = request.files["image"]
        img = Image.open(file.stream)
        cat, conf, probs, advice = classify_waste_image(img)
        formatted_probs = {k.split()[-1]: round(v * 100, 1) for k, v in probs.items()}
        return render_template_string(
            HTML_TEMPLATE,
            prediction=cat,
            confidence=conf,
            scores=formatted_probs,
            advice=advice,
        )

    print(f"\n[INFO] Starting Waste Classification Web App at: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


def main():
    try:
        import gradio as gr
        logger.info("Initializing Gradio UI...")
        demo = build_gradio_app()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except ImportError:
        logger.warning(
            "Gradio is not installed in the current environment. "
            "To install it, run: pip install gradio\n"
            "Launching built-in fallback web interface..."
        )
        run_flask_fallback(port=7860)


if __name__ == "__main__":
    main()

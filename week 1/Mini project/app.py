"""
AI Digit Recognizer - Flask Backend Application
================================================
Flask server serving the web interface and predicting digits from
canvas drawings using the trained PyTorch neural network.

Routes:
- GET  /         : Serves the web UI
- POST /predict  : Predicts digit and class probabilities from canvas image
- GET  /health   : Health check and model loading status
- GET  /api/samples : Returns sample digits data for instant testing
"""

import os
import sys
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

from model import DigitRecognizerNN
from preprocess import EmptyCanvasError, preprocess_canvas_image

# Initialize Flask application
app = Flask(__name__)

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "mnist_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_load_error = None


def load_digit_model():
    """
    Loads the trained PyTorch model into memory once at application startup.
    Sets model to evaluation mode.
    """
    global model, model_load_error
    if not os.path.exists(MODEL_PATH):
        model_load_error = (
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please train the model first by running: python train.py"
        )
        print(f"[WARNING] {model_load_error}")
        return False

    try:
        loaded_model = DigitRecognizerNN()
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        loaded_model.load_state_dict(state_dict)
        loaded_model.to(device)
        loaded_model.eval()  # Set to evaluation mode
        model = loaded_model
        model_load_error = None
        print(f"[INFO] Successfully loaded model from {MODEL_PATH} onto {device}.")
        return True
    except Exception as e:
        model_load_error = f"Failed to load model weights: {str(e)}"
        print(f"[ERROR] {model_load_error}")
        return False


# Attempt initial model load
load_digit_model()


@app.route("/")
def index():
    """Render the main user interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Reports server status, compute device, and whether the model is ready.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_path": MODEL_PATH,
        "error": model_load_error
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives base64 canvas image, preprocesses it, runs PyTorch inference,
    and returns predicted digit, confidence, and 10 class probabilities.
    """
    global model

    # 1. Verify model availability
    if model is None:
        # Re-attempt loading in case training finished after app started
        if not load_digit_model():
            return jsonify({
                "error": "Model not loaded.",
                "message": model_load_error or "Please run 'python train.py' to train and save the model."
            }), 503

    # 2. Parse and validate JSON request
    if not request.is_json:
        return jsonify({"error": "Bad Request", "message": "Expected JSON payload with 'image' key."}), 400

    data = request.get_json()
    image_data = data.get("image")

    if not image_data or not isinstance(image_data, str):
        return jsonify({"error": "Bad Request", "message": "Missing or invalid 'image' parameter."}), 400

    # 3. Preprocess canvas image
    try:
        tensor, _ = preprocess_canvas_image(image_data)
        tensor = tensor.to(device)
    except EmptyCanvasError as e:
        return jsonify({
            "error": "Empty Canvas",
            "message": str(e),
            "is_empty": True
        }), 400
    except ValueError as e:
        return jsonify({
            "error": "Invalid Image",
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Preprocessing Error",
            "message": f"Unexpected error while processing drawing: {str(e)}"
        }), 500

    # 4. Perform inference with PyTorch
    try:
        with torch.inference_mode():
            logits = model(tensor)  # Shape: (1, 10)
            probabilities_tensor = F.softmax(logits, dim=1)[0]  # Shape: (10,)

            # Extract predicted digit and confidence percentage
            probs_numpy = probabilities_tensor.cpu().numpy()
            predicted_digit = int(probs_numpy.argmax())
            confidence = float(probs_numpy[predicted_digit] * 100.0)

            # Build probabilities dictionary with keys "0" through "9" as percentages
            probabilities_dict = {
                str(i): round(float(probs_numpy[i] * 100.0), 2)
                for i in range(10)
            }

        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": probabilities_dict
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Inference Error",
            "message": f"Failed during neural network prediction: {str(e)}"
        }), 500


if __name__ == "__main__":
    # Ensure port 5000 (or custom via environment variable)
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting AI Digit Recognizer on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)

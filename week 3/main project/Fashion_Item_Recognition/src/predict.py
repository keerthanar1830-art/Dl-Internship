"""
Fashion Item Recognition - Inference and Prediction Module.

This module provides routines to load trained model checkpoints, preprocess input images
(from PIL Images or NumPy arrays), and generate prediction labels with confidence scores.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Ensure src directory is in sys.path for direct imports
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.model import FashionCNN, FashionFC, CLASS_NAMES
except ImportError:
    from model import FashionCNN, FashionFC, CLASS_NAMES

# Normalization constants (Fashion-MNIST training dataset statistics)
FASHION_MEAN = 0.2860
FASHION_STD = 0.3530


def load_model(checkpoint_path, device=None):
    """
    Load a trained Fashion-MNIST model checkpoint from disk.

    Automatically identifies the model architecture (FashionCNN or FashionFC)
    from the checkpoint's model_config metadata and maps tensors to the target device.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved PyTorch checkpoint file (.pth).
    device : torch.device, optional
        Target device for model execution. If None, automatically selects CUDA if
        available, otherwise falls back to CPU.

    Returns
    -------
    tuple of (torch.nn.Module, list of str, torch.device)
        - model: Loaded PyTorch neural network set in eval() mode.
        - class_names: List of category names stored in checkpoint or defaults.
        - device: Device the model was placed on.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract metadata from checkpoint
    model_config = checkpoint.get('model_config', {})
    architecture_name = model_config.get('architecture', '')
    model_name = model_config.get('model_name', '')
    num_classes = model_config.get('num_classes', 10)
    class_names = checkpoint.get('class_names', CLASS_NAMES)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Identify model architecture
    if 'CNN' in architecture_name or 'CNN' in model_name or 'FashionCNN' in architecture_name:
        model = FashionCNN(num_classes=num_classes)
    elif 'FC' in architecture_name or 'FC' in model_name or 'FashionFC' in architecture_name:
        model = FashionFC(num_classes=num_classes)
    else:
        # Fallback inspection of state_dict keys
        if any('conv' in key for key in state_dict.keys()):
            model = FashionCNN(num_classes=num_classes)
        else:
            model = FashionFC(num_classes=num_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded {model.__class__.__name__} from '{checkpoint_path}' onto {device}.")
    return model, class_names, device


def preprocess_image(image, mean=FASHION_MEAN, std=FASHION_STD):
    """
    Preprocess a PIL Image or NumPy array for model inference.

    Conversion steps:
    1. Converts image to PIL grayscale ('L' mode).
    2. Resizes spatial dimensions to 28x28 pixels using bilinear interpolation.
    3. Converts image to a float PyTorch tensor scaled to [0.0, 1.0].
    4. Normalizes pixel values using Fashion-MNIST dataset statistics (mean, std).
    5. Adds a batch dimension to return shape (1, 1, 28, 28).

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        Input image.
    mean : float, default=0.2860
        Dataset channel mean for normalization.
    std : float, default=0.3530
        Dataset channel standard deviation for normalization.

    Returns
    -------
    torch.Tensor
        Normalized image tensor with shape (1, 1, 28, 28) and dtype torch.float32.
    """
    # Convert numpy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        # Handle floating point images in [0, 1]
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Squeeze unnecessary single-dimensional entries
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)
        elif image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(axis=0)

        image = Image.fromarray(image)

    # Convert to grayscale ('L')
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to Fashion-MNIST standard dimensions (28x28)
    image = image.resize((28, 28), Image.Resampling.BILINEAR)

    # Transform to normalized tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(mean,), std=(std,))
    ])

    tensor = transform(image)  # Shape: (1, 28, 28)
    tensor = tensor.unsqueeze(0)  # Shape: (1, 1, 28, 28)

    return tensor


def predict(model, image_tensor, class_names, device):
    """
    Run inference on a single preprocessed image tensor.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to use for prediction.
    image_tensor : torch.Tensor
        Preprocessed image tensor with shape (1, 1, 28, 28).
    class_names : list of str
        List of class label names.
    device : torch.device
        Device to execute inference on.

    Returns
    -------
    dict
        Prediction result containing:
        - 'predicted_class': str, name of the predicted category.
        - 'confidence': float, probability of the predicted category (0.0 - 1.0).
        - 'probabilities': dict mapping each class name to its softmax probability.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    pred_idx_int = predicted_idx.item()
    predicted_class = class_names[pred_idx_int]
    confidence_float = float(confidence.item())

    probabilities_dict = {
        class_names[i]: float(probabilities[i].item())
        for i in range(len(class_names))
    }

    return {
        'predicted_class': predicted_class,
        'confidence': confidence_float,
        'probabilities': probabilities_dict
    }


if __name__ == '__main__':
    # Set random seed for reproducible demo run
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("============================================================")
    print("Fashion Item Recognition - Prediction Demo")
    print(f"Device: {device}")
    print("============================================================")

    # Candidate checkpoint paths
    cnn_checkpoint_path = os.path.join(SRC_DIR.parent, 'models', 'fashion_cnn.pth')
    fc_checkpoint_path = os.path.join(SRC_DIR.parent, 'models', 'fashion_fc.pth')

    if os.path.isfile(cnn_checkpoint_path):
        model, class_names, device = load_model(cnn_checkpoint_path, device=device)
    elif os.path.isfile(fc_checkpoint_path):
        model, class_names, device = load_model(fc_checkpoint_path, device=device)
    else:
        print("Note: No trained checkpoint found in 'models/'. Initializing FashionCNN demo model.")
        model = FashionCNN().to(device)
        model.eval()
        class_names = CLASS_NAMES

    # Fetch a sample image from FashionMNIST test set for demonstration
    from torchvision import datasets
    data_dir = os.path.join(SRC_DIR.parent, 'data')
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True)

    sample_index = 0
    sample_image, sample_label = test_dataset[sample_index]
    actual_class = class_names[sample_label]

    # Preprocess image
    image_tensor = preprocess_image(sample_image)

    # Run prediction
    result = predict(
        model=model,
        image_tensor=image_tensor,
        class_names=class_names,
        device=device
    )

    print(f"\n--- Inference Result for Sample #{sample_index} ---")
    print(f"Actual Class    : {actual_class}")
    print(f"Predicted Class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence'] * 100:.2f}%")

    # Display Top 3 Probabilities
    sorted_probs = sorted(result['probabilities'].items(), key=lambda item: item[1], reverse=True)
    print("\nTop 3 Probabilities:")
    for rank, (c_name, prob) in enumerate(sorted_probs[:3], start=1):
        print(f"  {rank}. {c_name:12s}: {prob * 100:.2f}%")
    print("============================================================")

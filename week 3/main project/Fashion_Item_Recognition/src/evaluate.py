"""
Fashion Item Recognition - Evaluation and Visualization Module.

This module provides comprehensive evaluation metrics and visualizations for Fashion-MNIST models:
- evaluate_model: Computes overall test loss, accuracy, and aggregates predictions.
- get_classification_report: Generates precision, recall, and F1-score breakdown per class.
- get_confusion_matrix: Computes confusion matrix of predictions vs. ground truth.
- plot_confusion_matrix: Renders and saves a matplotlib heatmap of the confusion matrix.
- find_misclassified: Extracts misclassified images, predictions, and model confidence scores.
- plot_misclassified: Visualizes top misclassified examples.
- plot_prediction_gallery: Visualizes test predictions colored green (correct) or red (incorrect).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_loader, criterion, device):
    """
    Perform full evaluation on the test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader providing test batches of (images, labels).
    criterion : torch.nn.Module
        Loss function (e.g. nn.CrossEntropyLoss).
    device : torch.device
        Device used for computation (CPU or CUDA).

    Returns
    -------
    tuple of (float, float, np.ndarray, np.ndarray)
        - test_loss: Average test loss across all samples.
        - test_accuracy: Overall test accuracy percentage (0.0 - 100.0).
        - all_preds: 1D numpy array of predicted class indices.
        - all_labels: 1D numpy array of true ground truth class indices.
    """
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    all_preds_list = []
    all_labels_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, dim=1)
            running_correct += torch.sum(preds == labels.data).item()
            total_samples += batch_size

            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    test_loss = running_loss / total_samples
    test_accuracy = (running_correct / total_samples) * 100.0
    all_preds = np.array(all_preds_list, dtype=np.int64)
    all_labels = np.array(all_labels_list, dtype=np.int64)

    return test_loss, test_accuracy, all_preds, all_labels


def get_classification_report(all_labels, all_preds, class_names):
    """
    Generate a text classification report containing Precision, Recall, and F1-Score.

    Parameters
    ----------
    all_labels : np.ndarray or list
        Ground truth class labels.
    all_preds : np.ndarray or list
        Predicted class labels.
    class_names : list of str
        Human-readable category names for each class index.

    Returns
    -------
    str
        Formatted classification report.
    """
    return classification_report(
        y_true=all_labels,
        y_pred=all_preds,
        target_names=class_names,
        digits=4
    )


def get_confusion_matrix(all_labels, all_preds):
    """
    Compute confusion matrix of predicted vs. true labels.

    Parameters
    ----------
    all_labels : np.ndarray or list
        Ground truth class labels.
    all_preds : np.ndarray or list
        Predicted class labels.

    Returns
    -------
    np.ndarray
        2D numpy array representing the confusion matrix.
    """
    return confusion_matrix(y_true=all_labels, y_pred=all_preds)


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix heatmap using pure matplotlib (without seaborn).

    Parameters
    ----------
    cm : np.ndarray
        2D confusion matrix array.
    class_names : list of str
        Names of the target classes.
    save_path : str, optional
        File path to save the generated figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    num_classes = len(class_names)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            text_color = "white" if count > threshold else "black"
            ax.text(
                j, i, format(count, "d"),
                ha="center", va="center",
                color=text_color, fontsize=10
            )

    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix plot saved to '{save_path}'")

    return fig


def find_misclassified(model, test_loader, device, num_examples=5):
    """
    Identify misclassified test samples along with predicted and true labels and confidence.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    test_loader : torch.utils.data.DataLoader
        Test data loader.
    device : torch.device
        Computation device (CPU or CUDA).
    num_examples : int, default=5
        Number of misclassified examples to collect.

    Returns
    -------
    list of dict
        Each dictionary contains:
        - 'image': 2D numpy array of the image (shape 28x28)
        - 'true_label': int, ground truth label index
        - 'pred_label': int, incorrect predicted label index
        - 'confidence': float, model confidence (max softmax probability)
        - 'probabilities': 1D numpy array of probabilities across all 10 classes
    """
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            for i in range(images.size(0)):
                pred = predictions[i].item()
                true = labels[i].item()

                if pred != true:
                    # Squeeze channel dimension to get (28, 28) image array
                    img_np = images[i].cpu().squeeze().numpy()
                    prob_np = probabilities[i].cpu().numpy()

                    misclassified.append({
                        'image': img_np,
                        'true_label': true,
                        'pred_label': pred,
                        'confidence': float(confidences[i].item()),
                        'probabilities': prob_np
                    })

                    if len(misclassified) >= num_examples:
                        return misclassified

    return misclassified


def plot_misclassified(misclassified, class_names, save_path=None):
    """
    Plot misclassified examples with true label, predicted label, and prediction confidence.

    Parameters
    ----------
    misclassified : list of dict
        List of misclassified item dictionaries as returned by find_misclassified.
    class_names : list of str
        List of class label names.
    save_path : str, optional
        File path to save the generated plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    n = len(misclassified)
    if n == 0:
        print("No misclassified examples found.")
        return None

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    if n == 1:
        axes = [axes]

    for idx, item in enumerate(misclassified):
        ax = axes[idx]
        ax.imshow(item['image'], cmap='gray')
        ax.axis('off')

        true_name = class_names[item['true_label']]
        pred_name = class_names[item['pred_label']]
        conf_pct = item['confidence'] * 100.0

        ax.set_title(
            f"True: {true_name}\nPred: {pred_name}\nConf: {conf_pct:.1f}%",
            fontsize=10,
            color='darkred'
        )

    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Misclassified examples plot saved to '{save_path}'")

    return fig


def plot_prediction_gallery(model, test_loader, device, class_names, num_images=15, save_path=None):
    """
    Create a prediction gallery displaying sample test images with predictions.
    Correct predictions are labeled in green; incorrect predictions are labeled in red.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    test_loader : torch.utils.data.DataLoader
        Test data loader.
    device : torch.device
        Computation device (CPU or CUDA).
    class_names : list of str
        List of class label names.
    num_images : int, default=15
        Number of images to display in the gallery.
    save_path : str, optional
        File path to save the generated figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    model.eval()

    collected_images = []
    collected_true = []
    collected_pred = []
    collected_conf = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            for i in range(images.size(0)):
                collected_images.append(images[i].cpu().squeeze().numpy())
                collected_true.append(labels[i].item())
                collected_pred.append(predictions[i].item())
                collected_conf.append(float(confidences[i].item()))

                if len(collected_images) >= num_images:
                    break
            if len(collected_images) >= num_images:
                break

    cols = 5
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows))
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < len(collected_images):
            img = collected_images[idx]
            true_idx = collected_true[idx]
            pred_idx = collected_pred[idx]
            conf_val = collected_conf[idx] * 100.0

            ax.imshow(img, cmap='gray')
            ax.axis('off')

            is_correct = (true_idx == pred_idx)
            color = 'darkgreen' if is_correct else 'red'
            status = "Correct" if is_correct else "Wrong"

            ax.set_title(
                f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]} ({conf_val:.1f}%)\n[{status}]",
                fontsize=9,
                color=color
            )
        else:
            ax.axis('off')

    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Prediction gallery plot saved to '{save_path}'")

    return fig

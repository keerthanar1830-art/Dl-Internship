# Waste Image Classification – Final Model Evaluation
**Skill Nexis – Deep Learning (Week 4: Assignment 4)**

---

## 📌 Project Overview
This repository contains the complete implementation and evaluation framework for **Skill Nexis Week 4 Assignment 4: Final Model Evaluation**.

The goal of this project is to develop, train, compare, and critically evaluate Convolutional Neural Network (CNN) architectures on a 5-class waste classification problem:
1. **Cardboard**
2. **Glass**
3. **Metal**
4. **Paper**
5. **Plastic**

Proper automated sorting of recyclables is a foundational environmental application of computer vision. This project adheres to rigorous machine learning standards:
- **Data Integrity & Exact Duplicate Removal:** Uses SHA-256 cryptographic hashing to identify exact duplicate images and prevent data leakage across splits.
- **Label Conflict Auditing:** Checks if identical image files exist under conflicting category labels and halts/warns without silent data loss.
- **Stratified Partitioning:** Splits data into 70% Training, 15% Validation, and 15% Testing while strictly preserving class balance.
- **High-Performance Input Pipelines:** Uses `tf.data.Dataset` with RGB standardization, 128x128 resizing, [0, 1] normalization, batching, caching, and prefetching.
- **Architectural Comparison:** Compares a clean **Baseline CNN** against an **Improved CNN** featuring data augmentation, batch normalization, global average pooling, and learning rate scheduling.
- **Comprehensive Evaluation:** Generates precision, recall, F1-scores, confusion matrix heatmaps, test loss/accuracy comparisons, misclassified image failure analysis, and an automated Markdown evaluation report.

---

## 📂 Project Structure

```text
waste_image_evaluation/
│
├── waste_image_evaluation.ipynb    # Main end-to-end executable Jupyter Notebook
│
├── dataset/                        # Image dataset organized by class
│   ├── cardboard/                  # Cardboard waste images
│   ├── glass/                      # Glass bottle and jar images
│   ├── metal/                      # Metal can and tin images
│   ├── paper/                      # Paper sheets and packaging images
│   └── plastic/                    # Plastic bottle and container images
│
├── outputs/                        # Generated evaluation artifacts
│   ├── graphs/                     # Training & validation loss/accuracy curves
│   │   ├── baseline_training_curves.png
│   │   ├── improved_training_curves.png
│   │   └── combined_accuracy_comparison.png
│   ├── confusion_matrix.png        # Test set confusion matrix heatmap
│   ├── model_comparison.csv        # Summary table of baseline vs improved metrics
│   ├── classification_report.csv   # Per-class precision, recall, and F1-score
│   ├── misclassified_images.csv    # Log of test misclassifications with confidences
│   ├── class_distribution.png      # Initial dataset class distribution chart
│   └── evaluation_report.md        # Formatted, auto-populated final evaluation report
│
├── models/                         # Serialized trained model weights
│   ├── baseline_cnn.keras          # Saved baseline model
│   └── improved_cnn.keras          # Saved improved model
│
├── requirements.txt                # Required Python packages
└── README.md                       # Comprehensive project documentation
```

---

## 🗄️ Dataset Setup & Acquisition

### Target Dataset: TrashNet
The project is designed to run seamlessly with the widely-used **TrashNet** dataset:
- **Original Source:** Gary Thung & Mindy Yang, *TrashNet* (Stanford University CS229 Project)
- **GitHub Repository:** [garythung/trashnet](https://github.com/garythung/trashnet)
- **Kaggle Mirror:** [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- **License / Usage:** Open for academic and educational research.

> **Note on 5 Classes:** TrashNet natively contains 6 folders (`cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`). For this 5-class assignment, only the 5 standard recyclable classes are used. The `trash` folder is simply excluded.

### Option A: Automatic Download via Colab or Shell
You can clone or download the dataset directly:
```bash
# Clone TrashNet data repository
git clone https://github.com/garythung/trashnet.git

# Copy the 5 required classes into dataset/
cp -r trashnet/data/dataset-resized/cardboard dataset/
cp -r trashnet/data/dataset-resized/glass dataset/
cp -r trashnet/data/dataset-resized/metal dataset/
cp -r trashnet/data/dataset-resized/paper dataset/
cp -r trashnet/data/dataset-resized/plastic dataset/
```

### Option B: Using Custom or Manually Downloaded Images
Place your images into the corresponding category subdirectories:
```text
dataset/cardboard/  (e.g., cardboard1.jpg, cardboard2.png, ...)
dataset/glass/      (e.g., glass1.jpg, ...)
dataset/metal/      (e.g., metal1.jpg, ...)
dataset/paper/      (e.g., paper1.jpg, ...)
dataset/plastic/    (e.g., plastic1.jpg, ...)
```
Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

---

## 🚀 Execution Guide

### Option 1: Running in Google Colab (Recommended)
Google Colab provides a free Linux GPU/CPU environment with TensorFlow and CUDA pre-installed.

1. Open [Google Colab](https://colab.research.google.com).
2. Click **File -> Upload notebook** and select `waste_image_evaluation.ipynb`.
3. In Colab, go to **Runtime -> Change runtime type** and select **T4 GPU** (or standard CPU).
4. Run Section 2 in the notebook, which provides an interactive upload cell for a dataset `.zip` file or a 1-click script to download TrashNet directly.
5. Execute the cells sequentially from top to bottom.

### Option 2: Running Locally via Jupyter Notebook

#### Prerequisites
- Python 3.10 or 3.11
- Microsoft Visual C++ 2015-2022 Redistributable (on Windows)

#### Step 1: Create and Activate a Virtual Environment
```bash
# Windows
python -m venv dl_env
.\dl_env\Scripts\activate

# macOS / Linux
python3 -m venv dl_env
source dl_env/bin/activate
```

#### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: Launch Jupyter Lab / Notebook
```bash
jupyter lab
# or
jupyter notebook
```
Navigate to `waste_image_evaluation.ipynb` and run all cells.

---

## 🧠 Model Architectures

| Component | Model 1: Baseline CNN | Model 2: Improved CNN |
| :--- | :--- | :--- |
| **Input Shape** | 128 × 128 × 3 | 128 × 128 × 3 |
| **Data Augmentation** | None | Random Flip, Rotation (±10%), Zoom (±10%), Translation (±10%) |
| **Feature Extraction** | 2 × Conv2D (16, 32 filters, 3×3, ReLU) + MaxPooling2D | 3 × Conv Blocks (32, 64, 128 filters, 3×3) + BatchNorm + ReLU + MaxPool |
| **Pooling / Aggregation**| `Flatten()` layer | `GlobalAveragePooling2D()` layer |
| **Dense Regularization**| Dense(64, ReLU) + Dropout(0.3) | Dense(64, ReLU) + Dropout(0.4) |
| **Output Layer** | Dense(5, Softmax) | Dense(5, Softmax) |
| **Optimizer** | Adam (lr=1e-3) | Adam (lr=1e-3) with `ReduceLROnPlateau` |
| **Loss Function** | Sparse Categorical Cross-Entropy | Sparse Categorical Cross-Entropy |
| **Max Epochs** | 15 (EarlyStopping patience=4) | 20 (EarlyStopping patience=5) |

### Why Model 2 Generalizes Better
1. **On-the-Fly Data Augmentation:** Artificially expands training variety by simulating scale, viewpoint, and rotational shifts, discouraging overfitting on specific object poses.
2. **Batch Normalization:** Stabilizes feature distributions across mini-batches, smoothing the loss surface and accelerating convergence.
3. **Global Average Pooling:** Replaces large fully connected flatten layers, drastically cutting trainable parameter count and reducing spatial overfitting.
4. **Learning Rate Decay:** Halves the learning rate when validation loss plateaus, allowing the optimizer to settle into narrower minima without oscillation.

---

## 📊 Evaluation & Metrics Explained

### Confusion Matrix & Metrics
- **True Positive (TP):** Item correctly identified as its category (e.g., Plastic bottle identified as Plastic).
- **False Positive (FP):** Non-target item incorrectly identified as target (e.g., Glass identified as Plastic). Contaminates the recyclable plastic stream.
- **False Negative (FN):** Target item missed and mislabeled as another material (e.g., Plastic identified as Glass). Recyclable plastic is sent to wrong facility or landfill.
- **Precision = $\frac{TP}{TP + FP}$:** Purity of the sorted output. High precision prevents facility contamination.
- **Recall = $\frac{TP}{TP + FN}$:** Recovery rate. High recall ensures maximum recyclable items are captured.
- **F1-Score = $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$:** Harmonic mean balancing purity and recovery.

---

## 🔮 Predicting New Images

The notebook defines an end-to-end inference function `predict_waste_image(image_path)`:
```python
result = predict_waste_image(
    image_path="test_sample.jpg",
    model=improved_model,
    class_names=['cardboard', 'glass', 'metal', 'paper', 'plastic']
)
print(f"Predicted: {result['class']} ({result['confidence']:.2f}%)")
```
It replicates the exact resizing (128x128), RGB conversion, and [0, 1] normalization applied during training.

---

## ⚠️ Known Limitations & Failure Analysis Insights
1. **Transparent Materials (Glass vs. Plastic):** Clear plastic water bottles and clear glass jars share high visual transparency and specular highlights, making them prone to mutual confusion.
2. **Deformation and Crushing:** Crushed aluminum soda cans can exhibit flat geometry similar to crumpled cardboard or paper.
3. **Background Occlusions:** Images with busy backgrounds or hands holding the object can cause false feature activations if not isolated.
4. **Out-of-Distribution Inputs:** The model only knows its 5 trained classes. It will produce a probability distribution over the 5 classes even for unrelated images (e.g., an apple or a shoe). Confidence scores must be treated with appropriate thresholding.

---

## 🛠️ Troubleshooting Windows TensorFlow Issues
If running on Windows and encountering:
`ImportError: DLL load failed while importing _pywrap_tensorflow_internal: A dynamic link library (DLL) initialization routine failed.`
- Official native TensorFlow support on Windows ended after version 2.10. Newer releases (2.16+) require either:
  1. WSL2 (Windows Subsystem for Linux),
  2. A Docker container, or
  3. Google Colab (recommended for zero setup).
- Ensure the latest **Microsoft Visual C++ 2015-2022 Redistributable** is installed if attempting local native Windows execution.

# ♻️ Waste Image Classification System
**Skill Nexis — Deep Learning Major Project (Week 4)**

A complete, production-grade Deep Learning project that classifies recyclable and solid waste images into five distinct categories using **Convolutional Neural Networks (CNNs)** and an interactive **Gradio** web application.

---

## 📌 Project Overview

Rapid urbanization and material consumption necessitate efficient municipal waste segregation. This project implements an end-to-end Computer Vision pipeline designed for automated sorting across five critical waste categories:

1. 📦 **Cardboard** (corrugated boxes, packaging cartons)
2. 🍾 **Glass** (clear, brown, and green beverage bottles and jars)
3. 🥫 **Metal** (aluminum soda cans, steel/tin food containers, foil)
4. 📰 **Paper** (office paper, newspaper, magazines, envelopes)
5. 🧴 **Plastic** (PET bottles, HDPE containers, food tubs)

The system compares a **Custom Baseline CNN** developed from scratch against an **Improved Model** featuring integrated spatial data augmentation, residual feature representation, and transfer learning capabilities.

---

## 🏗️ Folder Structure

```
Waste_Image_Classification/
│
├── app.py                      # Interactive Gradio & Fallback Web Application
├── train.py                    # Model training pipeline (Baseline & Improved)
├── evaluate.py                 # Evaluation, confusion matrix, error analysis
├── predict.py                  # Standalone CLI prediction & image annotator
├── data_loader.py              # Dataset splitting, PyDataset loader, & EDA
├── model.py                    # Baseline CNN & Improved Deep Architectures
├── config.py                   # Centralized paths, classes, & hyperparameters
├── requirements.txt            # Pinned dependency manifest
├── README.md                   # Full documentation & execution manual
├── .gitignore                  # Git tracking exclusion rules
│
├── data/
│   ├── raw/                    # Raw dataset directory (Cardboard, Glass, Metal, Paper, Plastic)
│   └── processed/              # Stratified train (70%), val (15%), and test (15%) splits
│
├── models/
│   ├── baseline_model.keras    # Trained custom Baseline CNN (.keras format)
│   └── improved_model.keras    # Trained Improved Model (.keras format)
│
├── notebooks/
│   └── waste_classification.ipynb  # End-to-end interactive student notebook
│
├── reports/
│   ├── evaluation_report.md        # Detailed quantitative assessment report
│   ├── class_distribution.png      # Class counts visualization across splits
│   ├── baseline_training_curves.png# Loss and accuracy curves for Baseline CNN
│   ├── improved_training_curves.png# Loss and accuracy curves for Improved Model
│   ├── confusion_matrix.png        # Multiclass confusion matrix heatmap
│   ├── classification_report.csv   # Precision, Recall, F1, and Support per class
│   └── error_analysis.png          # Visual breakdown of misclassified test samples
│
└── outputs/
    └── sample_predictions/         # Annotated visual prediction exports
```

---

## 🛠️ Technology Stack

- **Programming Language:** Python 3.8+ / 3.10
- **Deep Learning Framework:** Keras 3 / TensorFlow 2.x (with PyTorch backend support)
- **Computer Vision & Image Processing:** Pillow (PIL), NumPy
- **Data Analytics & Metrics:** Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **User Interface:** Gradio (with fallback Flask web server)
- **Hardware Requirement:** Standard CPU (GPU optional)

---

## ⚡ Quick Start & Installation

### 1. Clone or Navigate to the Project Directory
```bash
cd Waste_Image_Classification
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Option A: Using the Real-World TrashNet Dataset (Recommended)
1. Download the public **TrashNet** dataset:
   - **GitHub:** [garythung/trashnet](https://github.com/garythung/trashnet)
   - **Kaggle:** [TrashNet Dataset on Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/trashnet)
2. Extract the dataset folder.
3. Copy the five required category folders (`cardboard`, `glass`, `metal`, `paper`, `plastic`) into `data/raw/`:
   ```
   data/raw/
   ├── cardboard/
   ├── glass/
   ├── metal/
   ├── paper/
   └── plastic/
   ```
   *(Note: The 6th class `trash` present in raw TrashNet is automatically excluded to focus exclusively on the 5 recyclables).*
4. Run the automated dataset preparation and splitting script:
   ```bash
   python data_loader.py --split --report
   ```
   This stratifies the dataset into 70% training, 15% validation, and 15% testing partitions inside `data/processed/` and exports `reports/class_distribution.png`.

### Option B: Quick Dry-Run & Verification (Offline Sample Mode)
If you do not have immediate access to the full dataset, you can generate a synthetic test dataset to verify the pipeline:
```bash
python data_loader.py --create-sample --split --report
```

---

## 🚀 Running the Project

### 1. Train the Models
You can train either architecture individually or both sequentially:

- **Train Baseline CNN (from scratch):**
  ```bash
  python train.py --model baseline --epochs 15 --batch_size 32
  ```

- **Train Improved Model (Transfer Learning / Deep Separable CNN):**
  ```bash
  python train.py --model improved --epochs 20 --batch_size 32
  ```

- **Train Both Models:**
  ```bash
  python train.py --model both --epochs 15
  ```

Training outputs will automatically save:
- Model checkpoints: `models/baseline_model.keras` and `models/improved_model.keras`
- Epoch training curves: `reports/baseline_training_curves.png` and `reports/improved_training_curves.png`

---

### 2. Comprehensive Model Evaluation
Run the evaluation script to assess the trained model against the unseen test dataset:
```bash
python evaluate.py --model models/improved_model.keras
```
This generates:
- `reports/confusion_matrix.png`
- `reports/classification_report.csv`
- `reports/error_analysis.png`
- `reports/evaluation_report.md`

---

### 3. CLI Prediction on a Single Image
Run inference directly from the command line on any test image:
```bash
python predict.py --image data/processed/test/cardboard/sample_cardboard_001.jpg --save-annotated
```
Example Output:
```
=======================================================
       WASTE CLASSIFICATION PREDICTION RESULT
=======================================================
 Target Image       : sample_cardboard_001.jpg
 Predicted Category : 📦 CARDBOARD
 Confidence Score   : 92.40%
-------------------------------------------------------
 Class Probability Distribution:
  📦 Cardboard  :  92.4% | ############################
  📰 Paper      :   4.1% | #
  🥫 Metal      :   1.8% | 
  🧴 Plastic    :   1.1% | 
  🍾 Glass      :   0.6% | 
-------------------------------------------------------
 Recycling Advisory :
  Flatten boxes and remove plastic packing tape before placing in the recycling bin.
=======================================================
 [!] Model confidence scores reflect probabilistic estimates and
     do not guarantee accuracy. Verify before disposal.
=======================================================
```
Annotated prediction results are saved in `outputs/sample_predictions/`.

---

### 4. Launch the Web Application
Start the interactive student-friendly web application:
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:7860
```

#### Application Features:
- 🖼️ **Image Uploader:** Drag-and-drop or select waste images.
- 🔄 **Model Selector:** Compare predictions between the Baseline CNN and Improved Model.
- 🎯 **Confidence Metric:** Displays exact percentage confidence.
- 📊 **Probability Distribution:** Visual breakdown across all 5 classes.
- 💡 **Actionable Recycling Advice:** Material-specific disposal guidelines.
- ⚠️ **Mandatory Model Advisory:** Clear warning that model scores are probabilistic estimates and do not guarantee ground-truth accuracy.

---

## 🔬 Model Architectures

### 1. Baseline CNN (Custom From Scratch)
- **Input:** $224 \times 224 \times 3$ RGB image
- **Preprocessing:** Rescaling $[0, 255] \to [0.0, 1.0]$
- **Feature Extractor:** 4 sequential blocks:
  - Block 1: Conv2D(32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
  - Block 2: Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
  - Block 3: Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
  - Block 4: Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
- **Head:** GlobalAveragePooling2D + Dense(128, L2 regularized) + Dropout(0.4) + Softmax(5 classes)

### 2. Improved Model
- **Data Augmentation:** Random horizontal/vertical flip, random rotation ($15^\circ$), random zoom ($15\%$)
- **Feature Backbone:** MobileNetV2 pre-trained on ImageNet / Enhanced Deep Separable Residual blocks
- **Regularization:** Dual Batch Normalization and Dropout ($0.4$)
- **Classifier Head:** GlobalAveragePooling2D + Dense(256) + Softmax(5 classes)

---

## 📈 Evaluation & Key Insights

1. **Fiber Texture Overlap (Paper vs. Cardboard):**
   Paper and cardboard share biological cellulose fibers. The improved model leverages multi-scale spatial filters to identify cardboard fluting and corrugation.
2. **Specular Reflections (Glass vs. Plastic):**
   Clear plastic and glass bottles frequently exhibit specular glare. Data augmentation with contrast and translation enhances distinction based on thickness and closure design.
3. **Thresholding for Deployment:**
   Images resulting in top confidence below $65\%$ should be directed to a manual sorting stream or flagged for human review.

---

## 📜 Skill Nexis Submission Checklist

- [x] Complete project folder structure matching project specifications.
- [x] Five target categories implemented: Cardboard, Glass, Metal, Paper, Plastic.
- [x] Baseline CNN constructed and trained from scratch.
- [x] Improved Model implemented with augmentation and transfer learning.
- [x] Model checkpoints saved in `.keras` format inside `models/`.
- [x] Evaluation metrics, confusion matrix, and error analysis generated inside `reports/`.
- [x] Interactive Gradio application with mandatory model advisory warning in `app.py`.
- [x] Standalone inference CLI with probability distribution in `predict.py`.
- [x] Detailed Jupyter Notebook for presentation in `notebooks/waste_classification.ipynb`.
- [x] CPU-compatible and cross-platform verified.

---

## ⚖️ License & Attribution
Developed for the **Skill Nexis Deep Learning Four-Week Learning Program**.  
Waste image benchmarks inspired by the Stanford TrashNet dataset (Gary Thung and Mindy Yang).

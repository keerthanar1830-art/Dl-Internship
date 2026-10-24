# 👗 Fashion Item Recognition

## SKILL NEXIS Week 3 — Deep Learning: Computer Vision & Convolutional Neural Networks

---

## 📋 Project Overview

This project builds a deep learning model that recognizes clothing and fashion items from images using **Convolutional Neural Networks (CNNs)** and the **Fashion-MNIST** dataset. It classifies images into 10 categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

The project also includes a **Fully Connected (FC) baseline** model for comparison, demonstrating why CNNs excel at image classification tasks.

## 🎯 Objectives

- Load and explore the Fashion-MNIST dataset
- Build and train a CNN model for image classification
- Build and train a Fully Connected baseline model for comparison
- Visualize training/validation curves to monitor learning
- Evaluate models with accuracy, precision, recall, F1-score, and confusion matrix
- Analyze misclassified examples to understand model limitations
- Save and reload trained models for reuse
- Create an optional Streamlit web interface for interactive predictions

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Dataset loading and transforms |
| NumPy | Numerical operations |
| Matplotlib | Plotting and visualization |
| Pandas | Data analysis and tables |
| Scikit-learn | Evaluation metrics |
| Streamlit | Web interface (optional) |
| Jupyter Notebook | Interactive development |

## 📊 Dataset Information

**Fashion-MNIST** is a dataset of Zalando's article images:
- **Training set:** 60,000 images (split into train + validation)
- **Test set:** 10,000 images
- **Image size:** 28 × 28 pixels, grayscale
- **Classes:** 10 clothing categories

The dataset is downloaded automatically via `torchvision.datasets.FashionMNIST`.

## 📁 Project Structure

```
Fashion_Item_Recognition/
│
├── notebooks/
│   └── fashion_item_recognition.ipynb   # Main notebook
│
├── src/
│   ├── model.py        # CNN and FC model definitions
│   ├── train.py        # Training functions
│   ├── evaluate.py     # Evaluation and visualization functions
│   └── predict.py      # Prediction and model loading functions
│
├── models/
│   ├── fashion_cnn.pth # Trained CNN checkpoint
│   └── fashion_fc.pth  # Trained FC checkpoint
│
├── outputs/
│   ├── sample_images.png
│   ├── class_distribution.png
│   ├── training_validation_loss.png
│   ├── training_validation_accuracy.png
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── misclassified_examples.png
│   └── prediction_gallery.png
│
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or later installed on your system
- VS Code with the Python and Jupyter extensions

### Step 1: Open the Project in VS Code

Open the `Fashion_Item_Recognition` folder in VS Code.

### Step 2: Create a Virtual Environment

Open a PowerShell terminal in VS Code and run:

```powershell
py -m venv .venv
```

### Step 3: Install Dependencies

```powershell
..\.venv\Scripts\python.exe -m pip install --upgrade pip
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Note:** If `..\.venv\Scripts\python.exe` does not work, try:
> ```powershell
> & ".venv\Scripts\python.exe" -m pip install --upgrade pip
> & ".venv\Scripts\python.exe" -m pip install -r requirements.txt
> ```

### Step 4: Select the Python Interpreter in VS Code

1. Press `Ctrl+Shift+P` to open the Command Palette
2. Type "Python: Select Interpreter" and select it
3. Choose the `.venv` interpreter (it should appear as `.venv\Scripts\python.exe`)

### Step 5: Select the Notebook Kernel

1. Open `notebooks/fashion_item_recognition.ipynb` in VS Code
2. Click on the kernel selector in the top-right corner of the notebook
3. Select `.venv (Python 3.x.x)` as the kernel

## 📓 Running the Notebook

1. Open `notebooks/fashion_item_recognition.ipynb` in VS Code
2. Make sure the `.venv` kernel is selected
3. Click **"Run All"** or run cells sequentially from top to bottom
4. The notebook will:
   - Download Fashion-MNIST automatically
   - Train both CNN and FC models
   - Generate all visualizations in `outputs/`
   - Save trained models in `models/`

## 🏋️ Training the Models (via Script)

Alternatively, you can train models from the command line:

```powershell
..\.venv\Scripts\python.exe -m src.train
```

Or use the prediction script:

```powershell
..\.venv\Scripts\python.exe -m src.predict
```

## 🌐 Launching the Streamlit App

After training the model (by running the notebook), launch the web app:

```powershell
..\.venv\Scripts\python.exe -m streamlit run app.py
```

This opens a browser where you can upload fashion item images for classification.

## 📈 Expected Output Files

After running the notebook, these files will be generated:

| File | Description |
|---|---|
| `outputs/sample_images.png` | Grid of sample Fashion-MNIST images |
| `outputs/class_distribution.png` | Bar chart of class distribution |
| `outputs/training_validation_loss.png` | Loss curves during training |
| `outputs/training_validation_accuracy.png` | Accuracy curves during training |
| `outputs/model_comparison.png` | CNN vs FC comparison chart |
| `outputs/model_comparison.csv` | Model comparison data table |
| `outputs/confusion_matrix.png` | Classification confusion matrix |
| `outputs/misclassified_examples.png` | Examples the model got wrong |
| `outputs/prediction_gallery.png` | Gallery of predictions with confidence |
| `models/fashion_cnn.pth` | Trained CNN model checkpoint |
| `models/fashion_fc.pth` | Trained FC model checkpoint |

## 🔍 Model Comparison

The project trains two models and compares them:

- **CNN (Convolutional Neural Network):** Uses convolutional layers that detect spatial patterns like edges, textures, and shapes. CNNs preserve the 2D structure of images, making them ideal for image classification.

- **FC Baseline (Fully Connected):** Flattens each image into a 1D vector and uses dense layers. This loses spatial information, generally resulting in lower accuracy.

The CNN typically achieves higher accuracy because it can learn hierarchical visual features (edges → textures → shapes → objects).

## ⚠️ Limitations

- The model is trained on 28×28 grayscale Fashion-MNIST images, not real-world photos
- Real-world clothing images with complex backgrounds, colors, and textures may not be classified accurately
- The model only recognizes 10 specific categories from Fashion-MNIST
- Training on CPU may take longer than on GPU
- The Streamlit app's predictions on uploaded photos are approximate due to the domain gap between Fashion-MNIST and real photographs

## 📝 Conclusion

This project demonstrates:
- How CNNs outperform fully connected networks for image classification
- The importance of convolutional and pooling layers for learning visual features
- Practical skills in model training, evaluation, and deployment
- How to analyze model errors and understand classification challenges

---

**SKILL NEXIS Four-Week Deep Learning Program — Week 3**

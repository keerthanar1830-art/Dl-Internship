"""
Fashion Item Recognition - Model Architectures.

This module provides two PyTorch neural network architectures for Fashion-MNIST classification:
1. FashionCNN: A Convolutional Neural Network consisting of three feature extraction blocks
   (Conv2d -> ReLU -> MaxPool2d) followed by fully connected classification layers with dropout.
2. FashionFC: A Fully Connected baseline Multi-Layer Perceptron (MLP) for performance comparison.
"""

import torch
import torch.nn as nn

# Default class labels for Fashion-MNIST
CLASS_NAMES = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


class FashionCNN(nn.Module):
    """
    Convolutional Neural Network for Fashion-MNIST Item Classification.

    Network Architecture:
    ---------------------
    - Feature Extractor:
        * Conv Block 1:
            - Conv2d(1, 32, kernel_size=3, padding=1): Extracts 32 low-level feature maps
              such as edges and borders while preserving 28x28 spatial dimensions.
            - ReLU: Non-linear activation function.
            - MaxPool2d(kernel_size=2): Downsamples resolution from 28x28 to 14x14.
        * Conv Block 2:
            - Conv2d(32, 64, kernel_size=3, padding=1): Extracts 64 mid-level feature maps
              representing complex textures and localized shapes (14x14).
            - ReLU: Non-linear activation function.
            - MaxPool2d(kernel_size=2): Downsamples resolution from 14x14 to 7x7.
        * Conv Block 3:
            - Conv2d(64, 128, kernel_size=3, padding=1): Extracts 128 high-level semantic
              feature maps representing garment parts (collars, soles, sleeves) (7x7).
            - ReLU: Non-linear activation function.
            - MaxPool2d(kernel_size=2): Downsamples resolution from 7x7 to 3x3.
    - Flatten:
        * Flattens the 128 x 3 x 3 feature volume into a 1,152-element feature vector.
    - Classifier:
        * Linear(1152, 256) -> ReLU -> Dropout(0.5): First dense projection with 50% dropout
          to prevent overfitting and co-adaptation of hidden features.
        * Linear(256, 128) -> ReLU -> Dropout(0.3): Second dense projection with 30% dropout.
        * Linear(128, 10): Output classification layer producing raw unnormalized class logits.
    """

    def __init__(self, num_classes=10):
        super(FashionCNN, self).__init__()

        # Conv Block 1: Input (batch, 1, 28, 28) -> Output (batch, 32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: Input (batch, 32, 14, 14) -> Output (batch, 64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: Input (batch, 64, 7, 7) -> Output (batch, 128, 3, 3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer: (batch, 128, 3, 3) -> (batch, 1152)
        self.flatten = nn.Flatten()

        # Fully Connected Classification Layers
        self.fc1 = nn.Linear(in_features=128 * 3 * 3, out_features=256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Execute forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images of shape (batch_size, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Raw output logits of shape (batch_size, 10).
        """
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3: Output spatial size is (batch_size, 128, 3, 3)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten to 1D vector per sample
        x = self.flatten(x)

        # Classification Head
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)

        logits = self.fc3(x)
        return logits

    def count_parameters(self):
        """
        Calculate the total number of trainable parameters in the model.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class FashionFC(nn.Module):
    """
    Fully Connected (Multi-Layer Perceptron) Baseline Network for Fashion-MNIST.

    Network Architecture:
    ---------------------
    - Flatten:
        * Converts the 28x28 2D grayscale image into a 1D vector of 784 pixel values.
    - Hidden Layer 1:
        * Linear(784, 512): Dense projection from raw pixel inputs to 512 hidden units.
        * ReLU: Non-linear activation function.
        * Dropout(0.3): Regularization with 30% dropout probability.
    - Hidden Layer 2:
        * Linear(512, 256): Intermediate hidden layer with 256 units.
        * ReLU: Non-linear activation function.
        * Dropout(0.3): Regularization with 30% dropout probability.
    - Hidden Layer 3:
        * Linear(256, 128): Intermediate hidden layer with 128 units.
        * ReLU: Non-linear activation function.
        * Dropout(0.2): Regularization with 20% dropout probability.
    - Output Layer:
        * Linear(128, 10): Computes unnormalized class logits for each of the 10 fashion categories.
    """

    def __init__(self, num_classes=10):
        super(FashionFC, self).__init__()

        # Flatten input from (batch, 1, 28, 28) or (batch, 28, 28) to (batch, 784)
        self.flatten = nn.Flatten()

        # Hidden Layer 1
        self.fc1 = nn.Linear(in_features=784, out_features=512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        # Hidden Layer 2
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        # Hidden Layer 3
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        # Output Layer
        self.fc4 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Execute forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images of shape (batch_size, 1, 28, 28) or (batch_size, 784).

        Returns
        -------
        torch.Tensor
            Raw output logits of shape (batch_size, 10).
        """
        # Flatten image to 1D vector of length 784
        x = self.flatten(x)

        # Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Layer 4 (Output)
        logits = self.fc4(x)
        return logits

    def count_parameters(self):
        """
        Calculate the total number of trainable parameters in the model.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

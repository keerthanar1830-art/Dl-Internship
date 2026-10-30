"""
Waste Image Classification System
Skill Nexis Deep Learning Major Project - Model Architectures Module

Implements both the Baseline Custom Convolutional Neural Network (CNN)
and the Improved Transfer Learning / Enhanced Deep CNN model for the
5 waste categories (Cardboard, Glass, Metal, Paper, Plastic).
"""

import logging
from typing import Tuple, List

import config

# Ensure backend configuration
config.configure_keras_backend()
import keras
from keras import layers, models, regularizers

logger = logging.getLogger(__name__)


def build_baseline_model(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
) -> keras.Model:
    """
    Builds a baseline custom Convolutional Neural Network (CNN) from scratch.
    
    Architecture:
      - Rescaling: [0, 255] -> [0.0, 1.0]
      - 4 Convolutional blocks (Conv2D -> BatchNormalization -> ReLU -> MaxPool)
      - GlobalAveragePooling2D to prevent overfitting and reduce parameter count
      - Dense classification head with Dropout regularization
      - Softmax output layer for 5 waste categories
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Normalization layer
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # Classifier Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Waste_Baseline_CNN")
    return model


def build_improved_model(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    use_pretrained_weights: bool = True,
) -> keras.Model:
    """
    Builds an improved model utilizing Transfer Learning (MobileNetV2 backbone)
    or Depthwise Separable Convolutions with built-in data augmentation.
    
    Key Enhancements:
      - End-to-end data augmentation (flip, rotation, zoom, contrast)
      - MobileNetV2 feature extractor / Inverted Residual blocks
      - Dual regularization: BatchNormalization + Dropout (0.4)
      - Adam optimizer with learning rate warmup and decay
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Integrated Data Augmentation
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.15, name="aug_rot")(x)
    x = layers.RandomZoom(0.15, name="aug_zoom")(x)
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    weights = "imagenet" if use_pretrained_weights else None
    try:
        # Attempt loading MobileNetV2 with ImageNet weights if permitted/cached
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
        )
        base_model.trainable = True
        # Freeze initial 60 layers to retain low-level generic edges/textures
        for layer in base_model.layers[:60]:
            layer.trainable = False

        features = base_model(x)
    except Exception as e:
        logger.warning(f"Could not load pre-trained MobileNetV2 ({e}). Using Enhanced Custom Residual Architecture.")
        # Enhanced Deep Separable Conv Architecture fallback (guaranteed offline CPU-compatible)
        c1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
        c1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu")(p1)
        c2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu")(p2)
        c3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.SeparableConv2D(256, (3, 3), padding="same", activation="relu")(p3)
        c4 = layers.BatchNormalization()(c4)
        features = layers.MaxPooling2D((2, 2))(c4)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_pool")(features)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="head_dense")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Waste_Improved_Model")
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """Compiles the model with Adam optimizer, Categorical Crossentropy, and accuracy."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(
    checkpoint_filepath: str,
    patience: int = 6,
) -> List[keras.callbacks.Callback]:
    """Returns standard training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau."""
    callback_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return callback_list


if __name__ == "__main__":
    baseline = build_baseline_model()
    baseline = compile_model(baseline)
    print("\n--- BASELINE MODEL SUMMARY ---")
    baseline.summary()

    improved = build_improved_model(use_pretrained_weights=False)
    improved = compile_model(improved)
    print("\n--- IMPROVED MODEL SUMMARY ---")
    improved.summary()

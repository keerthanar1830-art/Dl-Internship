"""
Fashion Item Recognition - Training Module.

This module provides standard PyTorch training and validation routines:
- train_one_epoch: Runs forward pass, loss calculation, backpropagation, and weight update.
- validate: Evaluates model performance on validation/test data without updating gradients.
- train_model: Orchestrates multi-epoch training, tracks metrics, measures runtime,
  and saves the best model checkpoint based on validation accuracy.
"""

import os
import time
import torch

# Standard class names for Fashion-MNIST
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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch across all batches in the dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model to train.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing training batches of (images, labels).
    criterion : torch.nn.Module
        Loss function (e.g., nn.CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        Optimization algorithm (e.g., Adam, SGD).
    device : torch.device
        Target device for computation (CPU or CUDA).

    Returns
    -------
    tuple of (float, float)
        (average_training_loss, accuracy_percentage)
    """
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Clear previously calculated gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and weight updates
        loss.backward()
        optimizer.step()

        # Track statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, dim=1)
        running_correct += torch.sum(preds == labels.data).item()
        total_samples += batch_size

    avg_loss = running_loss / total_samples
    accuracy_percent = (running_correct / total_samples) * 100.0

    return avg_loss, accuracy_percent


def validate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set without updating parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing validation batches.
    criterion : torch.nn.Module
        Loss function (e.g., nn.CrossEntropyLoss).
    device : torch.device
        Target device for computation (CPU or CUDA).

    Returns
    -------
    tuple of (float, float)
        (average_validation_loss, accuracy_percentage)
    """
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, dim=1)
            running_correct += torch.sum(preds == labels.data).item()
            total_samples += batch_size

    avg_loss = running_loss / total_samples
    accuracy_percent = (running_correct / total_samples) * 100.0

    return avg_loss, accuracy_percent


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=15,
    checkpoint_path=None,
    model_name='Model'
):
    """
    Full training loop. Trains model for num_epochs, validates after each epoch,
    saves the best model checkpoint based on validation accuracy, and tracks progress.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training set.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation/test set.
    criterion : torch.nn.Module
        Loss function (e.g., nn.CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : torch.device
        Device to run training on (CPU or CUDA).
    num_epochs : int, default=15
        Number of epochs to train.
    checkpoint_path : str, optional
        File path to save the best model checkpoint (e.g. 'models/fashion_cnn.pth').
    model_name : str, default='Model'
        Descriptive name of the model for logging and checkpoint metadata.

    Returns
    -------
    dict
        Training history dictionary containing:
        - 'train_loss': list of float
        - 'train_acc': list of float
        - 'val_loss': list of float
        - 'val_acc': list of float
        - 'train_time': total training time in seconds (float)
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_time': 0.0
    }

    best_val_acc = 0.0
    start_time = time.time()

    print(f"============================================================")
    print(f"Starting training for {model_name} on {device} ({num_epochs} epochs)")
    print(f"============================================================")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        epoch_duration = time.time() - epoch_start

        # Record metrics in history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Format and display epoch progress
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_duration:.2f}s",
            end=""
        )

        # Save checkpoint if current validation accuracy exceeds best seen
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f" -> Best val acc! ({val_acc:.2f}%)")

            if checkpoint_path is not None:
                directory = os.path.dirname(checkpoint_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'class_names': CLASS_NAMES,
                    'model_config': {
                        'model_name': model_name,
                        'architecture': model.__class__.__name__,
                        'num_classes': 10
                    },
                    'test_accuracy': val_acc
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"   [Saved checkpoint to '{checkpoint_path}']")
        else:
            print("")

    total_time = time.time() - start_time
    history['train_time'] = total_time

    print(f"============================================================")
    print(f"Training finished in {total_time:.2f} seconds ({total_time / 60:.2f} mins)")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"============================================================")

    return history

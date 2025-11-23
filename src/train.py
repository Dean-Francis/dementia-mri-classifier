import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


class EarlyStopping:
    """Stop training when validation loss stops improving"""
    def __init__(self, patience=10, min_delta=0.001, model_save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.model_save_path = model_save_path
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model, epoch)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self._save_checkpoint(model, epoch)
        return False

    def _save_checkpoint(self, model, epoch):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(self.model_save_path) or '.', exist_ok=True)
        torch.save(model.state_dict(), self.model_save_path)


def train_model(model, train_loader, val_loader, epochs=20, device='cpu',
                use_class_weights=False, class_weights=None):
    """
    Train the CNN model on dementia with early stopping and metrics tracking

    Args:
        model (nn.Module): The Dementia model to train
        train_loader (DataLoader): Training data loader (learns from images)
        val_loader (DataLoader): Validation data loader (evaluates model)
        epochs (int): Number of training epochs (default: 20)
        device (str): Device to train on ('cpu' or 'cuda')
        use_class_weights (bool): Use weighted loss for imbalanced classes
        class_weights (Tensor): Class weights for loss function

    Returns:
        dict: Dictionary containing training history and metrics
    """

    # Loss function for binary classification
    # CrossEntropyLoss: combines LogSoftmax and NLLLoss
    # Measures how far predicted probability is from true label
    if use_class_weights and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer: adjusts model weights to reduce loss
    # Adam: adaptive learning rate with momentum
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler: reduces learning rate if validation loss plateaus
    # Helps fine-tune the model after initial learning phases
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Minimize validation loss
        factor=0.5,           # Reduce LR by 50%
        patience=3,           # Wait 3 epochs before reducing
        min_lr=1e-6           # Don't go below this learning rate
    )

    # Early stopping: stop if validation loss doesn't improve
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.001,
        model_save_path='best_model.pth'
    )

    # Track metrics for analysis
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'val_precisions': [],
        'val_recalls': [],
        'val_f1s': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    # Training loop: iterate through epochs
    for epoch in range(epochs):
        # ===== TRAINING PHASE =====
        model.train()  # Enable dropout, batch norm updates
        train_loss = 0.0

        for images, labels in train_loader:
            # Move data to device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass: compute predictions
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()         # Backpropagation

            # Gradient clipping: prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

        # Average training loss and compute accuracy
        train_loss /= len(train_loader)
        metrics['train_losses'].append(train_loss)

        # ===== VALIDATION PHASE =====
        model.eval()  # Disable dropout, freeze batch norm
        val_loss = 0.0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():  # Disable gradient computation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate loss
                val_loss += loss.item()

                # Store predictions and labels for metrics
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels_list.extend(labels.cpu().numpy())

        # Average validation loss
        val_loss /= len(val_loader)
        metrics['val_losses'].append(val_loss)

        # Compute validation metrics
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels_list)

        val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary', zero_division=0
        )

        metrics['val_accs'].append(val_acc)
        metrics['val_precisions'].append(precision)
        metrics['val_recalls'].append(recall)
        metrics['val_f1s'].append(f1)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Check early stopping
        if early_stopping(val_loss, model, epoch):
            print(f"\nEarly stopping at epoch {epoch+1}")
            metrics['best_epoch'] = early_stopping.best_epoch
            break

        # Track best validation loss
        if val_loss < metrics['best_val_loss']:
            metrics['best_val_loss'] = val_loss
            metrics['best_epoch'] = epoch

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            cn_preds = np.sum(val_preds == 0)
            ad_preds = np.sum(val_preds == 1)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Val F1: {f1:.4f} | Preds: CN={cn_preds}, AD={ad_preds}")

    return metrics


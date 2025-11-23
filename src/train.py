import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any
from torch.utils.data import DataLoader

DeviceLikeType = str | torch.device | int

def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    epochs: int = 20,
    device: DeviceLikeType = 'cpu'
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    model.to(device)

    train_losses = []
    val_losses = []

    # these will store the last epoch's val accuracy
    last_val_correct = 0
    last_val_total = 0

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # (B,1)

            outputs = model(images)              # logits (B,1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)          # logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # accuracy
                probs = torch.sigmoid(outputs)   # (B,1)
                preds = (probs >= 0.5).float()   # (B,1) in {0,1}
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)      # batch size

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        last_val_correct = val_correct
        last_val_total = val_total

        if (epoch + 1) % 5 == 0:
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

    return train_losses, val_losses, last_val_correct, last_val_total

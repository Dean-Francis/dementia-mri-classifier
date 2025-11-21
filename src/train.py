import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=20, device='cpu'):
    """
    Train the CNN model on dementia

    Args:
    model (nn.Module): The Dementia model to train
    train_loader (DataLoader): for training data (80%) -> The model learns from the images
    val_loader (DataLoader): for validation data (20%) -> Model is tested on these images
    We're keeping a 80/20 split
    epochs (int): Number of training epochs (default: 20)
    device (str): Device to train on cpu (This will be the default if GPU doesn't work)

    Returns:
        tuple: (train_losses, val_losses) - list of loss values per epoch
    """
    
    # Loss function for binary classification
    # A loss function is a way for a neural network to measure how wrong its predictions are compared to the true labels.
        # measures how far predicted probability is from the true label
    # Binary classification: only 2 classes (AD vs CN)
    # Smaller loss = better predictions

    # Turns logits into probabilities
    # Negative log-likelyhood: Penalizes the network if the predicted probability for the correct class is low.
    # Applies softmax to logbits (turns logbit values into probabilities)
    # Computes negative log-likelihood of the true class: gives a loss value
    # Will be used for backpropagation
    # It checks how confident the model is about the correct class
    criterion = nn.CrossEntropyLoss()

    # This is where the model learns
    # When your model makes a prediction, it's usually wrong initially
    # The loss function (CrossEntropyLoss()) tells us how wrong
    # But that something has to fix the model after every mistake
    # That something is the optimizer
    # It takes the loss and adjusts the model's weights so next time the prediction is closer to correct
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # Learning rate scheduler: reduces learning rate if validation loss plateaus
    # This helps fine-tune the model after initial learning phases
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Minimize validation loss
        factor=0.5,           # Reduce LR by 50%
        patience=3            # Wait 3 epochs before reducing
    )

    # Lists to track losses for plotting and analysis
    train_losses = []
    val_losses = []

    # Training loop: iterate through epochs
    for epoch in range(epochs):
        # ===== TRAINING PHASE =====
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        train_loss = 0.0

        for images, labels in train_loader:
            # Move data to device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass: compute predictions
            outputs = model(images)

            # Compute loss: how wrong are the predictions?
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            # zero_grad() clears old gradients (important before each iteration)
            optimizer.zero_grad()
            loss.backward()

            # Update weights: adjust model parameters to reduce loss
            optimizer.step()

            # Accumulate loss for averaging
            train_loss += loss.item()

        # Average training loss for this epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ===== VALIDATION PHASE =====
        model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm)
        val_loss = 0.0

        with torch.no_grad():  # Disable gradient computation (faster, uses less memory)
            for images, labels in val_loader:
                # Move data to device
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute loss (but don't backprop)
                loss = criterion(outputs, labels)

                # Accumulate loss
                val_loss += loss.item()

        # Average validation loss for this epoch
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


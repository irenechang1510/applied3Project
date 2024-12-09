import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Compute loss on the testing set
def calculate_testing_loss(model, X_test_batch, y_test_batch, criterion):
    model.eval()
    model = model.to("cuda")
    total_loss = 0
    dtype = next(model.parameters()).dtype  # Get the data type of the model's parameters

    with torch.no_grad():  # Disable gradient computation for validation/testing
        for i in range(X_test_batch.shape[0]):
            # Ensure data types match the model's parameters
            X_batch = X_test_batch[i].to(dtype=dtype)
            y_batch = y_test_batch[i].to(dtype=dtype)
            # Forward pass
            y_pred = model(X_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    
    # Return the average loss
    return total_loss / X_test_batch.shape[0]


def update_mlp(mlp, X, y, num_epochs=10):
    mlp.train()
    mlp = mlp.to("cuda")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        dtype = next(mlp.parameters()).dtype  # Get the data type of the model's parameters
        X_batch = X.to(dtype=dtype) 
        y_batch = y.to(dtype=dtype)

        # Forward pass
        y_pred = mlp(X_batch.to("cuda"))

        loss = criterion(y_pred, y_batch.to("cuda"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Fine-tuning Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    return mlp




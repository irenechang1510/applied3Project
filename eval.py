import torch
import torch.nn as nn
import torch.optim as optim
from src.synthetic_data import blr, X_init, y_init, X_pool, X_val, y_val, X_init_batch, y_init_batch, X_pool_batch, X_val_batch, y_val_batch, X_test_batch, y_test_batch
import numpy as np
from src.synthetic_data import generate_true_y
# Function to generate true y for a given x


# load initial MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Model parameters
input_dim = 10
hidden_dim = 32  # Hidden layer size
output_dim = 1
assert output_dim==1

# Initialize the model, loss function, and optimizer
mlp = MLP(input_dim, hidden_dim, output_dim)
state_dict = torch.load("model/mlp_trained_model.pth")
mlp.load_state_dict(state_dict)

# Compute loss on the testing set
def calculate_testing_loss(model, X_test_batch, y_test_batch, criterion):
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


# def eval(mlp, X, X_test_batch, y_test_batch):
    # generate true y

X = torch.randn(10)
y = generate_true_y(X)

## finetune MLP
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

num_epochs_target = 1  # Number of epochs for target domain
for epoch in range(num_epochs_target):
    # Ensure the input and target tensors match the model's data type
    dtype = next(mlp.parameters()).dtype  # Get the data type of the model's parameters
    X_batch = X.to(dtype=dtype)  # Convert X to the same dtype as the model
    y_batch = y.to(dtype=dtype)  # Convert y to the same dtype as the model

    # Forward pass
    y_pred = mlp(X_batch)

    # Compute loss
    loss = criterion(y_pred, y_batch)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Fine-tuning Epoch [{epoch + 1}/{num_epochs_target}], Loss: {loss.item():.4f}")

# get loss on testing data
mlp.eval()

# Assuming X_val_batch and y_val_batch are properly defined
testing_loss = calculate_testing_loss(mlp, X_test_batch, y_test_batch, criterion)
print(f"Testing Loss: {testing_loss:.4f}")



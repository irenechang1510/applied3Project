from src.synthetic_data import X_source_batch, y_source_batch, X_init, y_init, X_init_batch, y_init_batch 

import torch
import torch.nn as nn
import torch.optim as optim

X_source_batch = X_source_batch
y_source_batch = y_source_batch
X_init_batch = X_init_batch
y_init_batch = y_init_batch

# Define the MLP model
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
input_dim = X_source_batch.shape[-1]
hidden_dim = 32  # Hidden layer size
output_dim = y_source_batch.shape[-1]
assert output_dim==1

# Initialize the model, loss function, and optimizer
mlp = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

# Training loop for the source domain
num_epochs_source = 50  # Number of epochs for source domain
for epoch in range(num_epochs_source):
    for i in range(X_source_batch.shape[0]):
        X_batch = X_source_batch[i]
        y_batch = y_source_batch[i]

        # Forward pass
        y_pred = mlp(X_batch)

        # Compute loss
        loss = criterion(y_pred, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs_source}], Loss: {loss.item():.4f}")

# Fine-tuning with the target domain (initialization batches)
num_epochs_target = 5  # Number of epochs for target domain
for epoch in range(num_epochs_target):
    for i in range(X_init_batch.shape[0]):
        X_batch = X_init_batch[i]
        y_batch = y_init_batch[i]

        # Forward pass
        y_pred = mlp(X_batch)

        # Compute loss
        loss = criterion(y_pred, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Fine-tuning Epoch [{epoch + 1}/{num_epochs_target}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(mlp.state_dict(), "model/mlp_trained_model.pth")
print("Model training and fine-tuning complete. Model saved.")

print("Dtype of model parameters after all training:")
for name, param in mlp.named_parameters():
    print(f"{name}: {param.dtype}")

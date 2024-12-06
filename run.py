import numpy as np
from turbo.utility import ExpectedImprovementCustom  # Import your utility function class
from src.turbo_m import TurboM
from src.synthetic_data import X_init, y_init, X_pool, X_val, y_val, X_init_batch, y_init_batch, X_pool_batch, X_val_batch, y_val_batch, X_source_batch, y_source_batch
import torch
import torch.nn as nn, torch.optim as optim
from train import MLP

# Problem dimensions and bounds
dim = 10
lb = np.zeros(dim)
ub = np.ones(dim)

X_source_batch = X_source_batch.float()
y_source_batch = y_source_batch.float()
X_init_batch = X_init_batch.float()
y_init_batch = y_init_batch.float()

input_dim = X_source_batch.shape[-1]
hidden_dim = 32  # Hidden layer size
output_dim = y_source_batch.shape[-1]
assert output_dim==1

# Initialize the model, loss function, and optimizer
mlp = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

# Training loop for the source domain
num_epochs_source = 20  # Number of epochs for source domain
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

torch.save(mlp, 'model/mlp_trained_model.pth')
# Initialize the TuRBO-M optimizer
turbo_m = TurboM(
    f=mlp,
    lb=lb,
    ub=ub,
    n_init=dim,
    max_evals=21,
    n_trust_regions=2,
    batch_size=1,
    verbose=True,
    utility_function_class=ExpectedImprovementCustom,
    X_init=X_init_batch,
    fX_init=y_init_batch,
    X_val = X_val_batch,
    fX_val = y_val_batch
)

# Run optimization
turbo_m.optimize()

# Retrieve the evaluated points and corresponding function values
X, fX = turbo_m.X, turbo_m.fX

# Print the best found value
print(f"Best value found: {fX.min():.4f}")
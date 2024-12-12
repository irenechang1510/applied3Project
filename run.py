import faulthandler
faulthandler.enable()

import numpy as np
from turbo.utility import ExpectedImprovementCustom  # Import your utility function class
from src.turbo_m import TurboM
from src.synthetic_data import U_true, X_init_batch, y_init_batch, X_pool_batch, y_pool_batch,  X_val_batch, y_val_batch, X_source_batch, y_source_batch, X_test_batch, y_test_batch
import torch
import torch.nn as nn, torch.optim as optim
from train import MLP

# Problem dimensions and bounds
dim = X_init_batch.shape[1]
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
num_epochs_target = 1  # Number of epochs for target domain
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
n_trust_regions = 3 # FIXED
mlp = mlp.to("cuda")
# EVALUATION GOAL: show that the longer the autoregressive generation window is the 
# better it gets
loss_turbo_k, loss_rand_k = [], []
torch.manual_seed(0)
for k in range(50, 500, 50):
    X_init_batch_k, y_init_batch_k = X_init_batch[:k], y_init_batch[:k]
    turbo_m = TurboM(
        f=mlp,
        lb=lb,
        ub=ub,
        n_init=X_init_batch.shape[0],
        max_evals=X_init_batch.shape[0] * k + 1,
        n_trust_regions=n_trust_regions,
        batch_size=1,
        n_repeats = 10,
        generation_window_frac_10=5,
        verbose=True,
        utility_function_class=ExpectedImprovementCustom,
        X_init=X_init_batch.to("cuda"),
        fX_init=y_init_batch_k.to("cuda"),
        X_val = X_val_batch.to("cuda"),
        fX_val = y_val_batch.to("cuda"),
        device="cuda",
    )

    # Run optimization
    turbo_m.optimize()

    # Retrieve the evaluated points and corresponding function values
    X, fX = turbo_m.X, turbo_m.fX

    # Print the best found value + eval
    print(f"Best value found: {fX.min():.4f}")
    from eval import update_mlp, calculate_testing_loss
    from copy import deepcopy
    best_X_turbo = X[fX.argmin()]
    # reveal true Y label: calc XU (where we know true U), then add random noise
    true_Y_turbo = torch.normal((best_X_turbo * U_true).squeeze(), 1)
    # update MLP
    mlp_turbo = deepcopy(mlp)
    mlp_turbo = update_mlp(mlp_turbo, best_X_turbo, true_Y_turbo, num_epochs=5)
    # Calc MSE LOSS on test set
    testing_loss_turbo = calculate_testing_loss(mlp_turbo, X_test_batch.to("cuda"), y_test_batch.to("cuda"), criterion)
    loss_turbo_k.append(testing_loss_turbo)
    print(f"Testing Loss turbo: {testing_loss_turbo:.4f}")
    print(loss_turbo_k)

# best_X_random
rand_loss = []
rand_choice = torch.randint(0, X_pool_batch.shape[0], size=(10,))
for i in range(10):
    best_X_rand = X_pool_batch[rand_choice[i]]
    true_Y_rand = y_pool_batch[rand_choice[i]]
    mlp_rand = deepcopy(mlp)
    mlp_rand = update_mlp(mlp_rand, best_X_rand, true_Y_rand, num_epochs=5)
    rand_loss_baseline = calculate_testing_loss(mlp_rand, X_test_batch.to("cuda"), y_test_batch.to("cuda"), criterion)
    rand_loss.append(rand_loss_baseline)

testing_loss_baseline = np.mean(rand_loss)
loss_rand_k.append(testing_loss_baseline)

print(f"Testing Loss rand: {testing_loss_baseline:.4f}")
print(loss_rand_k)
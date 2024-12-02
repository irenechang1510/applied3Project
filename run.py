import numpy as np
from turbo.utility import ExpectedImprovementCustom  # Import your utility function class
from src.turbo_m import TurboM
from src.synthetic_data import X_init, y_init, X_pool, X_val, y_val, X_init_batch, y_init_batch, X_pool_batch, X_val_batch, y_val_batch

# Problem dimensions and bounds
dim = 10
lb = np.zeros(dim)
ub = np.ones(dim)

# Define the objective function to minimize
def sphere_function(x):
    return np.sum(x ** 2)

# Initialize the TuRBO-M optimizer
turbo_m = TurboM(
    f=sphere_function,
    lb=lb,
    ub=ub,
    n_init=2 * dim,
    max_evals=200,
    n_trust_regions=5,
    batch_size=1,
    verbose=True,
    utility_function_class=ExpectedImprovementCustom,
    X_init=X_init_batch,
    fX_init=y_init_batch
)

# Run optimization
turbo_m.optimize()

# Retrieve the evaluated points and corresponding function values
X, fX = turbo_m.X, turbo_m.fX

# Print the best found value
print(f"Best value found: {fX.min():.4f}")
import numpy as np
from scipy.stats import multivariate_normal
import torch

class BayesianLinearRegression:
    def __init__(self, mu_x, Sigma_x, mu_u, Sigma_u, sigma_y, dimX):
        self.mu_x = mu_x
        self.Sigma_x = Sigma_x
        self.mu_u = mu_u
        self.Sigma_u = Sigma_u
        self.sigma_y = sigma_y
        self.U = multivariate_normal.rvs(mean=self.mu_u, cov=self.Sigma_u)

        self.dimX = dimX
        # Ensure dimensions are correct
        assert self.mu_x.shape == (dimX,), "mu_x should be a vector of length dimX"
        assert self.Sigma_x.shape == (dimX, dimX), "Sigma_x should be a matrix of shape (dimX, dimX)"
        assert self.mu_u.shape == (dimX,), "mu_u should be a vector of length dimX"
        assert self.Sigma_u.shape == (dimX, dimX), "Sigma_u should be a matrix of shape (dimX, dimX)"
    
    def generate_data_given_U(self, U, n_samples, seed=1, logistic=False, epsilon=1):
        self.U = U
        np.random.seed(seed)

        # Generate X based on whether dimX is 1 or more
        if self.dimX == 1: 
            X = np.random.normal(self.mu_x.flatten(), self.Sigma_x.flatten(), n_samples)
        else:
            X = np.random.multivariate_normal(self.mu_x, self.Sigma_x, n_samples)
        
        epsilon = np.random.normal(0, self.sigma_y, n_samples)
        y = X.dot(U) + epsilon 

        if logistic:
            y_sigmoid = 1 / (1 + np.exp(-y))
            y = np.random.binomial(1, y_sigmoid)

        # If dimX=1, reshape X to (n_samples, 1)
        if self.dimX == 1:
            X = X.reshape(-1, 1)
        
        # Convert everything to torch.float32 for consistency
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        epsilon = torch.tensor(epsilon, dtype=torch.float32)

        return X, y, epsilon

    def sample_from_posterior(self, posterior_mean, posterior_cov):
        return multivariate_normal.rvs(mean=posterior_mean, cov=posterior_cov)


# GENERATE DATA FOR TARGET DOMAIN
dimX = 1
mu_x = np.zeros(dimX)
Sigma_x = np.eye(dimX)
mu_u = np.zeros(dimX)
Sigma_u = np.eye(dimX)
sigma_y = 1

n_init_samples, n_unlabeled_samples, n_val_samples = 200, 100, 200
blr = BayesianLinearRegression(mu_x, Sigma_x, mu_u, Sigma_u, sigma_y, dimX)
U_true = blr.U
seed1, seed2, seed3 = 1, 42, 20

# Now these returned values are already torch tensors (float32)
X_init, y_init, _ = blr.generate_data_given_U(blr.U, n_init_samples, seed=seed1, logistic=False, epsilon=None)
X_pool, y_pool, _ = blr.generate_data_given_U(blr.U, n_unlabeled_samples, seed=seed2, logistic=False, epsilon=None)
X_val, y_val, _ = blr.generate_data_given_U(blr.U, n_val_samples, seed=seed3, logistic=False, epsilon=None)

batch_size = 10
X_init_batch = X_init.reshape((-1, batch_size))
y_init_batch = y_init.reshape((-1, batch_size))
X_pool_batch = X_pool.reshape((-1, batch_size))
X_val_batch = X_val.reshape((-1, batch_size))
y_val_batch = y_val.reshape((-1, batch_size))

# GENERATE DATA FOR SOURCE DOMAIN
dimX_source = 1
mu_x_source = np.ones(dimX_source)
Sigma_x_source = 2 * np.eye(dimX_source)
mu_u_source = np.ones(dimX_source)
Sigma_u_source = 1.5 * np.eye(dimX_source)
sigma_y_source = 1.5

n_source_samples = 1000
seed_source = 123

blr_source = BayesianLinearRegression(mu_x_source, Sigma_x_source, mu_u_source, Sigma_u_source, sigma_y_source, dimX_source)
U_source = blr_source.U
X_source, y_source, _ = blr_source.generate_data_given_U(U_source, n_source_samples, seed=seed_source, logistic=False, epsilon=None)

X_source_batch = X_source.reshape((-1, batch_size))
y_source_batch = y_source.reshape((-1, batch_size)).mean(dim=1, keepdim=True)

import sys
sys.path.append('~/Documents/GitHub/applied3Project')
from botorch.acquisition import AcquisitionFunction
import torch
from src.UtilityFunction import UtilityFunction
from src.synthetic_data import X_init, y_init, X_pool, X_val, y_val, X_init_batch, y_init_batch, X_pool_batch, X_val_batch, y_val_batch
from torch.nn import MSELoss
from train import MLP

# import os
# current_file_directory = os.path.dirname(os.path.abspath(__file__))
# parent_directory = os.path.dirname(current_file_directory)
# data_dir = os.path.join(parent_directory, "sampling")
# output_dir = os.path.join(data_dir, "figs")  # Directory to save the figures
myMLP = torch.load('model/mlp_trained_model.pth')

class ExpectedImprovementCustom(AcquisitionFunction):
    """Custom Expected Improvement acquisition function."""

    def __init__(self, model, best_f, init_set, val_set, n_repeats):
        """Initialize the acquisition function.

        Parameters
        ----------
        model : gpytorch.models.ExactGP
            The trained GP model.
        best_f : float
            The best (minimum) objective function value observed so far.
        """
        super().__init__(model)
        self.best_f = best_f
        self.X_init, self.y_init = init_set
        self.X_val, self.y_val = val_set # these are batched
        self.n_repeats = n_repeats

    def forward(self, X):
        """Compute the Expected Improvement at points X.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_batch, d).

        Returns
        -------
        torch.Tensor
            Expected Improvement values at X, shape (n_batch,).
        """
        # posterior = self.model(X)
        # mean = posterior.mean.view(-1)
        # sigma = posterior.variance.clamp_min(1e-9).sqrt().view(-1)

        # # Ensure sigma is positive
        # sigma = torch.where(sigma > 0, sigma, torch.tensor(1e-9, device=sigma.device))

        # # Compute the standard normal CDF and PDF
        # u = (self.best_f - mean) / sigma
        # normal = torch.distributions.Normal(0, 1)
        # ucdf = normal.cdf(u)
        # updf = normal.log_prob(u).exp()

        # # Compute Expected Improvement
        # ei = sigma * (u * ucdf + updf)
        # with torch.set_grad_enabled(True):
        mean_squared_error = MSELoss()
        # ufunc = UtilityFunction(mean_squared_error, f) # TODO: HOW TO PASS f IN
        ufunc = UtilityFunction(mean_squared_error, myMLP, n_repeats=self.n_repeats)
        X_detached = X.detach().requires_grad_()
        X_reshaped = X_detached.contiguous().reshape((-1,1))
        before_loss = ufunc(self.X_init.view((-1,1)), self.y_init, None, self.X_val, self.y_val)
        after_loss = ufunc(self.X_init.view((-1,1)), self.y_init, X_reshaped, self.X_val, self.y_val)
        # print(loss)
        reduc = (before_loss - after_loss).view(-1)

        return reduc

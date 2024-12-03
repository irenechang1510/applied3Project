from botorch.acquisition import AcquisitionFunction
import torch
from src.UtilityFunction import UtilityFunction
from src.synthetic_data import X_init, y_init, X_pool, X_val, y_val, X_init_batch, y_init_batch, X_pool_batch, X_val_batch, y_val_batch
from torch.nn import MSELoss
from train import MLP
myMLP = torch.load('model/mlp_trained_model.pth')

class ExpectedImprovementCustom(AcquisitionFunction):
    """Custom Expected Improvement acquisition function."""

    def __init__(self, model, best_f):
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
        mean_squared_error = MSELoss()
        # ufunc = UtilityFunction(mean_squared_error, f) # TODO: HOW TO PASS f IN
        ufunc = UtilityFunction(mean_squared_error, myMLP)
        X = X.reshape((-1,1))
        ## TODO: implement the loss before adding new batch 
        before_loss = ufunc(X_init, y_init, None, X_val_batch, y_val_batch)
        after_loss = ufunc(X_init, y_init, X, X_val_batch, y_val_batch)
        # print(loss)
        return before_loss - after_loss

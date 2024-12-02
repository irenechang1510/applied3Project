import numpy as np
import pandas as pd
import gpytorch
import torch
from scipy.stats import multivariate_normal
from torch.nn import MSELoss
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from typing import Literal
import copy

# GP REGRESSION MODULE
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x = None, train_y = None, 
                 likelihood = GaussianLikelihood(), 
                 mean_module = gpytorch.means.ZeroMean(), 
                 covar_module = gpytorch.kernels.RBFKernel()):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def predict(self, x, **kwargs):
        self.likelihood.eval()
        self.eval()
        return self.likelihood(self(x)).mean.reshape(-1, 1)
    
    def update(self, X, y):
        model_device = next(self.parameters()).device
        if X.device != model_device:
            X = X.to(model_device)
        if y.device != model_device:
            y = y.to(model_device)

        fantasy_model = self.get_fantasy_model(X, y) # which device does this put on
        
        return fantasy_model


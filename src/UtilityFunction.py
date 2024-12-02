
import numpy as np
import gpytorch, torch, torchvision
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from src.GaussianProcess import GPRegressionModel
import torch.optim as optim
from src.utils import ignoreWarnings

ignoreWarnings()
class UtilityFunction(nn.Module):
    def __init__(self, loss_function, classifier, n_repeats = 100):
        '''
        loss_function: e.g MSE loss
        classifier: e.g 
        '''
        super().__init__()
        self.loss_function = loss_function
        self.classifier = deepcopy(classifier) # this will be the mlp
        self.n_repeats = n_repeats
    
    def forward(self, 
                X_init: np.ndarray, y_init: np.ndarray, 
                X_rem: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        '''
        (X_init, y_init): \mathcal{D}_0 
        X_rem: batch b = \{(x_i, y_i)\}_{i=1}^n

        '''
        loss_list = []
        # repeat to account for randomness in sampling Y from ppd
        for seed in tqdm(range(self.n_repeats)):
            newX, newy = self.sample_conditional_mean_gp(
                torch.tensor(X_init), torch.tensor(y_init), 
                torch.tensor(X_rem) if X_rem is not None else None, 
                seed)
            
            # refit the posterior classifier on X_context and y_context
            self.update_predictive_model(newX, newy)
            
            self.classifier.eval()
            with torch.no_grad(): new_preds = self.classifier(X_val)
            loss = self.loss_function(new_preds, y_val)
            loss_list.append(loss)

        return torch.tensor(loss_list).mean().reshape(1, 1)

    def update_predictive_model(self, X, y):
        '''
        update the MLP with an extra (X, y) pair using 1-step gradient update
        '''
        self.classifier.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = self.classifier(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        return self.classifier

    def sample_conditional_mean_gp(self, X_init, y_init, X_rem = None, seed = None):
        sequence_model = GPRegressionModel(train_x=X_init.flatten(), train_y=y_init.flatten()).double()

        #bootstrap generation_window samples from X_rem
        np.random.seed(seed)        
        #the autoregressive generation loop
        X_context, y_context = deepcopy(X_init), deepcopy(y_init)
        sequence_model.eval()
        sequence_model.likelihood.eval()
        generation_window = X_rem.shape[0] if X_rem is not None else 0
        for i in range(generation_window):
            next_token = X_rem[i]
            y_hat= sequence_model.predict(next_token)
            y_hat = y_hat.to('cpu') if hasattr(y_hat, 'device') else y_hat
            y_hat = torch.tensor(y_hat) if not isinstance(y_hat, torch.Tensor) else y_hat
            if y_hat.ndim != 2: 
                y_hat = y_hat.unsqueeze(0).unsqueeze(1) if y_hat.ndim==0 else y_hat.unsqueeze(1)

            sequence_model = sequence_model.update(next_token, y_hat)
            X_context = torch.vstack([X_context, next_token])
            y_context = torch.vstack([y_context, y_hat])
            
        return X_context.detach().numpy(), y_context.detach().numpy()

# EXAMPLE USAGE
# from src.synthetic_data import X_init, y_init, X_pool, X_val, y_val
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression # replace with MLP (also need to change line 38)
# from src.UtilityFunction import UtilityFunction
# lr = LinearRegression()
# ufunc = UtilityFunction(mean_squared_error, lr)
# loss = ufunc(X_init, y_init, None, X_val, y_val)
# print(loss)
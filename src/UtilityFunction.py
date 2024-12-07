import sys
sys.path.append('/Users/irenechang/Documents/GitHub/applied3Project')
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
    def __init__(self, loss_function, classifier, n_repeats = 100, batch_size = 10):
        '''
        loss_function: e.g MSE loss
        classifier: e.g 
        '''
        super().__init__()
        self.loss_function = loss_function
        self.classifier = deepcopy(classifier) # this will be the mlp
        self.n_repeats = n_repeats
        self.batch_size = batch_size
    
    def forward(self, 
            X_init: torch.Tensor, y_init: torch.Tensor, 
            X_rem: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor):

        # Ensure that X_init requires grad
        # X_rem.requires_grad_(True)
        # assert X_rem.requires_grad, "X_init does not require grad, please set requires_grad=True."

        loss_list = []
        # repeat to account for randomness in sampling Y from ppd
        for seed in tqdm(range(self.n_repeats)):
            newX, newy = self.sample_conditional_mean_gp(X_init, y_init, X_rem if X_rem is not None else None,seed)
            # print(f"newX: {newX.shape}")
            # print(f"newy: {newy.shape}")
            # refit the posterior classifier on X_context and y_context
            self.update_predictive_model(newX, newy)
            
            self.classifier.eval()
            
            X_val = X_val.float()  # Convert to torch.float32
            y_val = y_val.float()
            # classifer = myMLP -> Xval shold be batched
            new_preds = self.classifier(X_val)
            loss = self.loss_function(new_preds, y_val)
            loss_list.append(loss)

        return torch.stack(loss_list).mean().reshape(1, 1)

    def update_predictive_model(self, X, y):
        '''
        update the MLP with an extra (X, y) pair using 1-step gradient update
        '''
        # reshape into the right batch size
        # print(f"X: {X.shape}")
        # print(f"y: {y.shape}")
        with torch.enable_grad():
            X_reshaped = X.view((-1, self.batch_size)).float()
            y_reshaped = y.float()
            # print(f"X_reshaped: {X_reshaped.shape}")
            # print(f"y_reshaped: {y_reshaped.shape}")
            # print(X_reshaped.dtype)  # should print torch.float32
            # print(y_reshaped.dtype)  # should print torch.float32

            self.classifier.train()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)

            optimizer.zero_grad()

            # X_reshaped should be batched
            outputs = self.classifier(X_reshaped)
            # print(f"outputs: {outputs.shape}")
            # print(f"y_reshaped: {y_reshaped.shape}")
            loss = criterion(outputs, y_reshaped)
            loss.backward()
            optimizer.step()
        return self.classifier

    def sample_conditional_mean_gp(self, X_init, y_init, X_rem = None, seed = None):
        # print(f"train x:{X_init.shape}")
        y_init = y_init.repeat_interleave(self.batch_size) 
        # print(f"train y:{y_init.shape}")
        sequence_model = GPRegressionModel(train_x=X_init, train_y=y_init).double()

        #bootstrap generation_window samples from X_rem
        np.random.seed(seed)        
        #the autoregressive generation loop
        X_context, y_context = deepcopy(X_init).requires_grad_(), deepcopy(y_init).requires_grad_()
        sequence_model.eval()
        sequence_model.likelihood.eval()
        generation_window = X_rem.shape[0] if X_rem is not None else 0
        for i in range(generation_window):
            next_token = X_rem[i]
            # print(f"next token:{next_token.shape}")
            y_hat= sequence_model.predict(next_token)
            y_hat = y_hat.to('cpu') if hasattr(y_hat, 'device') else y_hat
            y_hat = torch.tensor(y_hat) if not isinstance(y_hat, torch.Tensor) else y_hat
            # if y_hat.ndim != 2: 
            #     y_hat = y_hat.unsqueeze(0).unsqueeze(1) if y_hat.ndim==0 else y_hat.unsqueeze(1)
            y_hat = y_hat.view(-1)

            sequence_model = sequence_model.update(next_token, y_hat) 
            X_context = torch.vstack([X_context, next_token])
            # print(f"y_context: {y_context.shape}")
            y_context = torch.cat((y_context, y_hat),dim=0)
            
        return X_context, y_context

# EXAMPLE USAGE
import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch

from turbo.gp import train_gp  # Ensure this module is available
from turbo import Turbo1  # Ensure this module is available
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube
from turbo.utility import ExpectedImprovementCustom
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

class TurboM(Turbo1):
    """The TuRBO-M algorithm with a customizable acquisition function.

    Parameters
    ----------
    f : callable
        The objective function to minimize.
    lb : numpy.ndarray
        Lower bounds of the variables, shape (d,).
    ub : numpy.ndarray
        Upper bounds of the variables, shape (d,).
    n_init : int
        Number of initial points *for each trust region* (2 * dim is recommended).
    max_evals : int
        Total evaluation budget.
    n_trust_regions : int
        Number of trust regions.
    batch_size : int, optional
        Number of points in each batch (default is 1).
    verbose : bool, optional
        If True, prints information about the optimization progress (default is True).
    use_ard : bool, optional
        If True, uses Automatic Relevance Determination for the GP kernel (default is True).
    max_cholesky_size : int, optional
        Largest number of training points where Cholesky decomposition is used (default is 2000).
    n_training_steps : int, optional
        Number of training steps for learning the GP hyperparameters (default is 50).
    min_cuda : int, optional
        Minimum number of points to use CUDA for GP fitting (default is 1024).
    device : str, optional
        Device to use for GP fitting ("cpu" or "cuda") (default is "cpu").
    dtype : str, optional
        Data type to use for GP fitting ("float32" or "float64") (default is "float64").
    utility_function_class : class, optional
        Custom utility function class for the acquisition function (default is None).
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        n_trust_regions,
        X_init=None,
        fX_init=None,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        utility_function_class=None,
    ):
        self.n_trust_regions = n_trust_regions
        self.X_init = X_init
        self.fX_init = fX_init
        
        super().__init__(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=max_evals,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            n_training_steps=n_training_steps,
            min_cuda=min_cuda,
            device=device,
            dtype=dtype,
        )
        

        self.succtol = 3
        self.failtol = max(5, self.dim)
        self.utility_function_class = utility_function_class or ExpectedImprovementCustom

        # Very basic input checks
        assert n_trust_regions > 1 and isinstance(max_evals, int)
        assert max_evals > n_trust_regions * n_init, "Not enough evaluations for initial points"
        assert max_evals > batch_size, "Not enough evaluations to do a single batch"

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._idx = np.zeros((0, 1), dtype=int)  # Track which trust region proposed each point
        self.failcount = np.zeros(self.n_trust_regions, dtype=int)
        self.succcount = np.zeros(self.n_trust_regions, dtype=int)
        self.length = self.length_init * np.ones(self.n_trust_regions)

    def _adjust_length(self, fX_next, i):
        assert 0 <= i <= self.n_trust_regions - 1

        fX_min = self.fX[self._idx[:, 0] == i, 0].min()  # Best value in trust region i
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)  # Add batch size for this trust region

        if self.succcount[i] == self.succtol:
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.dim))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand), y_cand.shape)
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            y_cand[i, j] = np.inf  # Exclude this point from future selection

        return X_next, idx_next

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
    # Convert data to torch tensors
        X_torch = torch.tensor(X, dtype=self.dtype, device=self.device)
        fX_torch = torch.tensor(fX, dtype=self.dtype, device=self.device)

        # Normalize the outputs
        y_mean = fX_torch.mean()
        y_std = fX_torch.std()
        fX_normalized = (fX_torch - y_mean) / y_std

        # Train GP model using your train_gp function
        if hypers is not None and hypers != {}:
            # If hypers are provided, load them and set num_steps to 0
            model = train_gp(
                train_x=X_torch,
                train_y=fX_normalized,
                use_ard=self.use_ard,
                num_steps=0,
                hypers=hypers,
            )
        else:
            # Train the model and get new hypers
            model = train_gp(
                train_x=X_torch,
                train_y=fX_normalized,
                use_ard=self.use_ard,
                num_steps=n_training_steps,
            )
            # Save the model's state_dict as hypers for future use
            hypers = model.state_dict()

        # Define the acquisition function using your utility function class
        best_f = fX_normalized.min() #.item()
        acquisition_function = self.utility_function_class(model, best_f=best_f)

        # Optimize the acquisition function to find candidate points
        X_cand = self.optimize_acquisition_function(acquisition_function, length)
        y_cand = acquisition_function(X_cand)

        X_cand_np = X_cand.detach().cpu().numpy()
        y_cand_np = y_cand.detach().cpu().numpy().ravel()
        return X_cand_np, y_cand_np, hypers

    # def optimize_acquisition_function(self, acquisition_function, length):
    #     # Ensure self.center is a torch tensor
    #     if not isinstance(self.center, torch.Tensor):
    #         self.center = torch.tensor(self.center, dtype=self.dtype, device=self.device)
        
    #     # Define bounds for the trust region using torch.clamp
    #     tr_lb = torch.clamp(self.center - length / 2.0, min=0.0, max=1.0)
    #     tr_ub = torch.clamp(self.center + length / 2.0, min=0.0, max=1.0)
        
    #     # Stack bounds into a tensor
    #     bounds = torch.stack([tr_lb, tr_ub])

    #     # Generate initial points within the bounds
    #     X_init = latin_hypercube(self.n_cand, self.dim)
    #     X_init = torch.tensor(X_init, dtype=self.dtype, device=self.device)
    #     X_init = tr_lb + (tr_ub - tr_lb) * X_init  # Scale to trust region bounds

    #     # Optimize the acquisition function
    #     X_cand, _ = optimize_acqf(
    #         acq_function=acquisition_function,
    #         bounds=bounds,
    #         q=self.batch_size,
    #         num_restarts=10,
    #         raw_samples=100,
    #         options={"batch_limit": 5, "maxiter": 200},
    #     )
    #     return X_cand
    
    def optimize_acquisition_function(self, acquisition_function, length):
    # Ensure self.center is a torch tensor
        if not isinstance(self.center, torch.Tensor):
            self.center = torch.tensor(self.center, dtype=self.dtype, device=self.device)
        
        # Define bounds for the trust region using torch.clamp
        tr_lb = torch.clamp(self.center - length / 2.0, min=0.0, max=1.0)
        tr_ub = torch.clamp(self.center + length / 2.0, min=0.0, max=1.0)
        
        # Ensure bounds have the correct shape
        tr_lb = tr_lb.view(-1)
        tr_ub = tr_ub.view(-1)
        bounds = torch.stack([tr_lb, tr_ub])

        # Increase raw_samples and num_restarts for higher dimensions
        raw_samples = max(156, 10 * self.dim)
        num_restarts = max(20, 2 * self.dim)

        # Optimize the acquisition function
        X_cand, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=self.batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )

        return X_cand



    def optimize(self):
        """Run the full optimization process."""
        # Check if initial data is provided
        if self.X_init is not None and self.fX_init is not None:
            # Use the provided initial data
            assert self.X_init.shape[0] == self.fX_init.shape[0], "Mismatch in initial data sizes"
            assert self.X_init.shape[1] == self.dim, "Dimension mismatch in X_init"
            n_init_total = self.X_init.shape[0]
            n_init_per_tr = n_init_total // self.n_trust_regions

            # Assign initial data to each trust region
            for i in range(self.n_trust_regions):
                idx_start = i * n_init_per_tr
                idx_end = idx_start + n_init_per_tr
                X_init_i = self.X_init[idx_start:idx_end, :]
                fX_init_i = self.fX_init[idx_start:idx_end]

                # Update budget and set as initial data for this trust region
                self.X = np.vstack((self.X, X_init_i))
                self.fX = np.vstack((self.fX, fX_init_i))
                self._idx = np.vstack((self._idx, i * np.ones((n_init_per_tr, 1), dtype=int)))
                self.n_evals += n_init_per_tr

                if self.verbose:
                    fbest = fX_init_i.min()
                    print(f"TR-{i} starting from provided data: {fbest:.4f}")
                    sys.stdout.flush()
        else:
            # Create initial points for each trust region using LHS
            for i in range(self.n_trust_regions):
                X_init = latin_hypercube(self.n_init, self.dim)
                X_init = from_unit_cube(X_init, self.lb, self.ub)
                fX_init = np.array([[self.f(x)] for x in X_init])

                # Update budget and set as initial data for this trust region
                self.X = np.vstack((self.X, X_init))
                self.fX = np.vstack((self.fX, fX_init))
                self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
                self.n_evals += self.n_init

                if self.verbose:
                    fbest = fX_init.min()
                    print(f"TR-{i} starting from: {fbest:.4f}")
                    sys.stdout.flush()


        # Main optimization loop
        while self.n_evals < self.max_evals:

            # Generate candidates from each trust region
            X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim))
            y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand))
            for i in range(self.n_trust_regions):
                idx = np.where(self._idx == i)[0]  # Indices of points in trust region i

                # Get the points and values for trust region i
                X = deepcopy(self.X[idx, :])
                X = to_unit_cube(X, self.lb, self.ub)
                fX = deepcopy(self.fX[idx, 0].ravel())

                # Determine if model retraining is needed
                n_training_steps = 0 if self.hypers[i] else self.n_training_steps

                # Create new candidates
                self.center = X[fX.argmin(), :]  # Update center to best point in TR
                X_cand[i, :, :], y_cand[i, :], self.hypers[i] = self._create_candidates(
                    X,
                    fX,
                    length=self.length[i],
                    n_training_steps=n_training_steps,
                    hypers=self.hypers[i],
                )

            # Select the next candidates
            X_next, idx_next = self._select_candidates(X_cand, y_cand)
            assert X_next.min() >= 0.0 and X_next.max() <= 1.0

            # Undo the warping
            X_next = from_unit_cube(X_next, self.lb, self.ub)

            # Evaluate batch
            fX_next = np.array([[self.f(x)] for x in X_next])

            # Update trust regions
            for i in range(self.n_trust_regions):
                idx_i = np.where(idx_next == i)[0]
                if len(idx_i) > 0:
                    self.hypers[i] = {}  # Remove model hypers
                    fX_i = fX_next[idx_i]

                    if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * abs(self.fX.min()):
                        n_evals, fbest = self.n_evals, fX_i.min()
                        print(f"{n_evals}) New best @ TR-{i}: {fbest:.4f}")
                        sys.stdout.flush()
                    self._adjust_length(fX_i, i)

            # Update budget and append data
            self.n_evals += self.batch_size
            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))
            self._idx = np.vstack((self._idx, deepcopy(idx_next)))

            # Check if any trust region needs to be restarted
            for i in range(self.n_trust_regions):
                if self.length[i] < self.length_min:
                    idx_i = self._idx[:, 0] == i

                    if self.verbose:
                        n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                        print(f"{n_evals}) TR-{i} converged to: {fbest:.4f}")
                        sys.stdout.flush()

                    # Reset length and counters, remove old data from trust region
                    self.length[i] = self.length_init
                    self.succcount[i] = 0
                    self.failcount[i] = 0
                    self._idx[idx_i, 0] = -1  # Remove points from trust region
                    self.hypers[i] = {}  # Remove model hypers

                    # Create a new initial design
                    X_init = latin_hypercube(self.n_init, self.dim)
                    X_init = from_unit_cube(X_init, self.lb, self.ub)
                    fX_init = np.array([[self.f(x)] for x in X_init])

                    # Print progress
                    if self.verbose:
                        n_evals, fbest = self.n_evals, fX_init.min()
                        print(f"{n_evals}) TR-{i} is restarting from: {fbest:.4f}")
                        sys.stdout.flush()

                    # Append data to local history
                    self.X = np.vstack((self.X, X_init))
                    self.fX = np.vstack((self.fX, fX_init))
                    self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
                    self.n_evals += self.n_init

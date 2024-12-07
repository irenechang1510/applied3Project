import math
import sys
from copy import deepcopy

import gpytorch
import torch

from turbo.gp import train_gp  # Ensure this module is available
from turbo import Turbo1  # Ensure this module is available
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube
from turbo.utility import ExpectedImprovementCustom
from botorch.acquisition import ExpectedImprovement

class TurboM(Turbo1):
    """The TuRBO-M algorithm with a customizable acquisition function using torch Tensors."""

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
        X_val=None,
        fX_val=None
    ):
        self.n_trust_regions = n_trust_regions
        self.X_init = X_init  # Expecting already a torch tensor
        self.fX_init_original = fX_init  # Expecting torch tensor
        self.fX_init = self.fX_init_original.mean(dim=1, keepdim=True) 
        self.X_val = X_val  # Expecting torch tensor
        self.fX_val = fX_val  # Expecting torch tensor

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

        assert n_trust_regions > 1 and isinstance(max_evals, int)
        assert max_evals > n_trust_regions * n_init, "Not enough evaluations for initial points"
        assert max_evals > batch_size, "Not enough evaluations to do a single batch"

        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Override self.X, self.fX, and self._idx to be torch tensors
        # Turbo1 may have them as np arrays initially, so we set them here:
        self.X = torch.zeros(0, self.dim, dtype=torch.float64, device=self.device)
        self.fX = torch.zeros(0, 1, dtype=torch.float64, device=self.device)
        self._idx = torch.zeros(0, 1, dtype=torch.int64, device=self.device)

        self._restart()

    def _restart(self):
        self._idx = torch.zeros(0, 1, dtype=torch.int64, device=self.device)
        self.failcount = torch.zeros(self.n_trust_regions, dtype=torch.int64, device=self.device)
        self.succcount = torch.zeros(self.n_trust_regions, dtype=torch.int64, device=self.device)
        self.length = torch.ones(self.n_trust_regions, dtype=torch.float64, device=self.device) * self.length_init

    def _adjust_length(self, fX_next, i):
        assert 0 <= i <= self.n_trust_regions - 1
        # fX_next and self.fX are torch tensors
        mask = (self._idx[:, 0] == i)
        fX_min = self.fX[mask, 0].min()
        if fX_next.min() < fX_min - 1e-3 * abs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += fX_next.shape[0]  # Add batch size for this trust region

        if self.succcount[i] == self.succtol:
            self.length[i] = min(2.0 * self.length[i], self.length_max)
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        # X_cand: shape (n_trust_regions, n_cand, dim), torch float tensor
        # y_cand: shape (n_trust_regions, n_cand), torch float tensor
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0
        assert torch.isfinite(y_cand).all()

        X_next = torch.zeros(self.batch_size, self.dim, dtype=X_cand.dtype, device=X_cand.device)
        idx_next = torch.zeros(self.batch_size, 1, dtype=torch.int64, device=X_cand.device)

        # Flatten y_cand to find global argmin
        for k in range(self.batch_size):
            # Find argmin
            idx_min = torch.argmin(y_cand)
            i = idx_min // y_cand.shape[1]
            j = idx_min % y_cand.shape[1]

            X_next[k, :] = X_cand[i, j, :]
            idx_next[k, 0] = i
            # Set this candidate's value to inf so it's not selected again
            y_cand[i, j] = float('inf')

        return X_next, idx_next

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        # X, fX are np arrays in the original code.
        # Convert them to torch:
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        if not torch.is_tensor(fX):
            fX = torch.tensor(fX, dtype=self.dtype, device=self.device)

        y_mean = fX.mean()
        y_std = fX.std()
        fX_normalized = (fX - y_mean) / y_std

        # Squeeze to ensure 1D shape
        fX_normalized = fX_normalized.squeeze(-1)

        print(fX_normalized.shape)
        if hypers is not None and hypers != {}:
            model = train_gp(
                train_x=X,
                train_y=fX_normalized,
                use_ard=self.use_ard,
                num_steps=0,
                hypers=hypers,
            )
        else:
            model = train_gp(
                train_x=X,
                train_y=fX_normalized,
                use_ard=self.use_ard,
                num_steps=n_training_steps,
            )
            hypers = model.state_dict()

        best_f = fX_normalized.min()
        acquisition_function = self.utility_function_class(
            model,
            best_f=best_f,
            init_set=(self.X_init, self.fX_init_original),
            val_set=(self.X_val, self.fX_val)
        )

        X_cand = self.optimize_acquisition_function(acquisition_function, length)
        y_cand = acquisition_function(X_cand)

        return X_cand, y_cand, hypers

    def optimize_acquisition_function(self, acquisition_function, length):
        # Ensure self.center is a torch tensor
        if not isinstance(self.center, torch.Tensor):
            self.center = torch.tensor(self.center, dtype=self.dtype, device=self.device)

        # Define trust region bounds
        tr_lb = torch.clamp(self.center - length / 2.0, min=0.0, max=1.0)
        tr_ub = torch.clamp(self.center + length / 2.0, min=0.0, max=1.0)

        # Number of random samples for candidate search (simple random search)
        n_samples = 10
        X_candidates = torch.rand(n_samples, self.dim, dtype=self.dtype, device=self.device)
        X_candidates = tr_lb + (tr_ub - tr_lb) * X_candidates

        with torch.no_grad():
            acq_values = acquisition_function(X_candidates)

        best_idx = torch.argmax(acq_values)
        X_best = X_candidates[best_idx].unsqueeze(0)  # (1, dim)
        return X_best

    def optimize(self):
        # Convert initial arrays if necessary
        # We assume self.X_init, self.fX_init_original, self.X_val, self.fX_val are already torch Tensors
        # If they are not, convert them here.

        # Initialization from provided data
        if self.X_init is not None and self.fX_init is not None:
            assert self.X_init.shape[0] == self.fX_init.shape[0], "Mismatch in initial data sizes"
            assert self.X_init.shape[1] == self.dim, "Dimension mismatch in X_init"
            n_init_total = self.X_init.shape[0]
            n_init_per_tr = n_init_total // self.n_trust_regions

            for i in range(self.n_trust_regions):
                idx_start = i * n_init_per_tr
                idx_end = idx_start + n_init_per_tr
                X_init_i = self.X_init[idx_start:idx_end, :]
                fX_init_i = self.fX_init[idx_start:idx_end, :]

                # Update budget and set as initial data for this trust region
                self.X = torch.cat((self.X, X_init_i), dim=0)
                self.fX = torch.cat((self.fX, fX_init_i), dim=0)
                new_idx = torch.full((n_init_per_tr, 1), i, dtype=torch.int64, device=self.device)
                self._idx = torch.cat((self._idx, new_idx), dim=0)
                self.n_evals += n_init_per_tr

                if self.verbose:
                    fbest = fX_init_i.min().item()
                    print(f"TR-{i} starting from provided data: {fbest:.4f}")
                    sys.stdout.flush()
        else:
            # Create initial points using LHS
            for i in range(self.n_trust_regions):
                X_init_np = latin_hypercube(self.n_init, self.dim)  # np array
                X_init_np = from_unit_cube(X_init_np, self.lb, self.ub)  # still np
                X_init = torch.tensor(X_init_np, dtype=self.dtype, device=self.device)
                # Evaluate f for these points:
                f_values = []
                for x_ in X_init:
                    # x_cpu = x_.cpu().numpy()
                    x_tensor = torch.tensor(x_).float()
                    val = self.f(x_tensor)
                    f_values.append([val])
                fX_init = torch.tensor(f_values, dtype=self.dtype, device=self.device)

                # Update budget and set as initial data
                self.X = torch.cat((self.X, X_init), dim=0)
                self.fX = torch.cat((self.fX, fX_init), dim=0)
                new_idx = torch.full((self.n_init, 1), i, dtype=torch.int64, device=self.device)
                self._idx = torch.cat((self._idx, new_idx), dim=0)
                self.n_evals += self.n_init

                if self.verbose:
                    fbest = fX_init.min().item()
                    print(f"TR-{i} starting from: {fbest:.4f}")
                    sys.stdout.flush()

        # Main optimization loop
        while self.n_evals < self.max_evals:
            X_cand = torch.zeros(self.n_trust_regions, self.n_cand, self.dim, dtype=self.dtype, device=self.device)
            y_cand = torch.full((self.n_trust_regions, self.n_cand), float('inf'), dtype=self.dtype, device=self.device)

            for i in range(self.n_trust_regions):
                idx_i = (self._idx[:, 0] == i).nonzero(as_tuple=True)[0]
                X_i = self.X[idx_i, :]
                fX_i = self.fX[idx_i, 0]
                X_i_unit = to_unit_cube(X_i.cpu().numpy(), self.lb, self.ub)
                X_i_unit = torch.tensor(X_i_unit, dtype=self.dtype, device=self.device)

                # Determine training steps
                n_training_steps = 0 if self.hypers[i] else self.n_training_steps

                # Update center
                argmin_idx = torch.argmin(fX_i)
                self.center = X_i_unit[argmin_idx, :]

                # Create new candidates
                Xnew_cand, ynew_cand, self.hypers[i] = self._create_candidates(
                    X_i_unit, fX_i.unsqueeze(1), length=self.length[i],
                    n_training_steps=n_training_steps, hypers=self.hypers[i]
                )

                # Xnew_cand, ynew_cand returned by _create_candidates are tensors
                # Ensure shape matches X_cand and y_cand
                X_cand[i, :, :] = Xnew_cand
                y_cand[i, :] = ynew_cand

            # Select the next candidates
            X_next, idx_next = self._select_candidates(X_cand, y_cand)
            assert (X_next.min() >= 0.0) and (X_next.max() <= 1.0)

            # Undo the warping
            X_next_np = from_unit_cube(X_next.cpu().numpy(), self.lb, self.ub)
            X_next = torch.tensor(X_next_np, dtype=self.dtype, device=self.device)

            # Evaluate batch
            fX_next_list = []
            for x_ in X_next:
                x_tensor = torch.tensor(x_).float()
                val = self.f(x_tensor)
                fX_next_list.append([val])
            fX_next = torch.tensor(fX_next_list, dtype=self.dtype, device=self.device)

            # Update trust regions
            for i in range(self.n_trust_regions):
                mask_i = (idx_next[:, 0] == i)
                if torch.any(mask_i):
                    self.hypers[i] = {}
                    fX_i = fX_next[mask_i, :]
                    if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * abs(self.fX.min()):
                        n_evals, fbest = self.n_evals, fX_i.min().item()
                        print(f"{n_evals}) New best @ TR-{i}: {fbest:.4f}")
                        sys.stdout.flush()
                    self._adjust_length(fX_i, i)

            # Update budget and append data
            self.n_evals += self.batch_size
            self.X = torch.cat((self.X, X_next), dim=0)
            self.fX = torch.cat((self.fX, fX_next), dim=0)
            self._idx = torch.cat((self._idx, idx_next), dim=0)

            # Check if any trust region needs to be restarted
            for i in range(self.n_trust_regions):
                if self.length[i] < self.length_min:
                    idx_i = (self._idx[:, 0] == i).nonzero(as_tuple=True)[0]

                    if self.verbose:
                        fbest = self.fX[idx_i, 0].min().item()
                        n_evals = self.n_evals
                        print(f"{n_evals}) TR-{i} converged to: {fbest:.4f}")
                        sys.stdout.flush()

                    # Reset length and counters, remove old data from trust region
                    self.length[i] = self.length_init
                    self.succcount[i] = 0
                    self.failcount[i] = 0
                    self._idx[idx_i, 0] = -1
                    self.hypers[i] = {}

                    # Create a new initial design
                    X_init_np = latin_hypercube(self.n_init, self.dim)
                    X_init_np = from_unit_cube(X_init_np, self.lb, self.ub)
                    X_init = torch.tensor(X_init_np, dtype=self.dtype, device=self.device)
                    fX_init_list = []
                    for x_ in X_init:
                        x_tensor = torch.tensor(x_).float()
                        val = self.f(x_tensor)
                        fX_init_list.append([val])
                    fX_init = torch.tensor(fX_init_list, dtype=self.dtype, device=self.device)

                    if self.verbose:
                        n_evals, fbest = self.n_evals, fX_init.min().item()
                        print(f"{n_evals}) TR-{i} is restarting from: {fbest:.4f}")
                        sys.stdout.flush()

                    # Append data to local history
                    self.X = torch.cat((self.X, X_init), dim=0)
                    self.fX = torch.cat((self.fX, fX_init), dim=0)
                    new_idx = torch.full((self.n_init, 1), i, dtype=torch.int64, device=self.device)
                    self._idx = torch.cat((self._idx, new_idx), dim=0)
                    self.n_evals += self.n_init

import jax
import jax.numpy as jnp
from typing import Union
from tqdm import trange
from ml_models.BayesianOptimization import (
    Prior, Acquisition,
    matern_kernel, rbf_kernel
)


class GPR:
    """
    Basic Gaussian Process Regression
    """
    def __init__(
        self, prior_distribution: Prior,
        acquisition: Acquisition,
        mean_kernel,
        cov_kernel: Union[matern_kernel, rbf_kernel],
        n_restarts: int = 5,
        alpha: float = 1e-10
    ):
        self.prior = prior_distribution
        self.acquisition = acquisition
        self.mean = mean_kernel
        self.cov = cov_kernel
        self.alpha = alpha
        self.n_restarts = n_restarts

    def gaussian_process_reset(self):
        pass

    def acquisition(self, x_prime: jnp.ndarray, **kwargs):
        # TODO: have optimization with gs or lbfgs
        pred = self.acquisition.acquisition()
        best_pred = jnp.argmin(pred)
        return x_prime[best_pred:best_pred + 1, :]

    def find_next_sample(self):
        pass

    def posterior(self, X, x_prime, **kwargs):
        inv_cov_prior = jnp.invert(self.cov(X, X, **kwargs))
        cov_update = self.cov(x_prime, X, **kwargs)
        mu = cov_update * inv_cov_prior * X
        sigma = self.cov(x_prime, x_prime, **kwargs) - cov_update * inv_cov_prior * self.cov(X, x_prime, **kwargs)
        return mu, sigma

    def likelihood(self, X, params):
        pass

    def objective(self, X, params):
        return jax.value_and_grad(lambda x, p: self.likelihood(x, p))(X, params)

    def surrogate_prediction(self):
        pass

    def train(self, X, x_primes, n_train):
        # NOTE: Gaussian Process Regression requires its own trainer
        #       and is not compatible with the general purpose
        #       Trainer class
        for i in trange(n_train, desc="Gaussian Process Regression"):
            # TODO: update posterior using all available data
            next_sample = []
            self.acquisition.optimize()

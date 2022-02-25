import jax
from jax.scipy.linalg import solve, cholesky, cho_solve
import jax.numpy as jnp
from typing import Union
from tqdm import trange
from ml_models.BayesianOptimization import (
    Prior, Acquisition,
    matern_kernel, rbf_kernel,
    matern_grad_kernel, rbf_grad_kernel
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
        cov_gradient_kernel: Union[matern_grad_kernel, rbf_grad_kernel],
        n_restarts: int = 5,
        alpha: float = 1e-10
    ):
        self.prior = prior_distribution
        self.acquisition = acquisition
        self.mean = mean_kernel
        self.cov = cov_kernel
        self.cov_gradient = cov_gradient_kernel
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

    def posterior(self, X, Y, x_prime, noise: float = 0, **kwargs):
        r"""
        Making new predictions on x' based on the prior previously computed on X by using
        the posterior :math:`p(y'|Y,X,x')`. Since the covariance matricex of X and x' are
        independent, we can find the posterior covariance as

        ..math::
            \Sigma_n = \Sigma(x',x') - \Sigma(x',X)\Sigma(X,X)^{-1}\Sigma(X,x')

        In the case of noise, :math:`\Sigma(X,X)` is computed as

        ..math::
            K(X,X) + \sigma_{\epsilon}^2 I

        and the mean as

        ..math::
            \mu_n = \Sigma(x',X)\Sigma(X,X)^{-1}Y

        Therefore the posterior is defined as

        ..math::
            p(y'|Y,X,x') = \mathcal{N}(\mu_n, \Sigma_n)



        :param X:
        :param Y:
        :param x_prime:
        :param noise:
        :param kwargs:
        :return:
        """
        cov_prior = self.cov(X, X, **kwargs) + (noise**2 * jnp.eye(X.shape[0]))
        cov_update = self.cov(X, x_prime, **kwargs)
        solved = solve(cov_prior, cov_update, assume_a="pos").T
        mu = jnp.dot(solved, Y)
        sigma = self.cov(x_prime, x_prime, **kwargs) - (jnp.dot(solved, cov_update))
        return mu, sigma

    def log_likelihood(self, X, params, include_gradient: bool = False):
        """
        Marginal log_likelihood of the gaussian process with given parameters :math:`\Theta`
        defined as

        ..math::
            log p(y|X,\Theta) = -\frac{1}{2} (y - \mu_{\Theta})^T\Sigma_{\Theta}^{-1}(y - \mu_{\Theta} - \frac{1}{2} log |\Sigma_{\Theta}| - \frac{d}{2}log2\pi

        This implementation is based on the corresponding scikit-learn implementation, which
        uses cholesky decomposition

        :param X:
        :param params:
        :param include_gradient
        :return:
        """
        if include_gradient:
            k, k_grad = self.cov_gradient(X)
        else:
            k = self.cov(X)
        # NOTE: X is assumed to be 2D => add a check
        k += params['sigma_square'] * jnp.eye(*X.shape)
        l = cholesky(k, lower=True)
        alpha = cho_solve((l, True), params['y'])
        ll = -.5 * jnp.einsum("ik,ik->k", params['y'], alpha) - jnp.log(jnp.diag(l)).sum()
        ll = (ll - (k.shape[0] / 2 * jnp.log(2 * jnp.pi))).sum(axis=-1)
        if include_gradient:
            inner = jnp.einsum("ik,ik->k", alpha, alpha) - cho_solve((l, True), jnp.eye(k.shape[0]))[..., jnp.newaxis]
            ll_grad = (.5 * jnp.einsum("ijl,jik->kl", inner, k_grad)).sum(axis=-1)
            return ll, ll_grad
        return ll

    def objective(self, X, params):
        return self.log_likelihood(X, params, include_gradient=True)

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

"""
These implementations are based on https://arxiv.org/pdf/1807.02811.pdf (Frazier 2018)
"""
from jax import vmap, jit
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.optimize import minimize


class Acquisition:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def acquisition(*args, **kwargs):
        pass

    @staticmethod
    def optimize(*args, **kwargs):
        pass


class ExpectedImprovement(Acquisition):
    def __init__(self):
        super(ExpectedImprovement, self).__init__()

    @staticmethod
    @jit
    @vmap(in_axes=-1)
    def acquisition(mean, sd, obs):
        """
        Expected improvement described by Jones et al. (1998) and Clark (1961).

        :param mean:
        :param sd:
        :param obs:
        :return:
        """
        delta_n = -(mean - obs)
        delta_clipped = jnp.clip(delta_n, a_min=0)
        delta_scaled = delta_n / sd
        return delta_clipped + sd * norm.cdf(delta_scaled) - abs(delta_n) * norm.pdf(delta_scaled)

    def optimize(self, X, mean, sd, obs, **kwargs):
        # default optimizer is the quasi-Newton method mentioned by Frazier 2018
        optimizer = minimize(
            lambda x: self.acquisition(mean, sd, x),
            X, method=kwargs.pop('method', 'L-BFGS-B')
        )
        return optimizer.x, optimizer.jac, optimizer.fun


class KnowledgeGradient(Acquisition):
    pass


class EntropySearch(Acquisition):
    pass

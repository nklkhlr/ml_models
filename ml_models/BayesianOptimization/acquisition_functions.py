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
    def acquisition(x, mean, sd):
        r"""
        Expected improvement described by Jones et al. (1998) and Clark (1961) and
        defined as:

        ..math::
            EI(x) \coloneqq \left[\Delta_n(x)\right]^+ + \sigma_n(x)\phi\left(\frac{\Delta_n(x)}{\sigma_n(x)}\right) - \left|\Delta_n(x)\right| \Phi\left(\frac{\Delta_n(x)}{\sigma_n(x)}\right)

        where :math:`\phi` is the cumulative density function and :math:`\Phi` is the probability density function
        and :math:`\Delta_n(x) \coloneqq \mu_n(x) - f^*_n`

        :param x:
        :param mean:
        :param sd:
        :return:
        """
        delta_n = -(mean - x)
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

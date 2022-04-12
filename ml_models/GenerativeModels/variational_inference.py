r"""
Variational inference implementation.

VI casts inference as an optimization problem by approximating the true
(intractable) probability distribution :math:`p`. This is done by finding a :math:`q \in \mathcal{Q}`
that has the highest similarity to :math:`p`, measured by the Kullback-Leibler Divergence

.. math::
    KL(q||p) = \sum_{x} q(x) log \frac{q(x)}{p(x)}

Since this divergence contains the intractable distribution :math:`p`, the optimization
is not on :math:`KL(q||p)` itself, but the Evidence Lower Bound (ELBO).

The ELBO is defined as

    .. math::
        ELBO(q) \coloneqq \sum_x q(x) log \frac{q(x)}{\hat{p}(x)}

    It is a lower bound for log :math:`Z(\Theta)`, the normalization constant of a factor
    distribution :math:`p`, because

    .. math::
        ELBO(q) &= \sum_x q(x) log \frac{q(x)}{\tilde{p}(x)}
                &= \sum_x q(x) log \frac{q(x)}{p(x)} - log Z(\Theta)
                &= KL(q||p) - log Z(\Theta)

    with :math:`\tilde{p}` defined as :math:`\prod_k\theta_k(x_k, \Theta)` (i.e. the
    unnormalized probability)

    Therefore :math:`log Z(\Theta) = KL(q||p) - ELBO(q) \geq -ELBO(q)`, because
    the Kullback-Leibler divergence is always :math:`\geq 0`. Maximizing ELBO(q) means
    minimizing KL(q||p), hence minimizing the difference between the approximate probability
    distribution :math:`q \in \mathcal{Q}` and the true (intractable) probability distribution
    :math:`p`

"""
from typing import Dict, Iterable
from functools import partial
import jax
import jax.numpy as jnp


# ================== #
# Sampling Functions #
# ================== #
# Diagonal Multivariate Normal
@jax.jit
def gaussian_init(shape, rng, mean_sd, sigma):
    mu = jax.random.normal(rng, shape) * mean_sd
    cov = jnp.ones(shape) * sigma
    return mu, cov


@jax.jit
def gaussian_kl(mu, sigma_square):
    return -.5 * jnp.sum(1 + jnp.log(sigma_square) - (mu**2 + sigma_square))


@jax.jit
def gaussian_logpdf(x, mu, sigma_square):
    x_hat = x - mu
    jnp.einsum('...i,i->...i', x_hat, 1. / jnp.log(sigma_square))


@jax.jit
def gaussian_sample(rng, mean, var):
    return mean + jnp.exp(.5 * jnp.log(var)) * jax.random.normal(rng, mean.shape)


# TODO: poisson implementation


# =========================== #
# Variational Inference Steps #
# =========================== #
@partial(jax.jit, static_argnums=(3,))
def single_elbo(sample_fun, log_p_fun, logprob_fun, rng, params):
    r"""
    Lower bound estimate (Monte-Carlo) for a single sample.

    Parameters
    ----------
    sample_fun: function
        Sampling function for the chosen distribution

    log_p_fun : function
        Log-probability function, either logpdf or logpmf function for the chosen distribution

    logprob_fun: function
        Log-probability function of the intractable probability distribution (not normalized).
        This function needs to take **exactly one** argument (the MC sample)

    rng: jax.PRNGKeyArray
        Random seed for sampling function

    params: Dict[str, Iterable[any]]
        Dictionary containing the parameters for the sample and the lodpdf/logpmf function.

        The structure needs to be:

        'sample_params'
        All arguments that need to passed to sample_fun in the correct order, **except** for the
        first position, which needs to be the random key

        'log_p_params'
        All arguments that need to passed to log_p_fun in the correct order, **except** for the
        first position, which needs to be the MC sample

    Returns
    -------
    float
        Lower bound estimate for a single MC sample
    """
    mc_sample = sample_fun(rng, *params['sample_params'])
    log_prob = logprob_fun(mc_sample)
    dist_p = log_p_fun(mc_sample, *params['log_p_params'])
    # variational lower bound => difference instead of ratio (log-space)
    return log_prob - dist_p


def vectorized_elbo(sample_fun, log_p_fun, logprop_fun):
    return jax.vmap(partial(single_elbo, sample_fun, log_p_fun, logprop_fun), in_axes=(0, None, None))


def elbo(sample_fun, log_p_fun, logprob_fun, rng, params):
    # random.split to get really random numbers
    random_keys = jax.random.split(rng)
    elbo_estimates = vectorized_elbo(sample_fun, log_p_fun, logprob_fun)(random_keys, params)
    # average elbo estimate of random MC samples
    return jnp.mean(elbo_estimates)


def coordinate_ascent():
    """
    Coordinate ascent for mean field approximation

    Returns
    -------

    """
    pass


def mean_field_approximation():
    r"""
    Computing the optimal parameters assuming independent latent variables
    (i.e. fully-factoring/individual distributions). The probability is
    therefore defined as :math:`q(z) = \prod_j q_j(z_j)`

    Returns
    -------
    """
    pass


if __name__ == "__main__":
    pass

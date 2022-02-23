"""
based on
* https://arxiv.org/pdf/1807.02811.pdf
* https://argmin.lis.tu-berlin.de/papers/18-vien-AAAI.pdf
"""
from jax import jit, vmap
import jax.numpy as jnp
# TODO: are there jax implementations of the gamma and modified bessel function?
from scipy.special import gamma, iv


@jit
@vmap
def gaussian_kernel(
    x: jnp.ndarray, x_prime: jnp.ndarray,
    alpha_0: float, alpha: jnp.ndarray
):
    r"""
    Guassian kernel defined as

    ..math::
        \Sigma(x,x') \coloneqq \alpha_0 exp\left(-||x - x'||^2\right)

    where :math:`||x - x'||^2 = \sum_{i=1}^d \alpha_i (x_i - x'_i)^2`

    :param x:
    :param x_prime:
    :param alpha_0:
    :param alpha:
    :return:
    """
    norm = (x - x_prime) ** 2
    return alpha_0 * jnp.exp(-(alpha * norm))


@vmap
def matern_kernel(
    x: jnp.ndarray, x_prime: jnp.ndarray,
    alpha_0: float, v: float
):
    r"""
    Matern kernel defined as

    ..math::
        \Sigma(x,x') \coloneqq \alpha_0 \frac{2^{1 - v}{\Gamma(v} \left(\sqrt{2v}||x - x'||\right)^v K_\left(\sqrt{2v}||x - x'||\right)v

    :param x:
    :param x_prime:
    :param alpha_0:
    :param v:
    :return:
    """
    norm = jnp.abs(x - x_prime)
    norm *= jnp.sqrt(2 * v)
    w = alpha_0 * (2**(1 - v)) / gamma(v)
    return w * norm**v * iv(norm)


@jit
@vmap
def rbf_kernel(
    x: jnp.ndarray, x_prime: jnp.ndarray,
    sigma_square: float
):
    r"""
    Radial base function kernel defined as

    ..math::
        K(x,x') \coloneqq exp\left(-\frac{||x - x'||^2}{2\sigma^2}\right)

    :param x:
    :param x_prime:
    :param sigma_square:
    :return:
    """
    # TODO: correct norm
    norm = (x - x_prime)**2
    return jnp.exp(-norm / (2 * sigma_square))

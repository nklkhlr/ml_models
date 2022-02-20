import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


@jax.jit
def mse(y_hat: jnp.ndarray, y: jnp.ndarray) -> float:
    r"""
    Mean squared error:
    :param y_hat: predicted outcome
    :param y: observed outcome
    :return:

    >>> x = jax.random.normal(jax.random.PRNGKey(42), (10, 15))
    >>> u = jax.random.normal(jax.random.PRNGKey(12), (10, 15))
    >>> print(mse(x, u))
    2.0604362
    """
    return jnp.mean((y - y_hat)**2)


@jax.jit
def mae(y_hat: jnp.ndarray, y: jnp.ndarray) -> float:
    r"""
    Mean squared error:
    :param y_hat: predicted outcome
    :param y: observed outcome
    :return:

    >>> x = jnp.array([0.11093666, 0.25388166, 2.45402365, 0.22552281, 0.64578421])
    >>> u = jnp.array([0.62389336, 0.20704534, 0.07012288, 0.59604209, 0.13083742])
    >>> print(mae(x, u))
    0.76583195
    """
    return jnp.mean(abs(y - y_hat))


@jax.jit
def mbe(y_hat: jnp.ndarray, y: jnp.ndarray) -> float:
    r"""
    Mean squared error:
    :param y_hat: predicted outcome
    :param y: observed outcome
    :return:

    >>> x = jnp.array([-0.11093666, -0.25388166,  2.45402365,  0.22552281,  0.64578421])
    >>> u = jnp.array([-0.62389336, -0.20704534, -0.07012288,  0.59604209,  0.13083742])
    >>> print(mbe(x, u))
    -0.62693894
    """
    return jnp.mean(y - y_hat)


@jax.jit
def cross_entropy(y_hat: jnp.ndarray, y: jnp.ndarray) -> float:
    r"""
    Negative Log-Likelihood/Cross-entropy loss
    :param y_hat: predicted outcome
    :param y: observed outcome
    :return:

    >>> x = jnp.array([0.25, 0.75, 0.45, .9])
    >>> u = jnp.array([0, 1, 0, 1])
    >>> print(cross_entropy(x, u))
    0.3196404
    """
    return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))


@jax.jit
def nll(y: jnp.ndarray, y_hat: jnp.ndarray) -> float:
    """

    :param y:
    :param y_hat:
    :return:

    >>>
    """
    probabilities = y * y_hat + (1 - y) * (1 - y_hat)
    return -jnp.sum(jnp.log(probabilities))


@jax.jit
def normal_nll(x, *args, y=None, **kwargs):
    return -norm.logpdf(x, *args, **kwargs).mean()


# TODO: move this to an own activation function file
@jax.jit
def relu(x):
    return jnp.nanmax(0, x)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

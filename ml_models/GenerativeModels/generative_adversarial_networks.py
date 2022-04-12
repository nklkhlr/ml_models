from typing import Tuple, Callable
from functools import partial
import jax
import jax.numpy as jnp
from typing import NamedTuple
from .variational_autencoder import generate_batches


class GAN(NamedTuple):
    """
    Class holding a Generative Adversarial Model
    """
    discriminator: Callable
    generator: Callable


@partial(jax.jit, static_argnums=(2,))
def discriminator_cross_entropy(
    X: jnp.ndarray, Z: jnp.ndarray,
    model: GAN, parameters: tuple
) -> float:
    r"""
    Discriminator loss for GAN-training

    The loss is defined as (see Goodfellow et al. 2014)

    .. math::
        \frac{1}{m} \sum_{i=1}^m \left[log D(x^i) + log \left(1 - D\left(G\left(z^i\right)\right)\right)\right]

    where :math:`D` is the discriminator model, :math:`G` is the generator model, :math:`x^i` is a sample from the
    data generating distribution (:math:`p_{data}(x)`) and :math:`z^i` is a sample from the noise prior (:math:`p_g(z)`)

    Parameters
    ----------
    X: jnp.ndarray
        Samples drawn from the data generating distribution (i.e. training samples)
    Z: jnp.ndarray
        Samples drawn from the noise prior distribution
    model: GAN
        GAN model
    parameters: 2-tuple
        Parameters for discriminator and generator model

    Returns
    -------
    float
        Mean cross-entropy loss for the current model and batch
    """
    disc_pred = model.discriminator(X, parameters[0])
    gen_pred = model.discriminator(model.generator(Z, parameters[1]), parameters[0])
    return jnp.mean(jnp.log(disc_pred) + jnp.log(1 - gen_pred))


@partial(jax.jit, static_argnums=(1,))
def generator_cross_entropy(Z: jnp.ndarray, model: GAN, parameters: tuple):
    r"""
    Generator loss for GAN-training

    The loss is defined as (see Goodfellow et al. 2014)

    .. math::
        \frac{1}{m} \sum_{i=1}^m log \left(1 - D\left(G\left(z^i\right)\right)\right)

    where :math:`D` is the discriminator model, :math:`G` is the generator model and :math:`z^i`
    is a sample from the noise prior (:math:`p_g(z)`)

    Parameters
    ----------
    Z: jnp.ndarray
        Samples drawn from the noise prior distribution
    model: GAN
        GAN model
    parameters: 2-tuple
        Parameters for discriminator and generator model

    Returns
    -------
    float
        Mean loss for the current model and batch
    """
    return jnp.mean(1 - model.discriminator(model.generator(Z, parameters[1]), parameters[0]))


def random_input(rng: jax.random.PRNGKey, X: jnp.ndarray):
    pass


def gan_train(
    rng: int, X: jnp.ndarray,
    init: Tuple[Callable, Callable], model: GAN,
    optimizer: Callable, loss: Callable, n_epochs: int, batch_size: int,
    optimizer_args: dict = None, model_args: Tuple[dict, dict] = None,
    model_is_batched: bool = False
):
    discriminator_grad = jax.jit(
        jax.grad(
            lambda x, z, m, theta_d, theta_g: discriminator_cross_entropy(x, z, m, (theta_d, theta_g))
        )
    )
    generator_grad = jax.jit(
        jax.value_and_grad(
            lambda z, m, theta_d, theta_g: generator_cross_entropy(z, m, (theta_d, theta_g))
        )
    )
    if model_args is None:
        disc_params, gen_params = init[0](), init[1]()
    else:
        disc_params, gen_params = init[0](**model_args[0]), init[1](**model_args[1])
    opt_init, opt_update, opt_params = optimizer(**optimizer_args)
    opt_state = opt_init((disc_params, gen_params))
    disc_params, gen_params = opt_params(opt_state)
    rand_key = jax.random.PRNGKey(rng)
    disc_loss = []
    gen_loss = []
    disc_null_grad = jnp.zeros(disc_params.shape)
    gen_null_grad = jnp.zeros(gen_params.shape)
    for epoch in range(n_epochs):
        batches = generate_batches(rand_key, X.shape[0], batch_size)
        for i, batch in enumerate(batches):
            Z = None
            # TODO: add in option for k
            disc_grad, _ = discriminator_grad(X[batch, :], Z, model, disc_params, gen_params)
            opt_state = opt_update(i, (disc_grad, gen_null_grad), opt_state)
            disc_params, _ = opt_update(opt_state)

            gen_grad, _ = generator_grad(X[batch, :], model, disc_params, gen_params)
            opt_state = opt_update(i, (gen_grad, disc_null_grad), opt_state)
            gen_params = opt_params(opt_state)
    return disc_params, gen_params, disc_loss, gen_loss


def wgan_train(
    rng: int, X: jnp.ndarray,
    init: Tuple[Callable, Callable], model: GAN,
    optimizer: Callable, loss: Callable, n_epochs: int, batch_size: int,
    optimizer_args: dict = None, model_args: Tuple[dict, dict] = None,
    model_is_batched: bool = False
):
    pass


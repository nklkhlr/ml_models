from typing import Tuple, Callable
import jax
import jax.numpy as jnp


def generate_batches(rng, n_samples: int, batch_size: int) -> list:
    x_shuff = jax.random.shuffle(rng, jnp.arange(n_samples))
    n_steps = x_shuff.size // batch_size
    return [
        x_shuff[(batch_size * i):(batch_size * (i + 1))]
        if batch_size * (i + 1) < n_steps
        else x_shuff[(batch_size * i):]
        for i in range(n_steps)
    ]


@jax.jit
def normal_vae_loss(
    rng: jax.random.PRNGKey, x: jnp.array, latent_size: int,
    encoder, decoder, encoder_params, decoder_params,
    epsilon: float = 1e-10
) -> jnp.ndarray:
    mu, sigma = encoder(x, encoder_params)
    # reparameterization trick
    z_sample = jax.random.normal(rng, shape=(latent_size,)) * mu + sigma
    # reconstruction loss
    recon_loss = -jnp.log(decoder(z_sample, decoder_params).dot(x.T) + epsilon)
    # kullback-leibler loss
    # TODO: this only works with standard normal => adapt for other (dynamic)
    kl_loss = jnp.mean(-jnp.log(sigma) + (sigma**2 / 2) + (mu**2 / 2) - .5)
    # has to be a jnp.ndarray to work with jax.grad
    return jnp.array([recon_loss, kl_loss])


def vae_train(
    rng: int, X: jnp.ndarray,
    init: Tuple[Callable, Callable], model: Tuple[Callable, Callable], latent_size: int,
    optimizer: Callable, loss: Callable, n_epochs: int, batch_size: int,
    optimizer_args: dict = None, model_args: Tuple[dict, dict] = None,
    model_is_batched: bool = False
):
    """
    Training a variational encoder

    Parameters
    ----------
    rng: int
        random seed
    X: jnp.ndarray
        Training data

    init: Tuple[Callable, Callable]
        Parameter initialization functions for encoder and decoder

    model: Tuple[Callable, Callable]
        Encoder and decoder function

    latent_size: int
        Size of the latent layer

    optimizer: Callable
        Optimizer function, should follow the same conventions as the
        optimizers from the optax package

    loss: Callable
        Loss function. Needs to have the same in- and output as :py:func:`normal_vae_loss`

    n_epochs: int
        Number of epochs to train

    batch_size: int
        Number of batches per epoch

    optimizer_args: dict, Optional
        Keyword arguments to pass to the optimizer

    model_args: Tuple[dict, dict], Optional
        Keyword arguments to pass to the encoder and decoder (at initialization)

    model_is_batched: bool, default False
        Whether encoder and decoder functions are already taking entire batches (True)
        or only single samples (False)

    Returns
    -------
    Tuple[any, any, list, list]

        1. Trained encoder parameters
        2. Trained decoder parameters
        3. reconstruction loss term per batch
        4. KL loss per batch

        Note that the sum of reconstruction loss and KL loss yield the ELBO
    """
    # single sample functions to batch functions + compilation
    if model_is_batched:
        batch_encoder = jax.jit(model[0])
        batch_decoder = jax.jit(model[1])
    else:
        batch_encoder = jax.jit(jax.vmap(model[0], in_axes=(0, None)))
        batch_decoder = jax.jit(jax.vmap(model[1], in_axes=(0, None)))
    batch_loss = jax.jit(jax.vmap(loss, in_axes=(None, 0, None, None, None, None, None)))
    jit_grad = jax.jit(
        jax.grad(
            lambda rkey, x, ep, dp: jnp.mean(batch_loss(rkey, x, latent_size, model[0], model[1], ep, dp))
        )
    )
    # initializing model parameters
    if model_args is None:
        encoder_params = init[0]()
        decoder_params = init[1]()
    else:
        encoder_params = init[0](**model_args[0])
        decoder_params = init[1](**model_args[1])
    # initializing optimizer
    opt_init, opt_update, opt_params = optimizer(**optimizer_args)
    opt_state = opt_init((encoder_params, decoder_params))
    # initializing
    grad_key = jax.random.PRNGKey(rng)
    recon_loss = []
    kl_loss = []
    # TODO: add in train vs. test loss
    for epoch in range(n_epochs):
        batches = generate_batches(grad_key, X.shape[0], batch_size)
        for i, batch in enumerate(batches):
            grad_key, loss_key = jax.random.split(grad_key)
            encoder_params, decoder_params = opt_params(opt_state)
            grads = jit_grad(grad_key, X[batch, :], encoder_params, decoder_params)
            opt_state = opt_update(i, grads, opt_state)
            mean_loss = jnp.mean(
                batch_loss(loss_key, X[batch, :], latent_size, batch_encoder, batch_decoder,
                           encoder_params, decoder_params),
                axis=0
            )
            recon_loss.append(mean_loss[0])
            kl_loss.append(mean_loss[1])
    return encoder_params, decoder_params, recon_loss, kl_loss

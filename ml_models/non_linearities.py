import jax
import jax.numpy as jnp


@jax.jit
def relu(x):
    return jnp.maximum(jnp.zeros(x.shape), x)

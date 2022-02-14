import jax
import jax.numpy as jnp
from typing import List


jax_seed = jax.random.PRNGKey(42)


class NormalizingFlowModel:
    r"""
    Base class for Normalizing Flow models.

    Estimating the output distribution as:
    p_2(x) = p_1(z) \left| \frac{\delta z}{\delta x} \right| = p_1(z) \left| \frac{\delta f^{-1}(x)}{\delta x} \right|

    Multiple Layers:
    p_K(x) = p_0(z_0) \prod_{i=1}^{K} \left| det\left(\frac{\delta f_i^{-1}(z_i)}{\delta z_i}\right) \right|
    in log space: log(p_K(x)) = log(p_0(z_0)) + \sum_{i=1}^{K} \left| det\left(\frac{\delta f_i^{-1}(z_i)}{\delta z_i}\right) \right|

    Transformation (f) should be __differentiable__ and __invertible__
    """
    def __init__(self, mode: str):
        self.mode = None
        self.change_mode(mode)

    def change_mode(self, mode):
        if mode != "forward" and mode != "reverse":
            raise ValueError(f"'mode' can only 'forward' or 'reverse'. {mode} is not a valid option")
        self.mode = mode
        self._reset_model_parameters_()

    def _reset_model_parameters_(self, *args, **kwargs):
        raise NotImplemented

    def forward_parametrization(self, x: jnp.ndarray):
        r"""
        Given a distribution sample points - i.e. map $z \rightarrow x$ - as

        p_X(x) = p_Z(f_{\theta}(x)) * \left|det\left(Jacobian\left(f_{\theta}(x)\right)\right)\right|^{-1}

        :return
        """
        raise NotImplemented

    def reverse_parametrization(self, zi: jnp.ndarray):
        r"""
        Given a distribution and transformation estimate the output density - i.e. map $x \rightarrow z$ - as

        p_X(x) = p_Z(g_{\phi}(x)) * \left|det\left(Jacobian\left(g_{\phi}(x)\right)\right)\right|

        :return transformed input z_{i-1} and det(grad(g(zi)))
        """
        raise NotImplemented

    def transformation(self, X, *args, **kwargs):
        """
        Forward transformation
        """
        raise NotImplemented

    def inverse_transformation(self, X, *args, **kwargs):
        """
        Reverse transformation
        """
        raise NotImplemented

    def train(self):
        raise NotImplemented

    # TODO: optimize jacfwd/jacinv usage
    def forward_jacobian_log_det(self, X, **kwargs):
        return -jnp.log(jnp.linalg.det(jax.jacfwd(self.transformation)(X, **kwargs)))

    def inverse_jacobian_log_det(self, X, **kwargs):
        return jnp.log(jnp.linalg.det(jax.jacfwd(self.transformation)(X, **kwargs)))


class PlanarFlow(NormalizingFlowModel):
    def __init__(self, input_dim, mode: str = "forward", non_linearity=jnp.tanh):
        super(PlanarFlow, self).__init__(mode)
        self._reset_model_parameters_(input_dim)
        self.non_linearity = non_linearity
        self.distribution = None

    def _reset_model_parameters_(self, input_dim):
        self.weights = jax.random.normal(jax_seed, (input_dim, 1))
        self.bias = jax.random.normal(jax_seed, (1, 1))
        self.u = jax.random.normal(jax_seed, (1, 1))

    def transformation(self, Z, *args, **kwargs):
        r"""
        Planar flow transformation defined as:

        f(z) = z + u \mathcal{h}(w^Tz + b)

        where h is as the tanh function by default, but can be change by setting self.non_linearity

        :param Z:
        :return:
        """
        return Z + self.u.dot(self.non_linearity(jnp.transpose(self.weights).dot(Z) + self.bias))

    def inverse_transformation(self, Z, *args, **kwargs):
        r"""
        Planar flow transformation defined as:

        g(z) = \mathcal{h}'(w^Tz + b)w

        where \mathcal{h}' is by default set as the derivative of the tanh function
        defined as 1 - tanh(x)^2 , but can be changed by setting self.non_linearity

        :param Z:
        :return:
        """
        return jax.grad(jnp.tanh)(jnp.transpose(self.weights).dot(Z) + self.bias).dot(self.weights)


class MAF(NormalizingFlowModel):
    def __init__(self, n_features):
        super(MAF, self).__init__()


class RealNVP(NormalizingFlowModel):
    def __init__(self):
        super(RealNVP, self).__init__()


class Spline(NormalizingFlowModel):
    def __init__(self):
        super(Spline, self).__init__()


class NormalizingFlows:
    """
    Wrapper to stack multiple normalizing flows
    """
    def __init__(self, models: List[NormalizingFlowModel]):
        self.layers = models

    def forward(self, X: jnp.ndarray):
        z = X.copy()
        for model in self.layers:
            z = model.transformation(z)
        return z

    def reverse(self, X: jnp.ndarray):
        z = X.copy()
        for model in self.layers:
            z = model.inverse_transformation(z)
        return z

    def backward(self, y: jnp.ndarray):
        # TODO: compute loss
        loss = 0
        for i in jnp.arange(len(self.layers) - 1, -1, -1):
            # TODO: compute gradient for each layer and pass to the next
            pass

    def train(self):
        # TODO
        pass


if __name__ == "__main__":
    pass

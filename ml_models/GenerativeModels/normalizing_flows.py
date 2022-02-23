import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, multivariate_normal
# TODO: replace optax with own optimizer and loss implementations
from typing import List
from ml_models.base import BaseModel
from ml_models.non_linearities import relu
from ml_models.NNLayers.mlp import RealNVPNN


jax_seed = jax.random.PRNGKey(42)


class NormalizingFlowModel:
    r"""
    Base class for Normalizing Flow models.

    Estimating the output distribution as:

    ..math::
    p_2(x) = p_1(z) \left| \frac{\delta z}{\delta x} \right| = p_1(z) \left| \frac{\delta f^{-1}(x)}{\delta x} \right|

    Multiple Layers:

    ..math::
    p_K(x) = p_0(z_0) \prod_{i=1}^{K} \left| det\left(\frac{\delta f_i^{-1}(z_i)}{\delta z_i}\right) \right|

    in log space:

    ..math::
    log(p_K(x)) = log(p_0(z_0)) + \sum_{i=1}^{K} \left| det\left(\frac{\delta f_i^{-1}(z_i)}{\delta z_i}\right) \right|

    The transformation function f needs to be __differentiable__ and __invertible__
    """
    def __init__(self, mode: str, **kwargs):
        self.mode = None
        self.change_mode(mode, **kwargs)

    def change_mode(self, mode, **kwargs):
        if mode != "forward" and mode != "reverse":
            raise ValueError(f"'mode' can only 'forward' or 'reverse'. {mode} is not a valid option")
        self.mode = mode
        self._reset_model_parameters_(**kwargs)

    @property
    def params(self) -> dict:
        raise NotImplemented

    def update_parameters(self, params: dict):
        raise NotImplemented

    def _reset_model_parameters_(self, *args, **kwargs):
        raise NotImplemented

    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplemented

    @staticmethod
    def inverse(*args, **kwargs):
        raise NotImplemented

    def neg_log_det_jacobian(self, *args, **kwargs):
        raise NotImplemented

    def log_det_jacobian(self, *args, **kwargs):
        raise NotImplemented

    @staticmethod
    def keep_invertible(params):
        raise NotImplemented


class PlanarFlow(NormalizingFlowModel):
    weights: jnp.ndarray
    bias: jnp.ndarray
    u: jnp.ndarray

    def __init__(self, input_dim, mode: str = "forward", non_linearity=jnp.tanh):
        super(PlanarFlow, self).__init__(mode, input_dim=input_dim)
        self._reset_model_parameters_(input_dim)
        self.non_linearity = non_linearity
        self.distribution = None

    def _reset_model_parameters_(self, input_dim):
        self.weights = jax.random.uniform(jax_seed, (input_dim,), minval=-1, maxval=1)
        self.bias = jax.random.uniform(jax_seed, (1,), minval=-1, maxval=1)[0]
        self.u = jax.random.uniform(jax_seed, (input_dim,), minval=-1, maxval=1)

    @property
    def params(self) -> dict:
        return {
            'weights': self.weights,
            'bias': self.bias,
            'u': self.u
        }

    def update_parameters(self, params: dict):
        self.weights = params['weights']
        self.bias = params['bias']
        self.u = params['u']

    @staticmethod
    def keep_invertible(params):
        wu_dot = jnp.tensordot(params['weights'], params['u'], 1)
        if wu_dot < -1:
            norm = params['weights'] / jnp.sqrt((params['weights']**2).sum())
            params['u'] += jnp.transpose((jax.nn.softplus(wu_dot) - 1 - wu_dot) * norm)
        return params

    @staticmethod
    @jax.jit
    def forward(params, X):
        lin_pass = jnp.tensordot(params['weights'], X, 1) + params['bias']
        return jnp.add(X, jnp.tensordot(jnp.tanh(lin_pass), params['u'], 0).T)

    @staticmethod
    @jax.jit
    def inverse(params, X):
        lin_pass = jnp.tensordot(params['weights'], X, 1) + params['bias']
        return X + jnp.tensordot(jnp.tanh(lin_pass), params['u'], 0)

    @staticmethod
    @jax.jit
    def neg_log_det_jacobian(params, X):
        hp = 1 - jnp.tanh(params['weights'].dot(X) + params['bias']) ** 2
        psi = 1 + jnp.tensordot(jnp.tensordot(hp, params['weights'], 0), params['u'], 1)
        return -jnp.log(jnp.abs(psi))

    @staticmethod
    @jax.jit
    def log_det_jacobian(params, X):
        hp = 1 - jnp.tanh(params['weights'].dot(X) + params['bias']) ** 2
        psi = 1 + jnp.tensordot(jnp.tensordot(hp, params['weights'], 0), params['u'], 1)
        return jnp.log(jnp.abs(psi))


class RealNVP(NormalizingFlowModel):
    def __init__(self, input_dim, hidden_sizes, mode: str = "forward", activation=relu):
        super(RealNVP, self).__init__(mode, input_dim=input_dim, hidden_sizes=hidden_sizes,
                                      activation=activation)
        if input_dim % 2 != 0:
            raise ValueError("")

    def _reset_model_parameters_(self, input_dim, hidden_sizes, activation):
        self.neural_net = RealNVPNN(input_dim // 2, hidden_sizes, activation)

    @property
    def params(self):
        return self.neural_net.parameters

    def update_parameters(self, params, *args, **kwargs):
        self.neural_net.update_parameters(params)

    @staticmethod
    @jax.jit
    def forward_log_shift(X, shift, log_scale):
        return (X + shift) * jnp.exp(log_scale)

    def forward(self, params, X):
        x1, x2 = jnp.split(X.T, 2, axis=-1)
        shift_, scale_ = self.neural_net.forward()(params, x2.T)
        y_ = self.forward_log_shift(x1.T, shift_, scale_)
        return jnp.concatenate([y_, x2.T], axis=-1)

    def forward_fun(self):
        nn = self.neural_net.forward()
        shift = self.forward_log_shift

        def _forward_fun(params, X):
            x1, x2 = jnp.split(X.T, 2, axis=-1)
            shift_, scale_ = nn(params, x2.T)
            y_ = shift(x1.T, shift_, scale_)
            return jnp.concatenate([y_, x2.T], axis=-1)

        return _forward_fun

    @staticmethod
    @jax.jit
    def inverse_log_shift(X, shift, log_scale):
        return (X - shift) * jnp.exp(-log_scale)

    def inverse(self, params, Y):
        y1, y2 = jnp.split(Y, 2, axis=-1)
        shift_, scale_ = self.neural_net.forward()(params, y2)
        x_ = self.inverse_log_shift(y1, shift_, scale_)
        return jnp.concatenate([x_, y2], axis=-1)

    def inverse_fun(self):
        nn = self.neural_net.forward()
        shift = self.inverse_log_shift

        def _inverse_fun(params, X):
            x1, x2 = jnp.split(X.T, 2, axis=-1)
            shift_, scale_ = nn(params, x2.T)
            y_ = shift(x1.T, shift_, scale_)
            return jnp.concatenate([y_, x2.T], axis=-1)

        return _inverse_fun

    @staticmethod
    def neg_log_det_jacobian(params, X, forward_fun):
        jacobian = jax.jacfwd(lambda x: forward_fun(params, x))
        jac_vmap = jax.vmap(jacobian)
        return -jnp.log(jnp.linalg.det(jac_vmap(X)))

    @staticmethod
    def log_det_jacobian(params, X, inverse_fun):
        jacobian = jax.jacfwd(lambda x: inverse_fun(params, x))(X)
        jac_vmap = jax.vmap(jacobian)
        return jnp.log(jnp.linalg.det(jac_vmap(X)))


class MAF(NormalizingFlowModel):
    # TODO
    def __init__(self, mode: str):
        super(MAF, self).__init__(mode)


class NormalizingFlow(BaseModel):
    """
    Wrapper to stack multiple normalizing flows
    """
    def __init__(self, mode: str, models: List[NormalizingFlowModel], distribution, sampling):
        super(NormalizingFlow, self).__init__()
        self.mode = mode
        self.layers = models
        self.distribution = distribution
        self.sampling = sampling
        self._transformation = self.forward_transformation if self.mode == "forward" else self.inverse_transformation
        # TODO: add logging in
        # self.logs = Logger()

    def change_mode(self, mode, **kwargs):
        if mode != "forward" and mode != "reverse":
            raise ValueError(f"'mode' can only 'forward' or 'reverse'. {mode} is not a valid option")
        self.mode = mode
        for layer in self.layers:
            layer.change_mode(mode, **kwargs)

    @property
    def parameters(self):
        return [model.params for model in self.layers]

    @staticmethod
    def forward_transformation(params, layers, X: jnp.ndarray):
        z = X.copy()
        distortion = 0
        for model, model_params in zip(layers, params):
            if isinstance(model, PlanarFlow):
                model_params = model.keep_invertible(model_params)
                distortion += model.neg_log_det_jacobian(model_params, z)
                z = model.forward(model_params, z)
            elif isinstance(model, RealNVP):
                distortion += model.neg_log_det_jacobian(model_params, z, model.forward_fun())
                z = model.forward(model_params, z)
        return z, distortion

    @staticmethod
    def inverse_transformation(params, layers, X: jnp.ndarray):
        z = X.copy()
        distortion = 0
        for model, model_params in zip(layers, params):
            if isinstance(model, PlanarFlow):
                model_params = model.keep_invertible(model_params)
                distortion += model.log_det_jacobian(model_params, z)
                z = model.inverse(model_params, z)
            elif isinstance(model, RealNVP):
                distortion += model.log_det_jacobian(model_params, z, model.inverse_fun())
                z = model.inverse(model_params, z)
        return z, distortion

    def train(self):
        if self.mode == 'forward':
            return NormalizingFlow.forward_transformation
        else:
            return NormalizingFlow.inverse_transformation

    def compute_loss(self, loss_fun):
        layers = self.layers
        if self.mode == "forward":
            dist = self.distribution
            transformation = NormalizingFlow.forward_transformation

            def _loss(params, x, *args, **kwargs):
                z, distortion = transformation(params, layers, x)
                return loss_fun(dist(z) + distortion)
        else:
            sample = self.sampling
            transformation = NormalizingFlow.inverse_transformation

            def _loss(params, x, *args, **kwargs):
                # TODO: turn this into a sampling function
                # should e.g. be jaz.random.normal(key, (n_samples, dim))
                samples = sample(**kwargs)
                # NOTE: no need to return distortion factor
                return transformation(params, layers, samples)[0]
        return _loss

    def update_parameters(self, params, *args, **kwargs):
        for layer, layer_params in zip(self.layers, params):
            layer.update_parameters(layer_params)


@jax.jit
def sample_normal(key, n_samples, dim):
    return jax.random.normal(key, n_samples, dim)


@jax.jit
def sample_multivariate_normal(key, mean, cov):
    return jax.random.normal(key, mean, cov)


@jax.jit
def normal_logpdf(X):
    return norm.logpdf(X)


@jax.jit
def multivariate_normal_logpdf(X):
    return multivariate_normal.logpdf(X)


if __name__ == "__main__":
    from torch.utils import data
    from torchvision.datasets import MNIST
    from optax import adam
    # TODO: run mnist test


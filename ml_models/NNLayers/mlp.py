import jax
import jax.numpy as jnp
from ml_models.base import BaseModel


jax_seed = jax.random.PRNGKey(12)


class MLPLayer:
    def __init__(self, input_dim, output_dim, activiation):
        self.weights = jax.random.normal(jax_seed, (input_dim, output_dim))
        self.bias = jax.random.normal(jax_seed, (1,))
        self.activation = activiation

    @property
    def parameters(self):
        return {
            'weights': self.weights,
            'bias': self.bias
        }

    def update_parameters(self, params):
        self.weights = params['weights']
        self.bias = params['bias']

    @staticmethod
    def forward(params, X, activation):
        lin_pass = jnp.dot(params['weights'].T, X)
        lin_pass += params['bias']
        return activation(lin_pass)


class MLP(BaseModel):
    def __init__(self, input_dim, hidden_sizes, activation_function):
        super(MLP, self).__init__()
        self.layers = []
        for i, hidden_layer in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(MLPLayer(input_dim, hidden_sizes[0], activation_function))
            else:
                self.layers.append(MLPLayer(hidden_sizes[i - 1], hidden_sizes[i], activation_function))
        self.activation_function = activation_function

    @property
    def parameters(self):
        return [layer.parameters for layer in self.layers]

    def update_parameters(self, params, *args, **kwargs):
        for layer, lp in zip(self.layers, params):
            layer.update_parameters(lp)

    def forward(self):
        layers = self.layers
        activation = self.activation_function

        def _forward(params, x):
            z = x.copy()
            for layer, lp in zip(layers, params):
                z = layer.forward(lp, z, activation)
            return z
        return _forward


class RealNVPNN(MLP):
    def __init__(self, input_dim, hidden_sizes, activation_function):
        super(RealNVPNN, self).__init__(
            input_dim, hidden_sizes, activation_function
        )
        self.log_layer = MLPLayer(hidden_sizes[-1], input_dim, activiation=jnp.tanh)
        self.final_layer = MLPLayer(hidden_sizes[-1], input_dim, activiation=lambda x: x)

    @property
    def parameters(self):
        layers = [layer.parameters for layer in self.layers]
        layers.append(self.log_layer.parameters)
        layers.append(self.final_layer.parameters)
        return layers

    def update_parameters(self, params, *args, **kwargs):
        for layer, lp in zip(self.layers, params):
            layer.update_parameters(lp)
        self.log_layer.update_parameters(params[-2])
        self.final_layer.update_parameters(params[-1])

    def forward(self):
        layers = self.layers
        log_layer = self.log_layer
        t_layer = self.final_layer
        activation = self.activation_function

        def _forward(params, x):
            z = x * 1
            for layer, lp in zip(layers, params):
                z = layer.forward(lp, z, activation)
            log_scale = log_layer.forward(params[-2], z, activation)
            shift = t_layer.forward(params[-1], z, activation)
            return log_scale, shift

        return _forward


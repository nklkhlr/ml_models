from normalizing_flows import (
    NormalizingFlow, PlanarFlow, RealNVP,
    sample_normal, sample_multivariate_normal,
    normal_logpdf, multivariate_normal_logpdf
)
from ml_models.training import Trainer
from ml_models.loss_functions import normal_nll
from jax.scipy.stats import norm
import jax
from optax import adam
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
# TODO: add MNIST/ImageNet test
# from torch.utils import data
# from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

N = 100
x, _ = make_moons(N, noise=.05)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# plt.hist2d(x[:, 0], x[:, 1], bins=500)
# plt.show()
# plt.scatter(x[:, 0], x[:, 1], s=20)
# plt.show()


flow = NormalizingFlow(
    mode="forward", models=[PlanarFlow(2)],  # RealNVP(2, [128, 128])],
    distribution=normal_logpdf, sampling=normal_logpdf
)

def nll_loss(x):
    return -x.mean()

trainer = Trainer(flow, adam, nll_loss, normal_nll, N, 10, 5, .8, {"learning_rate": 1e-10})
trainer.train(x_scaled.T, None)
trainer.plot_training_curves()
plt.show()
print("")

# TODO: visualize trained output

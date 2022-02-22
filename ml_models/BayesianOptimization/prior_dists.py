from jax import random
from jax.scipy.stats import multivariate_normal, uniform
from jax.numpy import sum as jnp_sum


class Prior:
    def __init__(self, *args, **kwargs):
        pass

    def sample(self, key: random.PRNGKey, n_samples: int):
        pass

    def pdf(self, x):
        pass


class UniformPrior(Prior):
    def __init__(self,  dim: int, lower_bound: int = 0, upper_bound: int = 1):
        super(UniformPrior, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if self.lower_bound > self.upper_bound:
            raise ValueError("")
        self.dim = dim

    def sample(self, key: random.PRNGKey, n_samples: int):
        """
        Sampling from a uniform distribution within the specified bounds

        :param key:
        :param n_samples:
        :return:
        """
        sampling = random.uniform(key, (n_samples, self.dim))
        # scaling
        return self.lower_bound + (self.upper_bound - self.lower_bound) * sampling

    def pdf(self, x):
        """
        Computing the probability density of a uniform distribution at x,
        where the mean is the lower bound and the scale the difference between
        upper and lower bound

        :param x:
        :return:
        """
        jnp_sum(uniform.pdf(x, self.lower_bound, self.upper_bound - self.lower_bound), axis=-1)


class GaussianPrior(Prior):
    def __init__(self, mean, covariance, dim):
        super(GaussianPrior, self).__init__()
        self.mean = mean
        self.cov = covariance
        self.dim = dim

    def sample(self, key: random.PRNGKey, n_samples: int):
        """
        Sampling from a multivariate gaussian distribution

        :param key:
        :param n_samples:
        :return:
        """
        return random.multivariate_normal(key, self.mean, self.cov, (n_samples,))

    def pdf(self, x):
        """
        Computing the probability density of a multivariate gaussian distribution at x

        :param x:
        :return:
        """
        multivariate_normal.pdf(x, self.mean, self.cov)

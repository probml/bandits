import jax.numpy as jnp
from ..nlds_lib.diagonal_extended_kalman_filter import DiagonalExtendedKalmanFilter
from .ekf_subspace import SubspaceNeuralBandit
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class DiagonalSubspaceNeuralBandit(SubspaceNeuralBandit):

    def __init__(self, num_features, num_arms, model, opt, prior_noise_variance, nwarmup=1000, nepochs=1000,
                 system_noise=0.0, observation_noise=1.0, n_components=0.9999, random_projection=False):
        super().__init__(num_features, num_arms, model, opt, prior_noise_variance, nwarmup, nepochs,
                         system_noise, observation_noise, n_components, random_projection)

    def init_bel(self, key, contexts, states, actions, rewards):
        bel = super().init_bel(key, contexts, states, actions, rewards)

        params_subspace_init, _, t = bel

        subspace_dim = self.n_components
        Q = jnp.ones(subspace_dim) * self.system_noise
        R = self.observation_noise

        covariance_subspace_init = jnp.ones(subspace_dim) * self.prior_noise_variance

        def fz(params):
            return params

        def fx(params, context, action):
            return self.predict_rewards(params, context)[action, None]

        ekf = DiagonalExtendedKalmanFilter(fz, fx, Q, R)
        self.ekf = ekf

        bel = (params_subspace_init, covariance_subspace_init, t)

        return bel

    def sample_params(self, key, bel):
        params_subspace, covariance_subspace, t = bel
        normal_dist = tfd.Normal(loc=params_subspace, scale=covariance_subspace)
        params_subspace = normal_dist.sample(seed=key)
        return params_subspace

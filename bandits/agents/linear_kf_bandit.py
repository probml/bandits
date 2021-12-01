import jax.numpy as jnp
from jax.ops import index_update
from jax.lax import scan
from jax.random import split
from nlds_lib.lds_lib_orig import KalmanFilterNoiseEstimation
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class LinearKFBandit:
    def __init__(self, num_features, num_arms, eta=6.0, lmbda=0.25):
        self.num_features = num_features
        self.num_arms = num_arms
        self.eta = eta
        self.lmbda = lmbda

    def init_bel(self, key, contexts, states, actions, rewards):
        v = 2 * self.eta * jnp.ones((self.num_arms,))
        tau = jnp.ones((self.num_arms,))

        Sigma0 = jnp.eye(self.num_features)
        mu0 = jnp.zeros((self.num_features,))

        Sigma = 1. / self.lmbda * jnp.repeat(Sigma0[None, ...], self.num_arms, axis=0)
        mu = Sigma @ mu0
        A = jnp.eye(self.num_features)
        Q = 0

        self.kf = KalmanFilterNoiseEstimation(A, Q, mu, Sigma, v, tau)

        def warmup_update(bel, cur):
            context, action, reward = cur
            bel = self.update_bel(bel, context, action, reward)
            return bel, None

        bel = (mu, Sigma, v, tau)
        bel, _ = scan(warmup_update, bel, (contexts, actions, rewards))

        return bel

    def update_bel(self, bel, context, action, reward):
        mu, Sigma, v, tau = bel
        state = (mu[action], Sigma[action], v[action], tau[action])
        xs = (context, reward)

        mu_k, Sigma_k, v_k, tau_k = self.kf.kalman_step(state, xs)

        mu = index_update(mu, action, mu_k)
        Sigma = index_update(Sigma, action, Sigma_k)
        v = index_update(v, action, v_k)
        tau = index_update(tau, action, tau_k)

        bel = (mu, Sigma, v, tau)

        return bel

    def sample_params(self, key, bel):
        sigma_key, w_key = split(key, 2)
        mu, Sigma, v, tau = bel

        lmbda = tfd.InverseGamma(v / 2., (v * tau) / 2.).sample(seed=sigma_key)
        V = lmbda[:, None, None]

        covariance_matrix = V * Sigma
        w = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance_matrix).sample(seed=w_key)

        return w

    def choose_action(self, key, bel, context):
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        w = self.sample_params(key, bel)
        predicted_reward = jnp.einsum("m,km->k", context, w)
        action = predicted_reward.argmax()
        return action

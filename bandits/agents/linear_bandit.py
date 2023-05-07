import jax.numpy as jnp
from jax import lax
from jax import random

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class LinearBandit:
    def __init__(self, num_features, num_arms, eta=6.0, lmbda=0.25):
        self.num_features = num_features
        self.num_arms = num_arms
        self.eta = eta
        self.lmbda = lmbda

    def init_bel(self, key, contexts, states, actions, rewards):
        mu = jnp.zeros((self.num_arms, self.num_features))
        Sigma = 1. / self.lmbda * jnp.eye(self.num_features) * jnp.ones((self.num_arms, 1, 1))
        a = self.eta * jnp.ones((self.num_arms,))
        b = self.eta * jnp.ones((self.num_arms,))

        initial_bel = (mu, Sigma, a, b)

        def update(bel, cur):  # could do batch update
            context, action, reward = cur
            bel = self.update_bel(bel, context, action, reward)
            return bel, None

        bel, _ = lax.scan(update, initial_bel, (contexts, actions, rewards))
        return bel

    def update_bel(self, bel, context, action, reward):
        mu, Sigma, a, b = bel

        mu_k, Sigma_k = mu[action], Sigma[action]
        Lambda_k = jnp.linalg.inv(Sigma_k)
        a_k, b_k = a[action], b[action]

        # weight params
        Lambda_update = jnp.outer(context, context) + Lambda_k
        Sigma_update = jnp.linalg.inv(Lambda_update)
        mu_update = Sigma_update @ (Lambda_k @ mu_k + context * reward)
        # noise params
        a_update = a_k + 1 / 2
        b_update = b_k + (reward ** 2 + mu_k.T @ Lambda_k @ mu_k - mu_update.T @ Lambda_update @ mu_update) / 2

        # Update only the chosen action at time t
        mu  = mu.at[action].set(mu_update)
        Sigma = Sigma.at[action].set(Sigma_update)
        a = a.at[action].set(a_update)
        b = b.at[action].set(b_update)
 
        bel = (mu, Sigma, a, b)

        return bel

    def sample_params(self, key, bel):
        mu, Sigma, a, b = bel

        sigma_key, w_key = random.split(key, 2)
        sigma2_samp = tfd.InverseGamma(concentration=a, scale=b).sample(seed=sigma_key)
        covariance_matrix = sigma2_samp[:, None, None] * Sigma
        w = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance_matrix).sample(
            seed=w_key)
        return w

    def choose_action(self, key, bel, context):
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        w = self.sample_params(key, bel)
        predicted_reward = jnp.einsum("m,km->k", context, w)
        action = predicted_reward.argmax()
        return action

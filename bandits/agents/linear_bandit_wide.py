# linear bandit with a single linear layer applied to a "wide" feature vector phi(s,a).
# So reward = w' * phi(s,a). If the vector is block structured one-hot, in which we put
# phi(s) into slot/block a, then we have w' phi(s,a) = w_a phi(s), which is a standard linaer model.
# For example, suppose phi(s)=[s1,s2] and we have 3 actions.
# Then phi(s,a=1) = [s1 s2 0 0 0 0], phi(s,a=3) = [0 0 0 0 s1 s2].
# Similarly let w = [w11 w12. w21 w22  w31 w32] where w(i,j) is weight for action i, feature j.
# Then w'phi(s,a=1) = [w11 w12] = w_1.

import jax.numpy as jnp
from jax import vmap
from jax.random import split
from jax.nn import one_hot
from jax.lax import scan

from .agent_utils import NIGupdate

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class LinearBanditWide:
    def __init__(self, num_features, num_arms, eta=6.0, lmbda=0.25):
        self.num_features = num_features
        self.num_arms = num_arms
        self.eta = eta
        self.lmbda = lmbda

    def widen(self, context, action):
        phi = jnp.zeros((self.num_arms, self.num_features))
        phi = phi.at[action].set(context)
        return phi.flatten()

    def init_bel(self, key, contexts, states, actions, rewards):
        mu = jnp.zeros((self.num_arms * self.num_features))
        Sigma = 1 / self.lmbda * jnp.eye(self.num_features * self.num_arms)
        a = self.eta * jnp.ones((self.num_arms * self.num_features,))
        b = self.eta * jnp.ones((self.num_arms * self.num_features,))

        initial_bel = (mu, Sigma, a, b)

        def update(bel, cur):  # could do batch update
            phi, reward = cur
            bel = NIGupdate(bel, phi, reward)
            return bel, None

        phis = vmap(self.widen)(contexts, actions)
        bel, _ = scan(update, initial_bel, (phis, rewards))
        return bel

    def update_bel(self, bel, context, action, reward):
        phi = self.widen(context, action)
        bel = NIGupdate(bel, phi, reward)
        return bel

    def sample_params(self, key, bel):
        mu, Sigma, a, b = bel

        sigma_key, w_key = split(key, 2)
        sigma2_samp = tfd.InverseGamma(concentration=a, scale=b).sample(seed=sigma_key)
        covariance_matrix = sigma2_samp * Sigma
        w = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance_matrix).sample(
            seed=w_key)
        return w

    def choose_action(self, key, bel, context):
        w = self.sample_params(key, bel)

        def get_reward(action):
            reward = one_hot(action, self.num_arms)
            phi = self.widen(context, action)
            reward = phi @ w
            return reward

        actions = jnp.arange(self.num_arms)
        rewards = vmap(get_reward)(actions)
        action = jnp.argmax(rewards)

        return action

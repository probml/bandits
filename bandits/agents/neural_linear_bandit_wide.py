# reward = w' * phi(s,a; theta), where theta is learned

import jax.numpy as jnp
from jax import vmap
from jax.random import split
from jax.nn import one_hot
from jax.lax import scan, cond

import optax

from flax.training import train_state

from .agent_utils import NIGupdate, train
from scripts.training_utils import  MLP

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class NeuralLinearBanditWide:
    def __init__(self, num_features, num_arms, model=None, opt=optax.adam(learning_rate=1e-2), eta=6.0, lmbda=0.25,
                 update_step_mod=100, batch_size=5000, nepochs=3000):
        self.num_features = num_features
        self.num_arms = num_arms

        if model is None:
            self.model = MLP(500, num_arms)
        else:
            try:
                self.model = model()
            except:
                self.model = model

        self.opt = opt
        self.eta = eta
        self.lmbda = lmbda
        self.update_step_mod = update_step_mod
        self.batch_size = batch_size
        self.nepochs = nepochs

    def init_bel(self, key, contexts, states, actions, rewards):

        key, mykey = split(key)
        initial_params = self.model.init(mykey, jnp.zeros((self.num_features,)))
        initial_train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=initial_params,
                                                            tx=self.opt)

        mu = jnp.zeros((self.num_arms, 500))
        Sigma = 1 * self.lmbda * jnp.eye(500) * jnp.ones((self.num_arms, 1, 1))
        a = self.eta * jnp.ones((self.num_arms,))
        b = self.eta * jnp.ones((self.num_arms,))
        t = 0

        def update(bel, x):
            context, action, reward = x
            return self.update_bel(bel, context, action, reward), None

        initial_bel = (mu, Sigma, a, b, initial_train_state, t)
        X = vmap(self.widen)(contexts)(actions)
        self.init_contexts_and_states(contexts, states)
        (bel, key), _ = scan(update, initial_bel, (contexts, actions, rewards))
        return bel

    def featurize(self, params, x, feature_layer="last_layer"):
        _, inter = self.model.apply(params, x, capture_intermediates=True)
        Phi, *_ = inter["intermediates"][feature_layer]["__call__"]
        return Phi

    def widen(self, context, action):
        phi = jnp.zeros((self.num_arms, self.num_features))
        phi[action] = context
        return phi.flatten()

    def cond_update_params(self, t):
        return (t % self.update_step_mod) == 0

    def init_contexts_and_states(self, contexts, states, actions, rewards):
        self.X = vmap(self.widen)(contexts)(actions)
        self.Y = rewards

    def update_bel(self, bel, context, action, reward):

        _, _, _, _, state, t = bel
        sgd_params = (state, t)

        phi = self.widen(self, context, action)
        state = cond(self.cond_update_params(t),
                     lambda sgd_params: train(self.model, sgd_params[0], phi, reward,
                                              nepochs=self.nepochs, t=sgd_params[1]),
                     lambda sgd_params: sgd_params[0], sgd_params)
        lin_bel = NIGupdate(bel, phi, reward)
        bel = (*lin_bel, state, t + 1)

        return bel

    def sample_params(self, key, bel):
        mu, Sigma, a, b, _, _ = bel
        sigma_key, w_key = split(key)
        sigma2 = tfd.InverseGamma(concentration=a, scale=b).sample(seed=sigma_key)
        covariance_matrix = sigma2[:, None, None] * Sigma
        w = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance_matrix).sample(seed=w_key)
        return w

    def choose_action(self, key, bel, context):
        w = self.sample_params(key, bel)

        def get_reward(action):
            reward = one_hot(action, self.num_arms)
            phi = self.widen(context, reward)
            reward = phi * w
            return reward

        actions = jnp.arange(self.num_arms)
        rewards = vmap(get_reward)(actions)
        action = jnp.argmax(rewards)

        return action

import jax.numpy as jnp
from jax import lax
from jax.random import split
from jax.ops import index_update

import optax

from flax.training import train_state

from .agent_utils import train
from experiments.training_utils import MLP
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class NeuralLinearBandit:
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
        Sigma = 1 / self.lmbda * jnp.eye(500) * jnp.ones((self.num_arms, 1, 1))
        a = self.eta * jnp.ones((self.num_arms,))
        b = self.eta * jnp.ones((self.num_arms,))
        t = 0

        def update(bel, x):
            context, action, reward = x
            return self.update_bel(bel, context, action, reward), None

        initial_bel = (mu, Sigma, a, b, initial_train_state, t)
        self.init_contexts_and_states(contexts, states)
        bel, _ = lax.scan(update, initial_bel, (contexts, actions, rewards))
        return bel

    def featurize(self, params, x, feature_layer="last_layer"):
        _, inter = self.model.apply(params, x, capture_intermediates=True)
        Phi, *_ = inter["intermediates"][feature_layer]["__call__"]
        return Phi.squeeze()

    def cond_update_params(self, t):
        return (t % self.update_step_mod) == 0

    def init_contexts_and_states(self, contexts, states):
        self.contexts = contexts
        self.states = states

    def update_bel(self, bel, context, action, reward):
        mu, Sigma, a, b, state, t = bel

        sgd_params = (state, t)

        def loss_fn(self, params):
            n_samples, *_ = self.contexts.shape
            final_t = lax.cond(t == 0, lambda t: n_samples, lambda t: t.astype(int), t)
            sample_range = (jnp.arange(n_samples) <= t)[:, None]

            pred_reward = self.model.apply({"params": params}, self.contexts)
            loss = (optax.l2_loss(pred_reward, self.states) * sample_range).sum() / final_t
            return loss, pred_reward

        state = lax.cond(self.cond_update_params(t),
                         lambda sgd_params: train(sgd_params[0], loss_fn=loss_fn, nepochs=self.nepochs)[0],
                         lambda sgd_params: sgd_params[0], sgd_params)

        transformed_context = self.featurize(state.params, context)

        mu_k, Sigma_k = mu[action], Sigma[action]
        Lambda_k = jnp.linalg.inv(Sigma_k)
        a_k, b_k = a[action], b[action]

        # weight params
        Lambda_update = jnp.outer(transformed_context, transformed_context) + Lambda_k
        Sigma_update = jnp.linalg.inv(Lambda_update)
        mu_update = Sigma_update @ (Lambda_k @ mu_k + transformed_context * reward)

        # noise params
        a_update = a_k + 1 / 2
        b_update = b_k + (reward ** 2 + mu_k.T @ Lambda_k @ mu_k - mu_update.T @ Lambda_update @ mu_update) / 2

        # update only the chosen action at time t
        mu = index_update(mu, action, mu_update)
        Sigma = index_update(Sigma, action, Sigma_update)
        a = index_update(a, action, a_update)
        b = index_update(b, action, b_update)
        t = t + 1

        bel = (mu, Sigma, a, b, state, t)
        return bel

    def sample_params(self, key, bel):
        mu, Sigma, a, b, _, _ = bel
        sigma_key, w_key = split(key)
        sigma2 = tfd.InverseGamma(concentration=a, scale=b).sample(seed=sigma_key)
        covariance_matrix = sigma2[:, None, None] * Sigma
        w = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance_matrix).sample(seed=w_key)
        return w

    def choose_action(self, key, bel, context):
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        state = bel[-2]
        context_transformed = self.featurize(state.params, context)
        w = self.sample_params(key, bel)
        predicted_reward = jnp.einsum("m,km->k", context_transformed, w)
        action = predicted_reward.argmax()
        return action

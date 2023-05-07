import jax.numpy as jnp
from jax import jit
from jax.random import split
from jax.lax import scan, cond
from jax.nn import one_hot

import optax

from flax.training import train_state

from .agent_utils import train
from scripts.training_utils import MLP
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class LimitedMemoryNeuralLinearBandit:
    """
    Neural-linear bandit on a buffer. We train the model in the warmup
    phase considering all of the datapoints. After the warmup phase, we
    train from the rest of the dataset considering only a fixed number
    of datapoints to train on.
    """

    def __init__(self, num_features, num_arms, buffer_size, model=None, opt=optax.adam(learning_rate=1e-2), eta=6.0,
                 lmbda=0.25,
                 update_step_mod=100, nepochs=3000):

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
        self.nepochs = nepochs
        self.buffer_size = buffer_size
        self.buffer_indexer = jnp.arange(self.buffer_size)

    def init_bel(self, key, contexts, states, actions, rewards):
        """
        Initialize the multi-armed bandit model by training the model on the warmup phase
        doing a round-robin of the actions.
        """

        # Initialize feature matrix
        nsamples, nfeatures = contexts.shape
        initial_params = self.model.init(key, jnp.ones(nfeatures))

        num_features_last_layer = initial_params["params"]["last_layer"]["bias"].size
        mu = jnp.zeros((self.num_arms, num_features_last_layer))
        Sigma = 1 / self.lmbda * jnp.eye(num_features_last_layer) * jnp.ones((self.num_arms, 1, 1))
        a = self.eta * jnp.ones((self.num_arms,))
        b = self.eta * jnp.ones((self.num_arms,))
        initial_train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=initial_params,
                                                            tx=self.opt)
        t = 0

        context_buffer = jnp.zeros((self.buffer_size, nfeatures))
        reward_buffer = jnp.zeros(self.buffer_size)
        action_buffer = -jnp.ones(self.buffer_size)
        buffer_ix = 0

        def update(bel, x):
            context, action, reward = x
            return self.update_bel(bel, context, action, reward), None

        buffer = (context_buffer, reward_buffer, action_buffer, buffer_ix)

        initial_bel = (mu, Sigma, a, b, initial_train_state, t, buffer)

        bel, _ = scan(update, initial_bel, (contexts, actions, rewards))

        return bel

    def _update_buffer(self, buffer, new_item, index):
        """
        source: https://github.com/google/jax/issues/4590
        """
        buffer = buffer.at[index].set(new_item)
        index = (index + 1) % self.buffer_size
        return buffer, index

    def cond_update_params(self, t):
        cond1 = (t % self.update_step_mod) == 0
        cond2 = t > 0
        return cond1 * cond2

    def featurize(self, params, x, feature_layer="last_layer"):
        _, inter = self.model.apply(params, x, capture_intermediates=True)
        Phi, *_ = inter["intermediates"][feature_layer]["__call__"]
        return Phi.squeeze()

    def update_bel(self, bel, context, action, reward):
        mu, Sigma, a, b, state, t, buffer = bel
        context_buffer, reward_buffer, action_buffer, buffer_ix = buffer

        update_buffer = jit(self._update_buffer)
        context_buffer, _ = update_buffer(context_buffer, context, buffer_ix)
        reward_buffer, _ = update_buffer(reward_buffer, reward, buffer_ix)
        action_buffer, buffer_ix = update_buffer(action_buffer, action, buffer_ix)

        Y_buffer = one_hot(action_buffer, self.num_arms) * reward_buffer[:, None]

        num_elements = jnp.minimum(self.buffer_size, t)
        valmap = self.buffer_indexer <= num_elements.astype(float)
        valmap = valmap[:, None]

        @jit
        def loss_fn(params):
            pred_reward = self.model.apply(params, context_buffer)
            loss = jnp.where(valmap, optax.l2_loss(pred_reward, Y_buffer), 0.0)
            loss = loss.sum() / num_elements
            return loss

        state = cond(self.cond_update_params(t),
                     lambda s: train(s, loss_fn=loss_fn, nepochs=self.nepochs, has_aux=False)[0],
                     lambda s: s, state)

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
        mu = mu.at[action].set(mu_update)
        Sigma = Sigma.at[action].set(Sigma_update)
        a = a.at[action].set(a_update)
        b = b.at[action].set(b_update)
        t = t + 1

        buffer = (context_buffer, reward_buffer, action_buffer, buffer_ix)

        bel = (mu, Sigma, a, b, state, t, buffer)

        return bel

    def sample_params(self, key, bel):
        mu, Sigma, a, b, _, _, _ = bel
        sigma_key, w_key = split(key)
        sigma2 = tfd.InverseGamma(concentration=a, scale=b).sample(seed=sigma_key)
        covariance_matrix = sigma2[:, None, None] * Sigma
        w = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance_matrix).sample(seed=w_key)
        return w

    def choose_action(self, key, bel, context):
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        state = bel[-3]
        context_transformed = self.featurize(state.params, context)
        w = self.sample_params(key, bel)
        predicted_reward = jnp.einsum("m,km->k", context_transformed, w)
        action = predicted_reward.argmax()
        return action

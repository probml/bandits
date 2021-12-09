import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree

import optax
from flax.training import train_state

from .agent_utils import train
from scripts.training_utils import MLP
from nlds_lib.diagonal_extended_kalman_filter import DiagonalExtendedKalmanFilter

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class DiagonalNeuralBandit:
    def __init__(self, num_features, num_arms, model, opt, prior_noise_variance, nwarmup=1000, nepochs=1000,
                 system_noise=0.0, observation_noise=1.0):
        """
        Subspace Neural Bandit implementation.
        Parameters
        ----------
        num_arms: int
            Number of bandit arms / number of actions
        environment : Environment
            The environment to be used.
        model : flax.nn.Module
            The flax model to be used for the bandits. Note that this model is independent of the
            model architecture. The only constraint is that the last layer should have the same 
            number of outputs as the number of arms.
        learning_rate : float
            The learning rate for the optimizer used for the warmup phase.
        momentum : float
            The momentum for the optimizer used for the warmup phase.
        nepochs : int
            The number of epochs to be used for the warmup SGD phase.
        """
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
        self.prior_noise_variance = prior_noise_variance
        self.nwarmup = nwarmup
        self.nepochs = nepochs
        self.system_noise = system_noise
        self.observation_noise = observation_noise

    def init_bel(self, key, contexts, states, actions, rewards):
        initial_params = self.model.init(key, jnp.ones((1, self.num_features)))["params"]
        initial_train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=initial_params,
                                                            tx=self.opt)

        def loss_fn(params):
            pred_reward = self.model.apply({"params": params}, contexts)[:, actions.astype(int)]
            loss = optax.l2_loss(pred_reward, states[:, actions.astype(int)]).mean()
            return loss, pred_reward

        warmup_state, _ = train(initial_train_state, loss_fn=loss_fn, nepochs=self.nepochs)

        params_full_init, reconstruct_tree_params = ravel_pytree(warmup_state.params)
        nparams = params_full_init.size

        Q = jnp.ones(nparams) * self.system_noise
        R = self.observation_noise

        params_subspace_init = jnp.zeros(nparams)
        covariance_subspace_init = jnp.ones(nparams) * self.prior_noise_variance

        def predict_rewards(params, context):
            params_tree = reconstruct_tree_params(params)
            outputs = self.model.apply({"params": params_tree}, context)
            return outputs

        self.predict_rewards = predict_rewards

        def fz(params):
            return params

        def fx(params, context, action):
            return predict_rewards(params, context)[action, None]

        ekf = DiagonalExtendedKalmanFilter(fz, fx, Q, R)
        self.ekf = ekf
        bel = (params_subspace_init, covariance_subspace_init, 0)

        return bel

    def sample_params(self, key, bel):
        params_subspace, covariance_subspace, t = bel
        normal_dist = tfd.Normal(loc=params_subspace, scale=covariance_subspace)
        params_subspace = normal_dist.sample(seed=key)
        return params_subspace

    def update_bel(self, bel, context, action, reward):
        xs = (reward, (context, action))
        bel, _ = jit(self.ekf.filter_step)(bel, xs)
        return bel

    def choose_action(self, key, bel, context):
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        w = self.sample_params(key, bel)
        predicted_reward = self.predict_rewards(w, context)
        action = predicted_reward.argmax()
        return action

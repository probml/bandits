import optax
import jax.numpy as jnp
from jax import jit, device_put
from jax.random import split
from jax.flatten_util import ravel_pytree
from flax.training import train_state
from sklearn.decomposition import PCA
from .agent_utils import train, generate_random_basis, convert_params_from_subspace_to_full
from scripts.training_utils import MLP
from nlds_lib.extended_kalman_filter import ExtendedKalmanFilter
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class SubspaceNeuralBandit:
    def __init__(self, num_features, num_arms, model, opt, prior_noise_variance, nwarmup=1000, nepochs=1000,
                 system_noise=0.0, observation_noise=1.0, n_components=0.9999, random_projection=False):
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
        opt: flax.optim.Optimizer
            The optimizer to be used for training the model.
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
        self.n_components = n_components
        self.random_projection = random_projection

    def init_bel(self, key, contexts, states, actions, rewards):
        warmup_key, projection_key = split(key, 2)
        initial_params = self.model.init(warmup_key, jnp.ones((1, self.num_features)))["params"]
        initial_train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=initial_params,
                                                            tx=self.opt)

        def loss_fn(params):
            pred_reward = self.model.apply({"params": params}, contexts)[:, actions.astype(int)]
            loss = optax.l2_loss(pred_reward, states[:, actions.astype(int)]).mean()
            return loss, pred_reward

        warmup_state, warmup_metrics = train(initial_train_state, loss_fn=loss_fn, nepochs=self.nepochs)

        thinned_samples = warmup_metrics["params"][::2]
        params_trace = thinned_samples[-self.nwarmup:]

        if not self.random_projection:
            pca = PCA(n_components=self.n_components)
            pca.fit(params_trace)
            subspace_dim = pca.n_components_
            self.n_components = pca.n_components_
            projection_matrix = device_put(pca.components_)
        else:
            if type(self.n_components) != int:
                raise ValueError(f"n_components must be an integer, got {self.n_components}")
            total_dim = params_trace.shape[-1]
            subspace_dim = self.n_components
            projection_matrix = generate_random_basis(projection_key, subspace_dim, total_dim)

        Q = jnp.eye(subspace_dim) * self.system_noise
        R = jnp.eye(1) * self.observation_noise

        params_full_init, reconstruct_tree_params = ravel_pytree(warmup_state.params)
        params_subspace_init = jnp.zeros(subspace_dim)
        covariance_subspace_init = jnp.eye(subspace_dim) * self.prior_noise_variance

        def predict_rewards(params_subspace_sample, context):
            params = convert_params_from_subspace_to_full(params_subspace_sample, projection_matrix, params_full_init)
            params = reconstruct_tree_params(params)
            outputs = self.model.apply({"params": params}, context)
            return outputs

        self.predict_rewards = predict_rewards

        def fz(params):
            return params

        def fx(params, context, action):
            return predict_rewards(params, context)[action, None]

        ekf = ExtendedKalmanFilter(fz, fx, Q, R)
        self.ekf = ekf

        bel = (params_subspace_init, covariance_subspace_init, 0)
        return bel

    def sample_params(self, key, bel):
        params_subspace, covariance_subspace, t = bel
        mv_normal = tfd.MultivariateNormalFullCovariance(loc=params_subspace, covariance_matrix=covariance_subspace)
        params_subspace = mv_normal.sample(seed=key)
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

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from rebayes.low_rank_filter import lofi
from bandits.agents.base import BanditAgent
from rebayes.utils.sampling import sample_dlr_single
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

class LowRankFilterBandit(BanditAgent):
    """
    Regression bandit with low-rank filter.
    We consider a single neural network with k 
    outputs corresponding to the k arms.
    """
    def __init__(self, num_features, num_arms, model, memory_size, emission_covariance,
                initial_covariance, dynamics_weights, dynamics_covariance):
        self.num_features = num_features
        self.num_arms = num_arms
        self.model = model
        self.memory_size = memory_size

        self.emission_covariance = emission_covariance
        self.initial_covariance = initial_covariance
        self.dynamics_weights = dynamics_weights
        self.dynamics_covariance = dynamics_covariance
    
    def init_bel(self, key, contexts, states, actions, rewards):
        _, dim_in = contexts.shape
        params = self.model.init(key, jnp.ones((1, dim_in)))
        flat_params, recfn = ravel_pytree(params)

        def apply_fn(flat_params, xs):
            context, action = xs
            return self.model.apply(recfn(flat_params), context)[action, None]
        
        def predict_rewards(flat_params, context):
            return self.model.apply(recfn(flat_params), context)
        
        agent = lofi.RebayesLoFiDiagonal(
            dynamics_weights=self.dynamics_weights,
            dynamics_covariance=self.dynamics_covariance,
            emission_mean_function=apply_fn,
            emission_cov_function=lambda m, x: self.emission_covariance,
            adaptive_emission_cov=False,
            dynamics_covariance_inflation_factor=0.0,
            memory_size=self.memory_size,
            steady_state=False,
            emission_dist=tfd.Normal
        )
        bel = agent.init_bel(flat_params, self.initial_covariance)
        self.agent = agent
        self.predict_rewards = predict_rewards

        return bel
    
    def sample_params(self, key, bel):
        params_samp = self.agent.sample_state(bel, key, 1).ravel()
        return params_samp
    
    def update_bel(self, bel, context, action, reward):
        xs = (context, action)
        bel = self.agent.update_state(bel, xs, reward)
        return bel


class LowRankGreedy(LowRankFilterBandit):
    """
    Low-rank filter with greedy action selection.
    """
    def __init__(self, num_features, num_arms, model, memory_size, emission_covariance,
                initial_covariance, dynamics_weights, dynamics_covariance, epsilon):
        super().__init__(num_features, num_arms, model, memory_size, emission_covariance,
                initial_covariance, dynamics_weights, dynamics_covariance)
        self.epsilon = epsilon
    
    def choose_action(self, key, bel, context):
        key, key_action = jax.random.split(key)
        greedy = jax.random.bernoulli(key, 1 - self.epsilon)
        if greedy:
            rewards = self.predict_rewards(bel.state, context)
            action = jnp.argmax(rewards)
        else:
            action = jax.random.randint(key_action, (1,), 0, self.num_arms)
        return action
    

    def choose_action(self, key, bel, context):
        key, key_action = jax.random.split(key)
        greedy = jax.random.bernoulli(key, 1 - self.epsilon)

        def explore():
            action = jax.random.randint(key_action, shape=(), minval=0, maxval=self.num_arms)
            return action
        
        def exploit():
            params = bel.mean
            predicted_rewards = self.predict_rewards(params, context) 
            action = predicted_rewards.argmax(axis=-1)
            return action
        
        action = jax.lax.cond(greedy == 1, exploit, explore)
        return action
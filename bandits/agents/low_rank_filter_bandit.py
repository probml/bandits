import jax
import jax.numpy as jnp
from rebayes.low_rank_filter import lofi

class LowRankFilterBandit:
    def __init__(self, num_features, num_arms, model, memory_size):
        self.num_features = num_features
        self.num_arms = num_arms
        self.model = model
        self.memory_size = memory_size


    def init_bel(self, key, contexts, states, actions, rewards):
        ...
    
    def sample_params(self, key, bel):
        ...
    
    def update_bel(self, bel, context, action, reward):
        ...

    def choose_action(self, key, bel, context):
        ...

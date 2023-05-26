import jax
import jax.numpy as jnp
from jax.random import split
from rebayes.sgd_filter import replay_sgd
from bandits.agents.base import BanditAgent


class NeuralGreedyBandit(BanditAgent):
    def __init__(
        self, num_features, num_arms, model, memory_size, tx,
        epsilon, n_inner=100
    ):
        self.num_features = num_features
        self.num_arms = num_arms
        self.model = model
        self.memory_size = memory_size
        self.tx = tx
        self.epsilon = epsilon
        self.n_inner = int(n_inner)
    
    def init_bel(self, key, contexts, states, actions, rewards):
        _, dim_in = contexts.shape
        X_dummy = jnp.ones((1, dim_in))
        params = self.model.init(key, X_dummy)
        out = self.model.apply(params, X_dummy)
        dim_out = out.shape[-1]

        def apply_fn(params, xs):
            return self.model.apply(params, xs)

        def predict_rewards(params, contexts):
            return self.model.apply(params, contexts)

        agent = replay_sgd.FifoSGD(
            lossfn=lossfn_rmse_extra_dim,
            apply_fn=apply_fn,
            tx=self.tx,
            buffer_size=self.memory_size,
            dim_features=dim_in + 1, # +1 for the action
            dim_output=1,
            n_inner=self.n_inner
        )

        bel = agent.init_bel(params, None)
        self.agent = agent
        self.predict_rewards = predict_rewards

        return bel
    
    def sample_params(self, key, bel):
        return bel.params
    
    def update_bel(self, bel, context, action, reward):
        xs = jnp.r_[context, action]
        bel = self.agent.update_state(bel, xs, reward)
        return bel
    
    def choose_action(self, key, bel, context):
        key, key_action = split(key)
        greedy = jax.random.bernoulli(key, 1 - self.epsilon)

        def explore():
            action = jax.random.randint(key_action, shape=(), minval=0, maxval=self.num_arms)
            return action
        
        def exploit():
            params = self.sample_params(key, bel)
            predicted_rewards = self.predict_rewards(params, context) 
            action = predicted_rewards.argmax(axis=-1)
            return action
        
        action = jax.lax.cond(greedy == 1, exploit, explore)
        return action


def lossfn_rmse_extra_dim(params, counter, xs, y, apply_fn):
    """
    Lossfunction for regression problems.
    We consider an extra dimension in the input xs, which is the action.
    """
    X = xs[..., :-1]
    action = xs[..., -1].astype(jnp.int32)
    buffer_size = X.shape[0]
    ix_slice = jnp.arange(buffer_size)
    yhat = apply_fn(params, X)[ix_slice, action].ravel()
    y = y.ravel()
    err = jnp.power(y - yhat, 2)
    loss = (err * counter).sum() / counter.sum()
    return loss

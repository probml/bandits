import jax
import jax.numpy as jnp
from jax.random import split
from jax import vmap, lax
from jax.nn import one_hot

from flax.training import train_state

from agent_utils import MLP


class NeuralGreedy:
    def __init__(self, num_features, num_arms, epsilon, nepochs=1000, memory=None):
        self.num_features = num_features
        self.num_arms = num_arms
        self.model = MLP(num_features, num_arms)
        self.epsilon = epsilon
        self.nepochs = nepochs
        # self.memory = memory

    def encode(self, context, action):
        action_onehot = one_hot(action, self.num_arms)
        x = jnp.concatenate([context, action_onehot])
        return x

    def init_bel(self, key, contexts, states, actions, rewards):
        key, mykey = split(key)

        initial_params = self.model.init(mykey, jnp.zeros((self.num_features,)))
        initial_train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=initial_params,
                                                            tx=self.opt)

        t = 0

        def update(carry, x):
            bel, key = carry
            context, action, reward = x
            return (self.update_bel(bel, context, action, reward), key), None

        initial_bel = (initial_train_state, t)
        (bel, key), _ = lax.scan(update, (initial_bel, key), (contexts, actions, rewards))
        return bel

    def cond_update_params(self, t):
        return (t % self.update_step_mod) == 0

    def update_bel(self, bel, context, action, reward):
        state = bel

        if self.memory is not None:  # finite memory
            if len(self.y) == self.memory:  # memory is full
                self.X.pop(0)
                self.y.pop(0)

        x = self.encode(context, action)
        self.X = jnp.vstack([self.X, x])
        self.y = jnp.append(self.y, reward)

        state = lax.cond(self.cond_update_params(t),
                         lambda bel: train(self.model, bel[0], self.X, self.y, nepochs=self.nepochs, t=bel[1]),
                         lambda bel: bel[0], bel)

        bel = (state)
        return bel

    def init_bel(self, key, contexts, actions, rewards):
        X = jax.vmap(self.encode)(contexts, actions)
        y = rewards
        params = self.model.init(key, X)["params"]

        initial_train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=initial_params,
                                                            tx=self.opt)
        params = fit_model(key, self.model, X, y, params)
        bel = (initial_train_state)
        return bel

    def update_bel(self, key, bel, context, action, reward):
        (X, y, variables) = bel
        if self.memory is not None:  # finite memory
            if len(y) == self.memory:  # memory is full
                X.pop(0)
                y.pop(0)
        x = self.encode(context, action)
        X = jnp.vstack([X, x])
        y = jnp.append(y, reward)
        variables = fit_model(key, self.model, X, y, variables)
        bel = (X, y, variables)
        return bel

    def choose_action(self, key, bel, context):
        (X, y, params) = bel
        key, mykey = split(key)

        def explore(actions):
            # random action
            _, mykey = split(key)
            action = jax.random.choice(mykey, actions)
            return action

        def exploit(actions):
            # greedy action
            def get_reward(a):
                x = self.encode(context, a)
                return self.model.apply({"params": params}, x)

            predicted_rewards = vmap(get_reward)(actions)
            action = predicted_rewards.argmax()
            return action

        coin = jax.random.bernoulli(mykey, self.epsilon, (1,))[0]
        actions = jnp.arange(self.num_arms)
        action = jax.lax.cond(coin == 0, explore, exploit, actions)

        return action

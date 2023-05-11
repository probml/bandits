import jax
import jax.numpy as jnp


class BanditEnvironment:
    def __init__(self, key, X, Y, opt_rewards):
        # Randomise dataset rows 
        n_obs, n_features = X.shape

        new_ixs = jax.random.choice(key, n_obs, (n_obs,), replace=False)

        X = jnp.asarray(X)[new_ixs]
        Y = jnp.asarray(Y)[new_ixs]
        opt_rewards = jnp.asarray(opt_rewards)[new_ixs]

        self.contexts = X
        self.labels_onehot = Y
        self.opt_rewards = opt_rewards
        _, self.n_arms = Y.shape
        self.n_steps, self.n_features = X.shape

    def get_state(self, t):
        return self.labels_onehot[t]

    def get_context(self, t):
        return self.contexts[t]

    def get_reward(self, t, action):
        return jnp.float32(self.labels_onehot[t][action])

    def warmup(self, num_pulls):
        num_steps, num_actions = self.labels_onehot.shape
        # Create array of round-robin actions: 0, 1, 2, 0, 1, 2, 0, 1, 2, ...
        warmup_actions = jnp.arange(num_actions)
        warmup_actions = jnp.repeat(warmup_actions, num_pulls).reshape(num_actions, -1)
        actions = warmup_actions.reshape(-1, order="F").astype(jnp.int32)
        num_warmup_actions = len(actions)

        time_steps = jnp.arange(num_warmup_actions)

        def get_contexts_and_rewards(t, a):
            context = self.get_context(t)
            state = self.get_state(t)
            reward = self.get_reward(t, a)
            return context, state, reward

        contexts, states, rewards = jax.vmap(get_contexts_and_rewards, in_axes=(0, 0))(time_steps, actions)

        return contexts, states, actions, rewards

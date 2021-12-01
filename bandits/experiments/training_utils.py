import jax.numpy as jnp
from jax import vmap
from jax.random import split
from jax.lax import scan

import flax.linen as nn

import warnings

warnings.filterwarnings("ignore")


class MLP(nn.Module):
    num_arms: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(50, name="last_layer")(x))
        x = nn.Dense(self.num_arms)(x)
        return x


class MLPWide(nn.Module):
    num_arms: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(200)(x))
        x = nn.relu(nn.Dense(200, name="last_layer")(x))
        x = nn.Dense(self.num_arms)(x)
        return x


class LeNet5(nn.Module):
    num_arms: int

    @nn.compact
    def __call__(self, x):
        x = x if len(x.shape) > 1 else x[None, :]
        x = x.reshape((x.shape[0], 28, 28, 1))
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=16, kernel_size=(5, 5), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84, name="last_layer")(x)  # There are 10 classes in MNIST
        x = nn.relu(x)
        x = nn.Dense(features=self.num_arms)(x)
        return x.squeeze()


def train(key, bandit_cls, env, npulls, ntrials, bandit_kwargs, neural=True):
    nsteps, nfeatures = env.contexts.shape
    _, narms = env.labels_onehot.shape
    bandit = bandit_cls(nfeatures, narms, **bandit_kwargs)

    warmup_contexts, warmup_states, warmup_actions, warmup_rewards = env.warmup(npulls)

    key, mykey = split(key)
    bel = bandit.init_bel(mykey, warmup_contexts, warmup_states, warmup_actions, warmup_rewards)
    warmup = (warmup_contexts, warmup_states, warmup_actions, warmup_rewards)

    def single_trial(key):
        _, _, rewards = run_bandit(key, bandit, bel, env, warmup, nsteps=nsteps, neural=neural)
        return rewards

    if ntrials > 1:
        keys = split(key, ntrials)
        rewards_trace = vmap(single_trial)(keys)
        # rewards_trace = vmap(single_trial)(keys)
    else:
        rewards_trace = single_trial(key)

    return warmup_rewards, rewards_trace, env.opt_rewards


def run_bandit(key, bandit, bel, env, warmup, nsteps, neural=True):
    def step(bel, cur):
        mykey, t = cur
        context = env.get_context(t)

        action = bandit.choose_action(mykey, bel, context)
        reward = env.get_reward(t, action)
        bel = bandit.update_bel(bel, context, action, reward)

        return bel, (context, action, reward)

    warmup_contexts, warmup_states, warmup_actions, warmup_rewards = warmup
    nwarmup = len(warmup_rewards)

    steps = jnp.arange(nsteps - nwarmup) + nwarmup
    keys = split(key, nsteps - nwarmup)

    if neural:
        bandit.init_contexts_and_states(env.contexts[steps], env.labels_onehot[steps])
        mu, Sigma, a, b, params, _ = bel
        bel = (mu, Sigma, a, b, params, 0)

    _, (contexts, actions, rewards) = scan(step, bel, (keys, steps))

    contexts = jnp.vstack([warmup_contexts, contexts])
    actions = jnp.append(warmup_actions, actions)
    rewards = jnp.append(warmup_rewards, rewards)

    return contexts, actions, rewards


def summarize_results(warmup_rewards, rewards, spacing="\t\t"):
    """
    Print a summary of running a Bandit algorithm for a number of runs
    """
    warmup_reward = warmup_rewards.sum()
    rewards = rewards.sum(axis=-1)
    r_mean = rewards.mean()
    r_std = rewards.std()
    r_total = r_mean + warmup_reward

    print(f"{spacing}Expected Reward : {r_total:0.2f} ± {r_std:0.2f}")
    return r_total, r_std

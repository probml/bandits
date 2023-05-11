"""
In this demo, we evaluate the performance of the
lofi bandit on a tabular dataset
"""
import jax
import pickle
import numpy as np
import flax.linen as nn
from datetime import datetime
from bayes_opt import BayesianOptimization
from bandits.agents.low_rank_filter_bandit import LowRankFilterBandit
from bandits.environments.tabular_env import TabularEnvironment

from bandits.scripts.training_utils import train # Move to bandits.training.warmup_deploy

class MLP(nn.Module):
    num_arms: int

    @nn.compact
    def __call__(self, x):
        # x = nn.Dense(50)(x)
        # x = nn.relu(x)
        x = nn.Dense(50, name="last_layer")(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_arms)(x)
        return x


if __name__ == "__main__":
    ntrials = 10
    npulls = 20
    key = jax.random.PRNGKey(314)
    key_env, key_train = jax.random.split(key)
    ntrain = 500 # 5000
    env = TabularEnvironment(key_env, ntrain=ntrain, name='statlog', intercept=False, path="./bandit-data")
    num_arms = env.labels_onehot.shape[-1]
    model = MLP(num_arms)

    kwargs_lofi = {
        "emission_covariance": 0.01,
        "initial_covariance": 1.0,
        "dynamics_weights": 1.0,
        "dynamics_covariance": 0.0,
        "memory_size": 10,
        "model": model,
    }
    
    warmup_rewards, rewards_trace, opt_rewards = train(
        key_train, LowRankFilterBandit, env, npulls, ntrials, kwargs_lofi, neural=False
    )

    res = {
        "warmup_rewards": warmup_rewards,
        "rewards_trace": rewards_trace,
        "opt_rewards": opt_rewards,
    }
    res = jax.tree_map(np.array, res)

    # Store results
    datestr = datetime.now().strftime("%Y%m%d%H%M%S")
    path_to_results = f"./results/lofi_{datestr}.pkl"
    with open(path_to_results, "wb") as f:
        pickle.dump(res, f)
    print(f"Results stored in {path_to_results}")

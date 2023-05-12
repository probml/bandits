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
from bandits import training as btrain
from bandits.agents.low_rank_filter_bandit import LowRankFilterBandit
from bandits.environments.tabular_env import TabularEnvironment

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


def warmup_and_run(eval_hparams, transform_fn, bandit_cls, env, key, npulls, n_trials=1, **kwargs):
    key_warmup, key_train = jax.random.split(key, 2)
    hparams = transform_fn(eval_hparams)
    hparams = {**hparams, **kwargs}

    bandit = bandit_cls(env.n_features, env.n_arms, **hparams)

    bel, hist_warmup = btrain.warmup_bandit(key_warmup, bandit, env, npulls)
    bel, hist_train = btrain.run_bandit_trials(key_train, bel, bandit, env, t_start=npulls, n_trials=n_trials)

    res = {
        "hist_warmup": hist_warmup,
        "hist_train": hist_train,
    }
    res = jax.tree_map(np.array, res)

    return res


def transform_hparams_lofi(hparams):
    emission_covariance = np.exp(hparams["log_em_cov"])
    initial_covariance = np.exp(hparams["log_init_cov"])
    dynamics_weights = 1 - np.exp(hparams["log_1m_dweights"])
    dynamics_covariance = np.exp(hparams["log_dcov"])

    hparams = {
        "emission_covariance": emission_covariance,
        "initial_covariance": initial_covariance,
        "dynamics_weights": dynamics_weights,
        "dynamics_covariance": dynamics_covariance,
    }
    return hparams


def transform_hparams_linear(hparams):
    eta = hparams["eta"]
    lmbda = np.exp(hparams["log_lambda"])
    hparams = {
        "eta": eta,
        "lmbda": lmbda,
    }
    return hparams


if __name__ == "__main__":
    ntrials = 10
    npulls = 20
    key = jax.random.PRNGKey(314)
    key_env, key_warmup, key_train = jax.random.split(key, 3)
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

    n_features = env.n_features
    n_arms = env.n_arms
    bandit = LowRankFilterBandit(n_features, n_arms, **kwargs_lofi)

    bel, hist_warmup = btrain.warmup_bandit(key_warmup, bandit, env, npulls)
    bel, hist_train = btrain.run_bandit_trials(key_train, bel, bandit, env, t_start=npulls, n_trials=ntrials)

    res = {
        "hist_warmup": hist_warmup,
        "hist_train": hist_train,
    }
    res = jax.tree_map(np.array, res)

    # Store results
    datestr = datetime.now().strftime("%Y%m%d%H%M%S")
    path_to_results = f"./results/lofi_{datestr}.pkl"
    with open(path_to_results, "wb") as f:
        pickle.dump(res, f)
    print(f"Results stored in {path_to_results}")

from jax.random import split, PRNGKey

import optax
import pandas as pd

import argparse
from time import time

from ..environments.tabular_env import TabularEnvironment
from ..agents.ekf_subspace import SubspaceNeuralBandit

from .training_utils import train, MLP, summarize_results
from .mnist_exp import mapping, method_ordering


def main(config):
    # Tabular datasets
    key = PRNGKey(314)
    key, shuttle_key, covetype_key, adult_key = split(key, 4)
    ntrain = 5000

    shuttle_env = TabularEnvironment(shuttle_key, ntrain=ntrain, name='statlog', intercept=True)
    covertype_env = TabularEnvironment(covetype_key, ntrain=ntrain, name='covertype', intercept=True)
    adult_env = TabularEnvironment(adult_key, ntrain=ntrain, name='adult', intercept=True)
    environments = {"shuttle": shuttle_env, "covertype": covertype_env, "adult": adult_env}

    learning_rate = 0.05
    momentum = 0.9

    # Subspace Neural Bandit with SVD
    npulls, nwarmup = 20, 2000
    observation_noise = 0.0
    prior_noise_variance = 1e-4
    nepochs = 1000
    random_projection = False

    ekf_sub_svd = {"opt": optax.sgd(learning_rate, momentum), "prior_noise_variance": prior_noise_variance,
                   "nwarmup": nwarmup, "nepochs": nepochs,
                   "observation_noise": observation_noise,
                   "random_projection": random_projection}

    # Subspace Neural Bandit without SVD
    ekf_sub_rnd = ekf_sub_svd.copy()
    ekf_sub_rnd["random_projection"] = True

    bandits = {"EKF Subspace SVD": {"kwargs": ekf_sub_svd,
                                    "bandit": SubspaceNeuralBandit
                                    },
               "EKF Subspace RND": {"kwargs": ekf_sub_rnd,
                                    "bandit": SubspaceNeuralBandit
                                    }
               }

    results = []
    subspace_dimensions = [2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 100, 150, 200, 300, 400, 500]
    model_name = "MLP1"
    for env_name, env in environments.items():
        print("Environment : ", env_name)
        num_arms = env.labels_onehot.shape[-1]
        for subspace_dim in subspace_dimensions:
            model = MLP(num_arms)
            for bandit_name, properties in bandits.items():
                properties["kwargs"]["n_components"] = subspace_dim
                properties["kwargs"]["model"] = model
                key, mykey = split(key)
                print(f"\tBandit : {bandit_name}")
                start = time()
                warmup_rewards, rewards_trace, opt_rewards = train(mykey, properties["bandit"], env, npulls,
                                                                   config.ntrials,
                                                                   properties["kwargs"], neural=False)

                rtotal, rstd = summarize_results(warmup_rewards, rewards_trace)
                end = time()
                print(f"\t\tTime : {end - start}")
                results.append((env_name, bandit_name, model_name, subspace_dim, end - start, rtotal, rstd))

    df = pd.DataFrame(results)
    df = df.rename(columns={0: "Dataset", 1: "Method", 2: "Model", 3: "Subspace Dim", 4: "Time", 5: "Reward", 6: "Std"})

    df["Method"] = df["Method"].apply(lambda v: mapping[v])

    df["Subspace Dim"] = df['Subspace Dim'].astype(int)
    df["Reward"] = df['Reward'].astype(float)
    df["Time"] = df['Time'].astype(float)
    df["Std"] = df['Std'].astype(float)

    df["Rank"] = df["Method"].apply(lambda v: method_ordering[v])
    df.to_csv(config.filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrials', type=int, nargs='?', const=10, default=10)
    filepath = "bandits/results/tabular_subspace_results.csv"
    parser.add_argument('--filepath', type=str, nargs='?', const=filepath, default=filepath)

    # Parse the argument
    args = parser.parse_args()
    main(args)

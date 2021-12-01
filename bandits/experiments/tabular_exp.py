import optax
import pandas as pd
from jax.random import split, PRNGKey

import argparse
from time import time

from environments.tabular_env import TabularEnvironment

from agents.linear_bandit import LinearBandit
from agents.linear_kf_bandit import LinearKFBandit
from agents.linear_bandit_wide import LinearBanditWide
from agents.ekf_subspace import SubspaceNeuralBandit
from agents.ekf_orig_diag import DiagonalNeuralBandit
from agents.ekf_orig_full import EKFNeuralBandit
from agents.diagonal_subspace import DiagonalSubspaceNeuralBandit
from agents.limited_memory_neural_linear import LimitedMemoryNeuralLinearBandit

from .training_utils import train, MLP, summarize_results
from .mnist_exp import mapping, method_ordering


def main(config):
    # Tabular datasets
    key = PRNGKey(0)
    shuttle_key, covetype_key, adult_key, stock_key = split(key, 4)
    ntrain = 5000

    shuttle_env = TabularEnvironment(shuttle_key, ntrain=ntrain, name='statlog', intercept=False, path="../bandit-data")
    covertype_env = TabularEnvironment(covetype_key, ntrain=ntrain, name='covertype', intercept=False, path="../bandit-data")
    adult_env = TabularEnvironment(adult_key, ntrain=ntrain, name='adult', intercept=False, path="../bandit-data")
    environments = {"shuttle": shuttle_env, "covertype": covertype_env, "adult": adult_env}

    # Linear & Linear Wide
    linear = {}

    # Neural Linear Limited
    eta = 6.0
    lmbda = 0.25

    learning_rate = 0.05
    momentum = 0.9
    prior_noise_variance = 1e-3
    observation_noise = 0.01

    update_step_mod = 100
    buffer_size = 20
    nepochs = 100

    nl_lim = {"buffer_size": buffer_size, "opt": optax.sgd(learning_rate, momentum), "eta": eta, "lmbda": lmbda,
              "update_step_mod": update_step_mod, "nepochs": nepochs}

    buffer_size = 5000

    # Neural Linear Limited
    nl_unlim = nl_lim.copy()
    nl_unlim["buffer_size"] = buffer_sizebuffer_size = 5000

    nl_unlim = nl_lim.copy()
    nl_unlim["buffer_size"] = buffer_size

    # Subspace Neural Bandit with SVD
    npulls, nwarmup = 20, 2000
    observation_noise = 0.0
    prior_noise_variance = 1e-4
    n_components = 470
    nepochs = 1000
    random_projection = False

    ekf_sub_svd = {"opt": optax.sgd(learning_rate, momentum), "prior_noise_variance": prior_noise_variance,
                   "nwarmup": nwarmup, "nepochs": nepochs,
                   "observation_noise": observation_noise, "n_components": n_components,
                   "random_projection": random_projection}

    # Subspace Neural Bandit without SVD
    ekf_sub_rnd = ekf_sub_svd.copy()
    ekf_sub_rnd["random_projection"] = True

    # EKF Neural & EKF Neural Diagonal
    system_noise = 0.0
    prior_noise_variance = 1e-3
    nepochs = 100
    nwarmup = 1000
    learning_rate = 0.05
    momentum = 0.9
    observation_noise = 0.01

    ekf_orig = {"opt": optax.sgd(learning_rate, momentum), "prior_noise_variance": prior_noise_variance,
                "nwarmup": nwarmup, "nepochs": nepochs,
                "system_noise": system_noise, "observation_noise": observation_noise}

    bandits = {"Linear": {"kwargs": linear,
                          "bandit": LinearBandit
                          },
               "Linear KF": {"kwargs": linear.copy(),
                             "bandit": LinearKFBandit
                             },
               "Linear Wide": {"kwargs": linear,
                               "bandit": LinearBanditWide
                               },
               "Limited Neural Linear": {"kwargs": nl_lim,
                                         "bandit": LimitedMemoryNeuralLinearBandit
                                         },
               "Unlimited Neural Linear": {"kwargs": nl_unlim,
                                           "bandit": LimitedMemoryNeuralLinearBandit
                                           },
               "EKF Subspace SVD": {"kwargs": ekf_sub_svd,
                                    "bandit": SubspaceNeuralBandit
                                    },
               "EKF Subspace RND": {"kwargs": ekf_sub_rnd,
                                    "bandit": SubspaceNeuralBandit
                                    },
               "EKF Diagonal Subspace SVD": {"kwargs": ekf_sub_svd,
                                             "bandit": DiagonalSubspaceNeuralBandit
                                             },
               "EKF Diagonal Subspace RND": {"kwargs": ekf_sub_rnd,
                                             "bandit": DiagonalSubspaceNeuralBandit
                                             },
               "EKF Orig Diagonal": {"kwargs": ekf_orig,
                                     "bandit": DiagonalNeuralBandit
                                     },
               "EKF Orig Full": {"kwargs": ekf_orig,
                                 "bandit": EKFNeuralBandit
                                 }
               }

    results = []

    for env_name, env in environments.items():
        print("Environment : ", env_name)
        num_arms = env.labels_onehot.shape[-1]
        models = {"MLP1": MLP(num_arms)}  # You could also add MLPWide(num_arms)

        for model_name, model in models.items():
            for bandit_name, properties in bandits.items():
                if not bandit_name.startswith("Linear"):
                    properties["kwargs"]["model"] = model
                print(f"\tBandit : {bandit_name}")
                key = PRNGKey(314)
                start = time()
                warmup_rewards, rewards_trace, opt_rewards = train(key, properties["bandit"], env, npulls,
                                                                   config.ntrials,
                                                                   properties["kwargs"], neural=False)

                rtotal, rstd = summarize_results(warmup_rewards, rewards_trace)
                end = time()
                print(f"\t\tTime : {end - start:0.3f}s")
                results.append((env_name, bandit_name, end - start, rtotal, rstd))

    # initialize results given in the paper

    lim2data = [["shuttle", "Lim2", 42.20236960171787, 4826.4, 319.82351111111],
                ["covertype", "Lim2", 124.96883611524915, 2660.7, 333.93744444444],
                ["adult", "Lim2", 34.89770766110576, 3985.5, 113.127926],
                ]

    neuraltsdata = [
        ["shuttle", "NeuralTS", 0.0, 4348, 265],
        ["covertype", "NeuralTS", 0.0, 1877, 83],
        ["adult", "NeuralTS", 0.0, 3769, 2], ]

    df = pd.DataFrame(results + lim2data + neuraltsdata)
    df = df.rename(columns={0: "Dataset", 1: "Method", 2: "Time", 3: "Reward", 4: "Std"})

    df["Method"] = df["Method"].apply(lambda v: mapping[v])

    df["Reward"] = df['Reward'].astype(float)
    df["Time"] = df['Time'].astype(float)
    df["Std"] = df['Std'].astype(float)

    df["Rank"] = df["Method"].apply(lambda v: method_ordering[v])
    df.to_csv(config.filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrials', type=int, nargs='?', const=10, default=10)
    filepath = "bandits/results/tabular_results.csv"
    parser.add_argument('--filepath', type=str, nargs='?', const=filepath, default=filepath)

    # Parse the argument
    args = parser.parse_args()
    main(args)

from jax.random import PRNGKey

import optax
import pandas as pd

import argparse
from time import time

from environments.movielens_env import MovielensEnvironment

from agents.linear_bandit import LinearBandit
from agents.linear_kf_bandit import LinearKFBandit
from agents.ekf_subspace import SubspaceNeuralBandit
from agents.ekf_orig_diag import DiagonalNeuralBandit
from agents.diagonal_subspace import DiagonalSubspaceNeuralBandit
from agents.limited_memory_neural_linear import LimitedMemoryNeuralLinearBandit

from .training_utils import train, MLP, MLPWide
from .mnist_exp import mapping, rank, summarize_results, method_ordering


def main(config):
    eta = 6.0
    lmbda = 0.25

    learning_rate = 0.01
    momentum = 0.9

    update_step_mod = 100
    buffer_size = 50
    nepochs = 100

    # Neural Linear Limited
    nl_lim = {"buffer_size": buffer_size, "opt": optax.sgd(learning_rate, momentum), "eta": eta, "lmbda": lmbda,
              "update_step_mod": update_step_mod, "nepochs": nepochs}

    # Neural Linear Unlimited
    buffer_size = 4800

    nl_unlim = nl_lim.copy()
    nl_unlim["buffer_size"] = buffer_size

    npulls, nwarmup = 2, 2000
    learning_rate, momentum = 0.8, 0.9
    observation_noise = 0.0
    prior_noise_variance = 1e-4
    n_components = 470
    nepochs = 1000
    random_projection = False

    # Subspace Neural with SVD
    ekf_sub_svd = {"opt": optax.sgd(learning_rate, momentum), "prior_noise_variance": prior_noise_variance,
                   "nwarmup": nwarmup, "nepochs": nepochs,
                   "observation_noise": observation_noise, "n_components": n_components,
                   "random_projection": random_projection}

    # Subspace Neural without SVD
    ekf_sub_rnd = ekf_sub_svd.copy()
    ekf_sub_rnd["random_projection"] = True

    system_noise = 0.0

    ekf_orig = {"opt": optax.sgd(learning_rate, momentum), "prior_noise_variance": prior_noise_variance,
                "nwarmup": nwarmup, "nepochs": nepochs,
                "system_noise": system_noise, "observation_noise": observation_noise}
    linear = {}

    bandits = {"Linear": {"kwargs": linear,
                          "bandit": LinearBandit
                          },
               "Linear KF": {"kwargs": linear.copy(),
                             "bandit": LinearKFBandit
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
               "EKF Diagonal Subspace SVD": {"kwargs": ekf_sub_svd.copy(),
                                             "bandit": DiagonalSubspaceNeuralBandit
                                             },
               "EKF Diagonal Subspace RND": {"kwargs": ekf_sub_rnd.copy(),
                                             "bandit": DiagonalSubspaceNeuralBandit
                                             },
               "EKF Orig Diagonal": {"kwargs": ekf_orig,
                                     "bandit": DiagonalNeuralBandit
                                     }
               }

    results = []
    repeats = [1]
    for repeat in repeats:
        key = PRNGKey(0)
        # Create the environment beforehand
        movielens = MovielensEnvironment(key, repeat=repeat)
        # Number of different digits
        num_arms = movielens.labels_onehot.shape[-1]
        models = {"MLP1": MLP(num_arms), "MLP2": MLPWide(num_arms)}

        for model_name, model in models.items():
            for bandit_name, properties in bandits.items():
                if not bandit_name.startswith("Linear"):
                    properties["kwargs"]["model"] = model
                print(f"Bandit : {bandit_name}")
                key = PRNGKey(314)
                start = time()
                warmup_rewards, rewards_trace, opt_rewards = train(key, properties["bandit"], movielens, npulls,
                                                                   config.ntrials, properties["kwargs"], neural=False)
                rtotal, rstd = summarize_results(warmup_rewards, rewards_trace, spacing="\t")
                end = time()
                print(f"\tTime : {end - start}:0.3f")
                results.append((bandit_name, model_name, end - start, rtotal, rstd))

    df = pd.DataFrame(results)
    df = df.rename(columns={0: "Method", 1: "Model", 2: "Time", 3: "Reward", 4: "Std"})
    df["Method"] = df["Method"].apply(lambda v: mapping[v])
    df["Rank"] = df["Method"].apply(lambda v: method_ordering[v])
    df["AltRank"] = df["Model"].apply(lambda v: rank[v])

    df["Reward"] = df['Reward'].astype(float)
    df["Time"] = df['Time'].astype(float)
    df["Std"] = df['Std'].astype(float)
    df.to_csv(config.filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrials', type=int, nargs='?', const=10, default=10)
    filepath = "bandits/results/movielens_results.csv"
    parser.add_argument('--filepath', type=str, nargs='?', const=filepath, default=filepath)

    # Parse the argument
    args = parser.parse_args()
    main(args)

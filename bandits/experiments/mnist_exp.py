import optax
from jax.random import PRNGKey

import argparse
from time import time

import pandas as pd

from environments.mnist_env import MnistEnvironment

from agents.linear_bandit import LinearBandit
from agents.linear_kf_bandit import LinearKFBandit
from agents.ekf_subspace import SubspaceNeuralBandit
from agents.ekf_orig_diag import DiagonalNeuralBandit
from agents.diagonal_subspace import DiagonalSubspaceNeuralBandit
from agents.limited_memory_neural_linear import LimitedMemoryNeuralLinearBandit

from .training_utils import train, MLP, MLPWide, LeNet5, summarize_results

method_ordering = {"EKF-Sub-SVD": 0,
                   "EKF-Sub-RND": 1,
                   "EKF-Sub-Diag-SVD": 2,
                   "EKF-Sub-Diag-RND": 3,
                   "EKF-Orig-Full": 4,
                   "EKF-Orig-Diag": 5,
                   "NL-Lim": 6,
                   "NL-Unlim": 7,
                   "Lin": 8,
                   "Lin-KF": 9,
                   "Lin-Wide": 9,
                   "Lim2": 10,
                   "NeuralTS": 11}

rank = {"MLP1": 0, "MLP2": 1, "LeNet5": 2}

mapping = {
    "Lim2": "Lim2",
    "EKF Subspace SVD": "EKF-Sub-SVD",
    "EKF Subspace RND": "EKF-Sub-RND",
    "EKF Diagonal Subspace SVD": "EKF-Sub-Diag-SVD",
    "EKF Diagonal Subspace RND": "EKF-Sub-Diag-RND",
    "EKF Orig Full": "EKF-Orig-Full",
    "Linear": "Lin",
    "Linear Wide": "Lin-Wide",
    "Linear KF": "Lin-KF",
    "Unlimited Neural Linear": "NL-Unlim",
    "Limited Neural Linear": "NL-Lim",
    "NeuralTS": "NeuralTS",
    "EKF Orig Diagonal": "EKF-Orig-Diag"
}


def main(config):
    key = PRNGKey(0)
    ntrain = 5000

    # Create the environment beforehand
    mnist_env = MnistEnvironment(key, ntrain=ntrain)

    # Number of different digits
    num_arms = 10
    models = {"MLP1": MLP(num_arms), "MLP2": MLPWide(num_arms), "LeNet5": LeNet5(num_arms)}

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

    npulls, nwarmup = 20, 2000
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

    for model_name, model in models.items():
        print(f"Model : {model_name}")
        for bandit_name, properties in bandits.items():
            if not bandit_name.startswith("Linear"):
                properties["kwargs"]["model"] = model
            elif model_name != "MLP1":
                continue
            print(f"\tBandit : {bandit_name}")
            key = PRNGKey(314)
            start = time()
            warmup_rewards, rewards_trace, opt_rewards = train(key, properties["bandit"], mnist_env, npulls,
                                                               config.ntrials,
                                                               properties["kwargs"], neural=False)
            rtotal, rstd = summarize_results(warmup_rewards, rewards_trace)
            end = time()
            print(f"\t\tTime : {end - start}")
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
    filepath = "bandits/results/mnist_results.csv"
    parser.add_argument('--filepath', type=str, nargs='?', const=filepath, default=filepath)

    # Parse the argument
    args = parser.parse_args()
    main(args)

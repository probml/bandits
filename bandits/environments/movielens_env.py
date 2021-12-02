import pandas as pd
import numpy as  np

import jax.numpy as jnp

from .environment import BanditEnvironment

MOVIELENS_NUM_USERS = 943
MOVIELENS_NUM_MOVIES = 1682


def load_movielens_data(filepath):
    dataset = pd.read_csv(filepath, delimiter='\t', header=None)
    columns = {0: 'user_id', 1: 'item_id', 2: 'ranking', 3: 'timestamp'}
    dataset = dataset.rename(columns=columns)
    dataset['user_id'] -= 1
    dataset['item_id'] -= 1
    dataset = dataset.drop(columns="timestamp")

    rankings_matrix = np.zeros((MOVIELENS_NUM_USERS, MOVIELENS_NUM_MOVIES))
    for i, row in dataset.iterrows():
        rankings_matrix[row["user_id"], row["item_id"]] = float(row["ranking"])
    return rankings_matrix


# https://github.com/tensorflow/agents/blob/master/tf_agents/bandits/environments/movielens_py_environment.py
def get_movielens(rank_k, num_movies, repeat=5):
    """Initializes the MovieLens Bandit environment.
    Args:
      rank_k : (int) Which rank to use in the matrix factorization.
      batch_size: (int) Number of observations generated per call.
      num_movies: (int) Only the first `num_movies` movies will be used by the
        environment. The rest is cut out from the data.
    """
    num_actions = num_movies
    context_dim = rank_k

    # Compute the matrix factorization.
    data_matrix = load_movielens_data("../bandit-data/ml-100k/u.data")
    # Keep only the first items.
    data_matrix = data_matrix[:, :num_movies]
    # Filter the users with no iterm rated.
    nonzero_users = list(np.nonzero(np.sum(data_matrix, axis=1) > 0.0)[0]) * repeat
    data_matrix = data_matrix[nonzero_users, :]
    effective_num_users = len(nonzero_users)

    # Compute the SVD.
    u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)

    # Keep only the largest singular values.
    u_hat = u[:, :context_dim] * np.sqrt(s[:context_dim])
    v_hat = np.transpose(np.transpose(vh[:rank_k, :]) * np.sqrt(s[:rank_k]))
    approx_ratings_matrix = np.matmul(u_hat, v_hat)
    opt_rewards = np.max(approx_ratings_matrix, axis=1)
    return u_hat, approx_ratings_matrix, opt_rewards


def MovielensEnvironment(key, rank_k=20, num_movies=20, repeat=5, intercept=False):
    X, y, opt_rewards = get_movielens(rank_k, num_movies, repeat)

    if intercept:
        X = jnp.hstack([jnp.ones_like(X[:, :1]), X])

    return BanditEnvironment(key, X, y, opt_rewards)

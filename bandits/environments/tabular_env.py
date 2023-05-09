import jax.numpy as jnp
from jax.random import split, permutation
from jax.nn import one_hot

import numpy as np
import pandas as pd

import pickle
import requests
import io

from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.datasets import fetch_openml

from .environment import BanditEnvironment


# https://github.com/ofirnabati/Neural-Linear-Bandits-with-Likelihood-Matching/blob/85b541c225ec453bbf54650291616759e28b59d5/bandits/data/data_sampler.py#L528
def safe_std(values):
    """Remove zero std values for ones."""
    return np.array([val if val != 0.0 else 1.0 for val in values])


def read_file_from_url(name):
    if name == 'adult':
        url = "https://raw.githubusercontent.com/probml/probml-data/main/data/adult.data"
    elif name == 'covertype':
        url = "https://raw.githubusercontent.com/probml/probml-data/main/data/covtype.data"
    else:
        url = "https://raw.githubusercontent.com/probml/probml-data/main/data/shuttle.trn"

    download = requests.get(url).content
    file = io.StringIO(download.decode('utf-8'))
    return file


# https://github.com/ofirnabati/Neural-Linear-Bandits-with-Likelihood-Matching/blob/85b541c225ec453bbf54650291616759e28b59d5/bandits/data/data_sampler.py#L528
def classification_to_bandit_problem(X, y, narms=None):
    """Normalize contexts and encode deterministic rewards."""

    if narms is None:
        narms = np.max(y) + 1

    ntrain = X.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    sstd = safe_std(np.std(X, axis=0, keepdims=True)[0, :])

    # Normalize features
    X = ((X - np.mean(X, axis=0, keepdims=True)) / sstd)

    # One hot encode labels as rewards
    y = one_hot(y, narms)

    opt_rewards = np.ones((ntrain,))

    return X, y, opt_rewards


# https://github.com/ofirnabati/Neural-Linear-Bandits-with-Likelihood-Matching/blob/85b541c225ec453bbf54650291616759e28b59d5/bandits/data/data_sampler.py#L165
def sample_shuttle_data():
    """Returns bandit problem dataset based on the UCI statlog data.
    Returns:
      dataset: Sampled matrix with rows: (X, y)
      opt_vals: Vector of deterministic optimal (reward, action) for each context.
    https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
    """
    file = read_file_from_url("shuttle")
    data = np.loadtxt(file)

    narms = 7  # some of the actions are very rarely optimal.

    # Last column is label, rest are features
    X = data[:, :-1]
    y = data[:, -1].astype(int) - 1  # convert to 0 based index

    return classification_to_bandit_problem(X, y, narms=narms)


# https://github.com/ofirnabati/Neural-Linear-Bandits-with-Likelihood-Matching/blob/85b541c225ec453bbf54650291616759e28b59d5/bandits/data/data_sampler.py#L165
def sample_adult_data():
    """Returns bandit problem dataset based on the UCI adult data.
    Returns:
      dataset: Sampled matrix with rows: (X, y)
      opt_vals: Vector of deterministic optimal (reward, action) for each context.
    Preprocessing:
      * drop rows with missing values
      * convert categorical variables to 1 hot encoding
    https://archive.ics.uci.edu/ml/datasets/census+income
    """
    file = read_file_from_url("adult")
    df = pd.read_csv(file, header=None, na_values=[' ?']).dropna()

    narms = 2

    y = df[14].astype('str')
    df = df.drop([14, 6], axis=1)

    y = y.str.replace('.', '')
    y = y.astype('category').cat.codes.to_numpy()

    # Convert categorical variables to 1 hot encoding
    cols_to_transform = [1, 3, 5, 7, 8, 9, 13]
    df = pd.get_dummies(df, columns=cols_to_transform)
    X = df.to_numpy()

    return classification_to_bandit_problem(X, y, narms=narms)


# https://github.com/ofirnabati/Neural-Linear-Bandits-with-Likelihood-Matching/blob/85b541c225ec453bbf54650291616759e28b59d5/bandits/data/data_sampler.py#L165
def sample_covertype_data():
    """Returns bandit problem dataset based on the UCI Cover_Type data.
    Returns:
      dataset: Sampled matrix with rows: (X, y)
      opt_vals: Vector of deterministic optimal (reward, action) for each context.
    Preprocessing:
      * drop rows with missing labels
      * convert categorical variables to 1 hot encoding
    https://archive.ics.uci.edu/ml/datasets/Covertype
    """

    file = read_file_from_url("covertype")
    df = pd.read_csv(file, header=None, na_values=[' ?']).dropna()

    narms = 7

    # Assuming what the paper calls response variable is the label?
    # Last column is label.
    y = df[df.columns[-1]].astype('category').cat.codes.to_numpy()
    df = df.drop([df.columns[-1]], axis=1)

    X = df.to_numpy()

    return classification_to_bandit_problem(X, y, narms=narms)


def get_tabular_data_from_url(name):
    if name == 'adult':
        return sample_adult_data()
    elif name == 'covertype':
        return sample_covertype_data()
    elif name == 'statlog':
        return sample_shuttle_data()
    else:
        raise RuntimeError('Dataset does not exist')


def get_tabular_data_from_openml(name):
    if name == 'adult':
        X, y = fetch_openml('adult', version=2, return_X_y=True, as_frame=False)
    elif name == 'covertype':
        X, y = fetch_openml('covertype', version=3, return_X_y=True, as_frame=False)
    elif name == 'statlog':
        X, y = fetch_openml('shuttle', version=1, return_X_y=True, as_frame=False)
    else:
        raise RuntimeError('Dataset does not exist')

    X[np.isnan(X)] = - 1
    X = normalize(X)

    # generate one_hot coding:
    y = OneHotEncoder(sparse=False).fit_transform(y.reshape((-1, 1)))

    opt_rewards = jnp.ones((len(X),))

    return X, y, opt_rewards


def get_tabular_data_from_pkl(name, path):
    with open(f"{path}/bandit-{name}.pkl", "rb") as f:
        sampled_vals = pickle.load(f)

    contexts, opt_rewards, (*_, actions) = sampled_vals
    contexts = jnp.c_[jnp.ones_like(contexts[:, :1]), contexts]
    narms = len(jnp.unique(actions))
    actions = one_hot(actions, narms)
    return contexts, actions, opt_rewards


def TabularEnvironment(key, name, ntrain=0, intercept=True, load_from="pkl", path="./bandit-data"):
    """
    Parameters
    ----------
    key: jax.random.PRNGKey
        Random number generator key.
    name: str
        One of ['adult', 'covertype', 'statlog'].
    """
    if load_from == "url":
        X, y, opt_rewards = get_tabular_data_from_openml(name)
    elif load_from == "openml":
        X, y, opt_rewards = get_tabular_data_from_url(name)
    elif load_from == "pkl":
        X, y, opt_rewards = get_tabular_data_from_pkl(name, path)
    else:
        raise ValueError('load_from must be equal to pkl, openml or url.')

    ntrain = ntrain if ntrain < len(X) and ntrain > 0 else len(X)
    X, y = jnp.float32(X)[:ntrain], jnp.float32(y)[:ntrain]

    if intercept:
        X = jnp.hstack([jnp.ones_like(X[:, :1]), X])

    return BanditEnvironment(key, X, y, opt_rewards)

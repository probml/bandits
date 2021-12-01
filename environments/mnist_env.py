from jax.nn import one_hot
from jax.random import split, permutation

import numpy as np
from sklearn.datasets import fetch_openml

from .environment import BanditEnvironment


def get_mnist(key, ntrain):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    X = X / 255.
    y = y.astype(np.int32)

    perm = permutation(key, np.arange(len(X)))
    ntrain = ntrain if ntrain < len(X) else len(X)
    perm = perm[:ntrain]
    X, y = X[perm], y[perm]

    narms = len(np.unique(y))
    Y = one_hot(y, narms)

    opt_rewards = np.ones((ntrain,))
    return X, Y, opt_rewards


def MnistEnvironment(key, ntrain=0):
    key, mykey = split(key)
    X, Y, opt_rewards = get_mnist(mykey, ntrain)
    return BanditEnvironment(key, X, Y, opt_rewards)

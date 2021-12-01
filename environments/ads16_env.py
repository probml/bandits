import jax.numpy as jnp
from jax.ops import index_add
from jax.random import split, permutation
from jax.nn import one_hot

import pandas as pd

import requests
import io

from environment import BanditEnvironment


def get_ads16(key, ntrain, intercept):
    url = "https://raw.githubusercontent.com/probml/probml-data/main/data/ads16_preprocessed.csv"
    download = requests.get(url).content

    dataset = pd.read_csv(io.StringIO(download.decode('utf-8')))
    dataset.drop(columns=['Unnamed: 0'], inplace=True)
    dataset = dataset.sample(frac=1).reset_index(drop=True).to_numpy()

    ntrain = ntrain if ntrain > 0 and ntrain < len(dataset) else len(dataset)
    nusers, nads = 120, 300
    users = jnp.arange(nusers)

    n_ads_per_user, rem = divmod(ntrain, nusers)

    mykey, key = split(key)
    indices = permutation(mykey, users)[:rem]
    n_ads_per_user = jnp.ones((nusers,)) * int(n_ads_per_user)
    n_ads_per_user = index_add(n_ads_per_user, indices, 1).astype(jnp.int32)

    indices = jnp.array([])

    for user, nrow in enumerate(n_ads_per_user):
        mykey, key = split(key)
        df_indices = jnp.arange(user * nads, (user + 1) * nads)
        indices = jnp.append(indices, permutation(mykey, df_indices)[:nrow]).astype(jnp.int32)

    narms = 2
    dataset = dataset[indices]

    X = dataset[:, :-1]
    Y = one_hot(dataset[:, -1], narms)

    if intercept:
        X = jnp.concatenate([jnp.ones_like(X[:, :1]), X])

    opt_rewards = jnp.ones((len(X),))

    return X, Y, opt_rewards


def ADS16Environment(key, ntrain, intercept=False):
    mykey, key = split(key)
    X, Y, opt_rewards = get_ads16(mykey, ntrain, intercept)
    return BanditEnvironment(key, X, Y, opt_rewards)

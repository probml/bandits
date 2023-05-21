import jax
import jax.numpy as jnp
from functools import partial


def reshape_vvmap(x):
    """
    Reshape a 2D array to a 1D array.
    This is taken to be the output of a double vmap or pmap.
    """
    shape_orig = x.shape
    shape_new = (shape_orig[0] * shape_orig[1], *shape_orig[2:]) 
    return jnp.reshape(x, shape_new)


def step(bel, t, key_base, bandit, env):
    key = jax.random.fold_in(key_base, t)
    context = env.get_context(t)

    action = bandit.choose_action(key, bel, context)
    reward = env.get_reward(t, action)
    bel = bandit.update_bel(bel, context, action, reward)

    hist = {
        "actions": action,
        "rewards": reward
    }

    return bel, hist


def warmup_bandit(key, bandit, env, npulls):
    warmup_contexts, warmup_states, warmup_actions, warmup_rewards = env.warmup(npulls)
    bel = bandit.init_bel(key, warmup_contexts, warmup_states, warmup_actions, warmup_rewards)
    
    hist = {
        "states": warmup_states,
        "actions": warmup_actions,
        "rewards": warmup_rewards,
    }
    return bel, hist


def run_bandit(key, bel, bandit, env, t_start=0):
    step_part = partial(step, key_base=key, bandit=bandit, env=env)
    steps = jnp.arange(t_start, env.n_steps)
    bel, hist = jax.lax.scan(step_part, bel, steps)
    return bel, hist


def run_bandit_trials(key, bel, bandit, env, t_start=0, n_trials=1):
    keys = jax.random.split(key, n_trials)
    run_partal = partial(run_bandit, bel=bel, bandit=bandit, env=env, t_start=t_start)
    run_partial = jax.vmap(run_partal)

    bel, hist = run_partial(keys)
    return bel, hist

def run_bandit_trials_pmap(key, bel, bandit, env, t_start=0, n_trials=1):
    keys = jax.random.split(key, n_trials)
    run_partial = partial(run_bandit, bel=bel, bandit=bandit, env=env, t_start=t_start)
    run_partial = jax.pmap(run_partial)

    bel, hist = run_partial(keys)
    return bel, hist


def run_bandit_trials_multiple(key, bel, bandit, env, t_start, n_trials):
    """
    Run vmap over multiple trials, and pmap over multiple devices
    """
    ndevices = jax.local_device_count()
    nsamples_per_device = n_trials // ndevices
    keys = jax.random.split(key, ndevices)
    run_partial = partial(run_bandit_trials, bel=bel, bandit=bandit, env=env, t_start=t_start, n_trials=nsamples_per_device)
    run_partial = jax.pmap(run_partial)

    bel, hist = run_partial(keys)
    hist = jax.tree_map(reshape_vvmap, hist)
    bel = jax.tree_map(reshape_vvmap, bel)

    return bel, hist

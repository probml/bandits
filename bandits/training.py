import jax
import jax.numpy as jnp
from functools import partial

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

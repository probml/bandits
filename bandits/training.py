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
        "action": action,
        "reward": reward
    }

    return bel, hist


def warmup_bandit(key, bandit, env, npulls):
    warmup_contexts, warmup_states, warmup_actions, warmup_rewards = env.warmup(npulls)
    bel = bandit.init_bel(key, warmup_contexts, warmup_states, warmup_actions, warmup_rewards)
    
    hist = {
        "warmup_states": warmup_states,
        "warmup_actions": warmup_actions,
        "warmup_rewards": warmup_rewards,
    }
    return bel, hist


def run_bandit(key, bandit, bel, env, t_start=0):
    step_part = partial(step, key_base=key, bandit=bandit, env=env)
    steps = jnp.arange(t_start, env.nsteps)
    bel, hist = jax.lax.scan(step_part, bel, steps)
    return bel, hist

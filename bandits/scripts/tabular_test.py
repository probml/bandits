from time import time
from jax import random
from agents.linear_bandit import LinearBandit
from environments.tabular_env import TabularEnvironment
from .training_utils import train, MLP, summarize_results


def main(ntrials=10, npulls=20, nwarmup=2000, seed=314):
    key = random.PRNGKey(seed)
    ntrain = 5000
    env = TabularEnvironment(key, ntrain=ntrain, name="statlog", intercept=False, path="./bandit-data")
    linear_params = {}
    num_arms = env.labels_onehot.shape[-1]

    time_init = time()
    warmup_rewards, rewards_trace, opt_rewards = train(key, LinearBandit, env, npulls,
                                                        ntrials,
                                                        linear_params, neural=False)

    rtotal, rstd = summarize_results(warmup_rewards, rewards_trace)
    time_end = time()
    running_time = time() - time_init
    print(f"Time : {running_time:0.3f}s")

if __name__ == "__main__":
    main()
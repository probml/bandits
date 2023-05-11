from abc import ABC, abstractmethod
from functools import partial

class BanditAgent(ABC):
    def __init__(self, bandit):
        self.bandit = bandit

    
    @abstractmethod
    def init_bel(self, key, contexts, states, actions, rewards):
        ...

    @abstractmethod
    def sample_params(self, key, bel):
        ...

    @abstractmethod
    def update_bel(self, bel, context, action, reward):
        ...

    def step(self, bel, key, t, env):
        context = env.get_context(t)

        action = self.choose_action(key, bel, context)
        reward = env.get_reward(t, action)
        bel = self.update_bel(bel, context, action, reward)

        hist = {
            "context": context,
            "action": action,
            "reward": reward
        }

        return bel, hist

    def scan():
        ...

    def choose_action(self, key, bel, context):
        params = self.sample_params(key, bel)
        predicted_rewards = self.predict_rewards(params, context) 
        action = predicted_rewards.argmax()
        return action

    def __str__(self):
        return "BanditAgent"

    def __repr__(self):
        return str(self)

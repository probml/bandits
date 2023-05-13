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

    # TODO: Make it abstractmethod
    # @abstractmethod
    def predict_rewards(self, params, context):
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

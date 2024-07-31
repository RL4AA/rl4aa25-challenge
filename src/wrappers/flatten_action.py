import gymnasium as gym
from gymnasium.spaces.utils import flatten, flatten_space, unflatten


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = flatten_space(self.env.action_space)

    def action(self, action):
        return unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return flatten(self.env.action_space, action)

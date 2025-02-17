import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FilterAction(gym.ActionWrapper):
    def __init__(self, env, filter_indicies, replace="random"):
        super().__init__(env)

        self.filter_indicies = filter_indicies
        self.replace = replace

        self.action_space = spaces.Box(
            low=env.action_space.low[filter_indicies],
            high=env.action_space.high[filter_indicies],
            shape=env.action_space.low[filter_indicies].shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        if self.replace == "random":
            unfiltered = self.env.action_space.sample()
        else:
            unfiltered = np.full(
                self.env.action_space.shape,
                self.replace,
                dtype=self.env.action_space.dtype,
            )

        unfiltered[self.filter_indicies] = action

        return unfiltered

    def action_names(self):
        return self.env.action_names[self.filter_indicies]

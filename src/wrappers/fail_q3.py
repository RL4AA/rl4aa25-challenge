from typing import Union

import gymnasium as gym
import numpy as np


class FailQ3(gym.Wrapper):
    """Turn magnet Q3 off as if it had failed."""

    def __init__(self, env: gym.Env, at_step: int = 0) -> None:
        super().__init__(env)

        self.at_step = at_step

    def reset(self) -> Union[np.ndarray, dict]:
        obs = super().reset()

        self.steps_taken = 0
        self.has_magnet_failed = False

        if self.at_step == 0:
            magnet_values = self.env.unwrapped.backend.get_magnets()
            magnet_values[3] = 0.0
            self.env.unwrapped.backend.set_magnets(magnet_values)

            self.has_magnet_failed = True

            obs["magnets"][3] = 0.0

        return obs

    def step(self, action: np.ndarray) -> tuple:
        self.steps_taken += 1

        if self.steps_taken > self.at_step:
            magnet_values = self.env.unwrapped.backend.get_magnets()
            magnet_values[3] = 0.0
            self.env.unwrapped.backend.set_magnets(magnet_values)

            self.has_magnet_failed = True

            action[3] = 0.0

        return super().step(action)

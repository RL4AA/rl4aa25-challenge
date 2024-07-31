from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridScanPolicy:
    """
    Policy that performs a grid scan over the action space.
    """

    def __init__(self, env: gym.Env, samples_per_action: int) -> None:
        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.is_bounded()

        self._env_observation_space = env.observation_space

        action_component_linspaces = [
            np.linspace(
                env.action_space.low[i],
                env.action_space.high[i],
                samples_per_action,
                endpoint=True,
            )
            for i in range(env.action_space.shape[0])
        ]
        self._action_samples = np.array(
            list(product(*action_component_linspaces)), dtype=np.float32
        )

        self._action_sample_index = 0  # Track which action we're on

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Return the next action to take.
        """
        if self._action_sample_index >= len(self._action_samples):
            raise Exception("No more actions to sample")
        action = self._action_samples[self._action_sample_index]
        self._action_sample_index += 1
        return action

    @property
    def num_actions_in_grid(self) -> int:
        """
        Return the number of actions in the grid.
        """
        return len(self._action_samples)

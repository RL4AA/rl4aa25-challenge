import gymnasium as gym
import numpy as np


class PolishedDonkeyReward(gym.Wrapper):
    """Overrides the reward with the one used by polished donkey."""

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)

        self._previous_objective = self._objective_fn(
            achieved=observation["beam"], desired=observation["target"]
        )

        return self.observation(observation), info

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)

        objective = self._objective_fn(
            achieved=observation["beam"], desired=observation["target"]
        )
        reward_hat = self._previous_objective - objective
        new_reward = reward_hat if reward_hat > 0 else 2 * reward_hat

        self._previous_objective = objective

        return observation, new_reward, terminated, truncated, info

    def _objective_fn(self, achieved: np.ndarray, desired: np.ndarray):
        """
        Objective function copied over from environment in polished donkey commit.

        The only difference is that the weights were reordered to match the order of
        the beam parameters in the observation space.
        """
        offset = achieved - desired
        weights = np.array([1, 2, 1, 2])

        return np.log((weights * np.abs(offset)).sum())

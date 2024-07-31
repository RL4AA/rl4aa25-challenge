import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PolishedDonkeyCompatibility(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    super().observation_space.low[4],
                    super().observation_space.low[5],
                    super().observation_space.low[7],
                    super().observation_space.low[6],
                    super().observation_space.low[8],
                    super().observation_space.low[9],
                    super().observation_space.low[11],
                    super().observation_space.low[10],
                    super().observation_space.low[12],
                    super().observation_space.low[0],
                    super().observation_space.low[2],
                    super().observation_space.low[1],
                    super().observation_space.low[3],
                ]
            ),
            high=np.array(
                [
                    super().observation_space.high[4],
                    super().observation_space.high[5],
                    super().observation_space.high[7],
                    super().observation_space.high[6],
                    super().observation_space.high[8],
                    super().observation_space.high[9],
                    super().observation_space.high[11],
                    super().observation_space.high[10],
                    super().observation_space.high[12],
                    super().observation_space.high[0],
                    super().observation_space.high[2],
                    super().observation_space.high[1],
                    super().observation_space.high[3],
                ]
            ),
        )

        self.action_space = spaces.Box(
            low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32) * 0.1,
            high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32) * 0.1,
        )

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        return self.observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(
            self.action(action)
        )
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation):
        return np.array(
            [
                observation[4],
                observation[5],
                observation[7],
                observation[6],
                observation[8],
                observation[9],
                observation[11],
                observation[10],
                observation[12],
                observation[0],
                observation[2],
                observation[1],
                observation[3],
            ]
        )

    def action(self, action):
        return np.array(
            [
                action[0],
                action[1],
                action[3],
                action[2],
                action[4],
            ]
        )

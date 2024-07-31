from typing import Literal

import gymnasium as gym
import numpy as np


class LatticeAgnosticCompatibility(gym.Wrapper):
    """
    Compatibility wrapper for lattice-agnostic running on specific environments for, for
    example the EA.
    """

    def __init__(
        self, env: gym.Env, order: Literal["normal", "ea", "bc"] = "normal"
    ) -> None:
        super().__init__(env)

        if order not in ["normal", "ea", "bc"]:
            raise ValueError(f"Order {order} not supported.")
        self.order = order

        if order == "ea":  # change q3 and c
            # wanted observation in form: [mu_x, sigma_x, mu_y, sigma_y, q1, q2, q3,
            # cv, ch, mu_x_target, sigma_x_target, mu_y_target, sigma_y_target]
            newlow = np.copy(self.env.observation_space.low)
            newlow[6] = self.env.observation_space.low[7]
            newlow[7] = self.env.observation_space.low[6]

            newhigh = np.copy(self.env.observation_space.high)
            newhigh[6] = self.env.observation_space.high[7]
            newhigh[7] = self.env.observation_space.high[6]

            self.observation_space = gym.spaces.Box(
                low=newlow,
                high=newhigh,
            )
        elif order == "bc":
            # wanted observation in form: [mu_x, sigma_x, mu_y, sigma_y, q1, q2, q3,
            # cv, ch, mu_x_target, sigma_x_target, mu_y_target, sigma_y_target]
            newlow = np.copy(self.env.observation_space.low)
            newlow[6] = self.env.observation_space.low[8]
            newlow[7] = self.env.observation_space.low[6]
            newlow[8] = self.env.observation_space.low[7]

            newhigh = np.copy(self.env.observation_space.high)
            newhigh[6] = self.env.observation_space.high[8]
            newhigh[7] = self.env.observation_space.high[6]
            newhigh[8] = self.env.observation_space.high[7]

            self.observation_space = gym.spaces.Box(
                low=newlow,
                high=newhigh,
            )

        self.action_space = gym.spaces.Box(
            low=np.array([-30, -30, -30, -6e-3, -6e-3], dtype=np.float32) * 0.1,
            high=np.array([30, 30, 30, 6e-3, 6e-3], dtype=np.float32) * 0.1,
        )

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        transformed_observation = self.observation(observation)
        return transformed_observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            self.action(action)
        )
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation):
        if self.order == "ea":
            return np.array(
                [
                    observation[0],
                    observation[1],
                    observation[2],
                    observation[3],
                    observation[4],  # Q1
                    observation[5],  # Q2
                    observation[7],  # Q3
                    observation[6],  # CV
                    observation[8],  # CH
                    observation[9],
                    observation[10],
                    observation[11],
                    observation[12],
                ]
            )
        elif self.order == "bc":
            return np.array(
                [
                    observation[0],
                    observation[1],
                    observation[2],
                    observation[3],
                    observation[4],  # Q1 <- 4
                    observation[5],  # Q2 <- 5
                    observation[8],  # Q3 <- 6
                    observation[6],  # CV <- 7
                    observation[7],  # CH <- 8
                    observation[9],
                    observation[10],
                    observation[11],
                    observation[12],
                ]
            )
        else:
            return self.observation

    def action(self, action):
        if self.order == "ea":
            return np.array(
                [
                    action[0],
                    action[1],
                    action[3],
                    action[2],
                    action[4],
                ]
            )
        elif self.order == "bc":
            return np.array(
                [
                    action[0],
                    action[1],
                    action[3],
                    action[4],
                    action[2],
                ]
            )
        elif self.order == "normal":
            return action

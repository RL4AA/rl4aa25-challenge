from copy import deepcopy

import gymnasium as gym


class ReorderMagnets(gym.Wrapper):
    """
    Reorder the magnets of the environment to match the order assumed by the policy.
    This wrapper modifies both actions and observations.

    Pass the order of the environment magnets and the order of the policy magnets, e.g.:
     - env_magnet_order = ["Q1", "Q2", "CV", "Q3", "CH"]
     - policy_magnet_order = ["Q1","Q2", "Q3", "CV", "CH"]
    """

    def __init__(
        self,
        env,
        env_magnet_order: list[str],
        policy_magnet_order: list[str],
    ):
        super().__init__(env)

        # List of indicies of environment magnets in the policy order
        self.obs_indicies = [
            env_magnet_order.index(magnet) for magnet in policy_magnet_order
        ]
        # List of indicies of policy magnets in the environment order
        self.action_indicies = [
            policy_magnet_order.index(magnet) for magnet in env_magnet_order
        ]

        self.observation_space = deepcopy(env.observation_space)
        self.observation_space["magnets"] = gym.spaces.Box(
            low=self.observation_space["magnets"].low[self.obs_indicies],
            high=self.observation_space["magnets"].high[self.obs_indicies],
            dtype=self.observation_space["magnets"].dtype,
        )
        self.action_space = gym.spaces.Box(
            low=self.action_space.low[self.action_indicies],
            high=self.action_space.high[self.action_indicies],
            dtype=self.action_space.dtype,
        )

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self.observation(observation), info

    def step(self, action):
        obs, reward, terminated, trucated, info = self.env.step(self.action(action))
        return self.observation(obs), reward, terminated, trucated, info

    def observation(self, obs):
        obs["magnets"] = obs["magnets"][self.obs_indicies]
        return obs

    def action(self, action):
        return action[self.action_indicies]

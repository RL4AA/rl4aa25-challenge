import pickle

import gymnasium as gym


class NotVecNormalize(gym.Wrapper):
    """
    Normal Gym wrapper that replicates the functionality of Stable Baselines3's
    VecNormalize wrapper for non VecEnvs (i.e. `gym.Env`) in production.
    """

    def __init__(self, env, path):
        super().__init__(env)

        with open(path, "rb") as file_handler:
            self.vec_normalize = pickle.load(file_handler)

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self.vec_normalize.normalize_obs(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.vec_normalize.normalize_obs(observation)
        reward = self.vec_normalize.normalize_reward(reward)
        return observation, reward, terminated, truncated, info

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import is_wrapped, unwrap_wrapper
from tqdm import tqdm


class TQDMWrapper(gym.Wrapper):
    """
    Uses TQDM to show a progress bar for every step taken by the environment. If the
    passed `env` is already wrapper in a `TimeLimit` wrapper, this wrapper will use that
    as the maximum number of steps for the progress bar.
    """

    def reset(self, seed=None, options=None):
        if hasattr(self, "pbar"):
            self.pbar.close()

        obs, info = super().reset(seed=seed, options=options)

        if is_wrapped(self.env, TimeLimit):
            time_limit = unwrap_wrapper(self.env, TimeLimit)
            self.pbar = tqdm(total=time_limit._max_episode_steps)
        else:
            self.pbar = tqdm()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.pbar.update()
        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self, "pbar"):
            self.pbar.close()

        super().close()

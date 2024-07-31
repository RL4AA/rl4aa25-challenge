import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RescaleAction, TimeLimit
from stable_baselines3.common.env_checker import check_env

from src.environments import ea
from src.wrappers import CompensateEarlyTermination


def test_check_env():
    """
    Test that the `CompensateEarlyTermination` wrapper throws no exceptions under
    `check_env`.
    """
    env = ea.TransverseTuning(backend="cheetah")
    env = TimeLimit(env, max_episode_steps=50)
    env = CompensateEarlyTermination(env)
    env = RescaleAction(env, -1, 1)

    check_env(env)


def test_compensation_amount():
    """
    Test that the amount of compensation is correct, using an environment that always
    gives a reward of 1.0.
    """

    class OneRewardEnv(gym.Env):
        """Environment that always gives a reward of 1.0."""

        observation_space = spaces.Box(low=-1, high=1, shape=(1,))
        action_space = spaces.Box(low=-1, high=1, shape=(1,))

        def __init__(self, terminate_early=None):
            super().__init__()
            self.terminate_early = terminate_early

        def reset(self, seed=None, options=None):
            self._num_steps_taken = 0
            return np.zeros(1), {}

        def step(self, action):
            terminated = (
                False
                if self.terminate_early is None
                else self._num_steps_taken >= self.terminate_early
            )
            return np.zeros(1), 1.0, terminated, False, {}

    # Episode that runs until TimeLimit truncates it
    env = OneRewardEnv(terminate_early=None)
    env = TimeLimit(env, max_episode_steps=10)
    env = CompensateEarlyTermination(env)
    _, _ = env.reset()
    done = False
    episode_return_truncation = 0
    while not done:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        episode_return_truncation += reward
        done = terminated or truncated

    # Episode that terminates early due to "success"
    env = OneRewardEnv(terminate_early=5)
    env = TimeLimit(env, max_episode_steps=10)
    env = CompensateEarlyTermination(env)
    _, _ = env.reset()
    done = False
    episode_return_early_termination = 0
    while not done:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        episode_return_early_termination += reward
        done = terminated or truncated

    assert episode_return_truncation == episode_return_early_termination

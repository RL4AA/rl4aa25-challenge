import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import is_wrapped, unwrap_wrapper


class CompensateEarlyTermination(gym.Wrapper):
    """
    If rewards are always positive and the environment terminates early when certain
    conditions are met, the agent would learn to avoid this early termination. This
    behaviour is contrary to the desired behaviour.

    This wrapper compensates for the rewards lost through early termination by adding
    a reward on 1.0 for each step remaining until truncation.

    NOTE: This wrapper requires that the environment is wrapped with a `TimeLimit`
    wrapper.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert is_wrapped(env, TimeLimit), (
            "This wrapper requires that the environment is wrapped with a `TimeLimit`"
            " wrapper."
        )
        self._time_limit_wrapper = unwrap_wrapper(env, TimeLimit)

    def reset(self, seed=None, options=None):
        self._num_steps_taken = 0

        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._num_steps_taken += 1

        if terminated and not truncated:
            reward += (
                self._time_limit_wrapper._max_episode_steps - self._num_steps_taken
            )

        return observation, reward, terminated, truncated, info

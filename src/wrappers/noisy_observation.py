import gymnasium as gym
import numpy as np


class NoisyObservation(gym.ObservationWrapper):
    """
    Add normal-distributed shot-to-shot noise to the observation of the transverse tuning environments.
    """

    def __init__(self, env: gym.Env, noise_level: float = 1e-5):
        super().__init__(env)

        self.noise_level = noise_level
        self.env = env

    def observation(self, observation: dict) -> dict:
        noisy_beam_observation = observation[
            "beam"
        ] + self.noise_level * np.random.randn(*observation["beam"].shape)

        observation["beam"] = np.clip(
            noisy_beam_observation,
            self.env.observation_space["beam"].low,
            self.env.observation_space["beam"].high,
        )

        return observation

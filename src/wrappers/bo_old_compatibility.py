from functools import partial

import gymnasium as gym
import numpy as np


class BOOldCompatibility(gym.Wrapper):
    """
    Wrapper to wrap the env as a black-box function for running optimization algorithms.

    TODO: In the future this should put all interesting information into the
    observation, but to get BO old working for the next shift, it will conform to
    `ea_bo.py`, keeping the observation the same and putting the objective value into
    `reward`, while also retaining the v2 objective function.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert (
            env.unwrapped.action_mode == "direct"
        ), "The environment must be using the direct action mode."
        assert env.unwrapped.unidirectional_quads, (
            "The old BO implementation is meant to handle only unidrectional"
            " quadruopoles."
        )

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)

        # Weights from old environment use for BO old
        w_beam = 1.0
        w_on_screen = 10.0

        # Compute objective
        on_screen_objective = 1 if info["is_on_screen"] else -1
        beam_objective = self.compute_beam_objective(
            current_beam=observation["beam"], target_beam=observation["target"]
        )

        objective = 0
        objective += w_on_screen * on_screen_objective
        objective += w_beam * beam_objective
        objective = float(objective)

        return observation, objective, terminated, truncated, info

    def compute_beam_objective(
        self,
        current_beam: np.ndarray,
        target_beam: np.ndarray,
        beam_distance_ord: int = 1,
        logarithmic_beam_distance: bool = True,
    ) -> float:
        """Compute objective about the current beam's difference to the target beam."""
        compute_beam_distance = partial(
            self.compute_beam_distance, ord=beam_distance_ord
        )

        # TODO I'm not sure if the order with log is okay this way

        if logarithmic_beam_distance:
            compute_raw_beam_distance = compute_beam_distance
            compute_beam_distance = lambda beam, target_beam: np.log(  # noqa: E731
                compute_raw_beam_distance(beam, target_beam)
            )

        current_distance = compute_beam_distance(current_beam, target_beam)
        beam_objective = -current_distance

        return beam_objective

    def compute_beam_distance(
        self, beam: np.ndarray, target_beam: np.ndarray, ord: int = 2
    ) -> float:
        """
        Compute distance of `beam` to `target_beam`. Eeach beam parameter is weighted by
        its configured weight.
        """
        w_mu_x = 1.0
        w_sigma_x = 1.0
        w_mu_y = 1.0
        w_sigma_y = 1.0

        weights = np.array([w_mu_x, w_sigma_x, w_mu_y, w_sigma_y])
        weighted_current = weights * beam
        weighted_target = weights * target_beam
        return float(np.linalg.norm(weighted_target - weighted_current, ord=ord))

    def close(self):
        super().close()

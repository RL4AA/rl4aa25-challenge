from typing import Union

import cheetah
import gymnasium as gym
import numpy as np


class AnimateIncomingBeam(gym.Wrapper):
    """A wrapper to change the incoming beam parameter gruadually to the new values.

    Args:
        env (gym.Env): The environment to wrap.
        over_n_steps (int): The number of steps over which the incoming beam parameters should be changed.
        to_beam_parameters (np.ndarray): The new incoming beam parameters.
        parameter_indices (np.ndarray, optional): The indices of the incoming beam parameters to change. Defaults to np.arange(11).
            Use [0] for energy, [1] for mu_x, [2] for mu_xp,
            [3] for mu_y, [4] for mu_yp, [5] for sigma_x, [6] for sigma_xp,
            [7] for sigma_y, [8] for sigma_yp, [9] for sigma_s, [10] for sigma_p.
        param_mode (str, optional):
            The mode of the target beam parameters, "direct" or "delta".
        animate_mode (str, optional):
            The mode of animating the incoming beam, "linear" or "sinusoidal".
            Defaults to "linear".
        n_start_steps (int, optional):
            The number of steps to take before animating. Defaults to 0.

    Raises:
        ValueError: If the length of `to_beam_parameters` and `parameter_indices` do not match.
    """

    def __init__(
        self,
        env: gym.Env,
        over_n_steps: int,
        to_beam_parameters: np.ndarray,
        parameter_indices: np.ndarray = np.arange(11),
        param_mode: str = "direct",  # ["direct", "delta"]
        animate_mode: str = "linear",  # ["linear", "sinusoidal"]
        n_start_steps: int = 0,  # Number of steps to take before animating
    ) -> None:
        super().__init__(env)

        self.over_n_steps = over_n_steps
        assert len(to_beam_parameters) == len(parameter_indices)
        self.to_beam_parameters = to_beam_parameters
        self.parameter_indices = parameter_indices
        self.param_mode = param_mode
        self.animate_mode = animate_mode
        self.n_start_steps = n_start_steps

    def reset(self, seed=None, options=None) -> Union[np.ndarray, dict]:
        obs = super().reset(seed=seed, options=options)

        self.steps_taken = 0
        initial_beam = self.env.unwrapped.backend.incoming
        self.initial_beam_parameters = np.array(
            [
                initial_beam.energy,
                initial_beam.mu_x,
                initial_beam.mu_xp,
                initial_beam.mu_y,
                initial_beam.mu_yp,
                initial_beam.sigma_x,
                initial_beam.sigma_xp,
                initial_beam.sigma_y,
                initial_beam.sigma_yp,
                initial_beam.sigma_s,
                initial_beam.sigma_p,
            ]
        )

        # Set the new target beam parameters
        self.target_beam_parameters = np.copy(self.initial_beam_parameters)
        for parameter_index, parameter_value in zip(
            self.parameter_indices, self.to_beam_parameters
        ):
            if self.param_mode == "direct":
                self.target_beam_parameters[parameter_index] = parameter_value
            elif self.param_mode == "delta":
                self.target_beam_parameters[parameter_index] += parameter_value
        return obs

    def step(self, action: np.ndarray) -> tuple:
        self.steps_taken += 1
        self._delta_steps_taken = self.steps_taken - self.n_start_steps
        self._delta_beam_parameters = (
            self.target_beam_parameters - self.initial_beam_parameters
        )
        # Animate the incoming beam
        if self._delta_steps_taken >= 0:
            fraction_done = self._delta_steps_taken / self.over_n_steps
            if self.animate_mode == "linear":
                new_beam_parameters = (
                    self.initial_beam_parameters
                    + self._delta_beam_parameters * fraction_done
                )
            elif self.animate_mode == "sinusoidal":
                new_beam_parameters = (
                    self.initial_beam_parameters
                    + self._delta_beam_parameters * np.sin(np.pi * fraction_done)
                )
            else:
                raise ValueError(
                    "animate_mode must be either 'linear' or 'sinusoidal'."
                )
            self.set_incoming_beam(new_beam_parameters)

        return super().step(action)

    def set_incoming_beam(self, new_beam_parameters: np.ndarray) -> None:
        self.env.unwrapped.backend.incoming = cheetah.ParameterBeam.from_parameters(
            energy=new_beam_parameters[0],
            mu_x=new_beam_parameters[1],
            mu_xp=new_beam_parameters[2],
            mu_y=new_beam_parameters[3],
            mu_yp=new_beam_parameters[4],
            sigma_x=new_beam_parameters[5],
            sigma_xp=new_beam_parameters[6],
            sigma_y=new_beam_parameters[7],
            sigma_yp=new_beam_parameters[8],
            sigma_s=new_beam_parameters[9],
            sigma_p=new_beam_parameters[10],
        )

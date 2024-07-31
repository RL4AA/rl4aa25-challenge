from typing import Union

import cheetah
import gymnasium as gym
import numpy as np


class SetIncomingBeamAtStep(gym.Wrapper):
    """Before the `n`-th step change the incoming beam to `incoming_beam_parameters`."""

    def __init__(
        self, env: gym.Env, steps_to_trigger: int, incoming_beam_parameters: np.ndarray
    ) -> None:
        super().__init__(env)

        # assert isinstance(env.unwrapped.backend, CheetahBackend)

        self.steps_to_trigger = steps_to_trigger
        self.incoming_beam_parameters = incoming_beam_parameters

    def reset(self) -> Union[np.ndarray, dict]:
        self.steps_taken = 0
        self.has_incoming_beam_changed = False

        return super().reset()

    def step(self, action: np.ndarray) -> tuple:
        self.steps_taken += 1
        if (
            self.steps_taken > self.steps_to_trigger
            and not self.has_incoming_beam_changed
        ):
            self.change_incoming_beam()
            self.has_incoming_beam_changed = True
        return super().step(action)

    def change_incoming_beam(self) -> None:
        self.env.unwrapped.backend.incoming = cheetah.ParameterBeam.from_parameters(
            energy=self.incoming_beam_parameters[0],
            mu_x=self.incoming_beam_parameters[1],
            mu_xp=self.incoming_beam_parameters[2],
            mu_y=self.incoming_beam_parameters[3],
            mu_yp=self.incoming_beam_parameters[4],
            sigma_x=self.incoming_beam_parameters[5],
            sigma_xp=self.incoming_beam_parameters[6],
            sigma_y=self.incoming_beam_parameters[7],
            sigma_yp=self.incoming_beam_parameters[8],
            sigma_s=self.incoming_beam_parameters[9],
            sigma_p=self.incoming_beam_parameters[10],
        )

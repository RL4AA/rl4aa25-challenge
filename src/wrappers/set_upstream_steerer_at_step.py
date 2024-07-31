import time
from typing import Union

import gymnasium as gym
import numpy as np


class SetUpstreamSteererAtStep(gym.Wrapper):
    """Before the `n`-th step change the value of an upstream `steerer`."""

    def __init__(
        self, env: gym.Env, steps_to_trigger: int, steerer: str, mrad: float
    ) -> None:
        super().__init__(env)

        global pydoocs
        import pydoocs  # type: ignore

        assert steerer in [
            "ARLIMCHM1",
            "ARLIMCVM1",
            "ARLIMCHM2",
            "ARLIMCVM2",
            "ARLIMSOG1+-",
        ], f"{steerer} is not one of the four upstream steerers"

        self.steps_to_trigger = steps_to_trigger
        self.steerer = steerer
        self.mrad = mrad

    def reset(self) -> Union[np.ndarray, dict]:
        self.steps_taken = 0
        self.is_steerer_set = False

        # Reset steerer to default
        # pydoocs.write(
        #     f"SINBAD.MAGNETS/MAGNET.ML/{self.steerer}/KICK_MRAD.SP", 0.8196
        # )
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/FIELD.SP", -0.1468)

        # Wait until magnets have reached their setpoints

        time.sleep(3.0)  # Wait for magnets to realise they received a command

        is_busy = True
        is_ps_on = True
        while is_busy or not is_ps_on:
            is_busy = pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/BUSY")["data"]
            is_ps_on = pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/PS_ON")[
                "data"
            ]

        return super().reset()

    def step(self, action: np.ndarray) -> tuple:
        self.steps_taken += 1
        if self.steps_taken > self.steps_to_trigger and not self.is_steerer_set:
            print("Triggering disturbance")
            self.set_steerer()
            self.is_steerer_set = True
        return super().step(action)

    def set_steerer(self) -> None:
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/FIELD.SP", self.mrad)

        # Wait until magnets have reached their setpoints

        time.sleep(3.0)  # Wait for magnets to realise they received a command

        is_busy = True
        is_ps_on = True
        while is_busy or not is_ps_on:
            is_busy = pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/BUSY")["data"]
            is_ps_on = pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/PS_ON")[
                "data"
            ]

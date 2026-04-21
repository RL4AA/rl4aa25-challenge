from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .base_backend import TransverseTuningBaseBackend


class TransverseTuning(gym.Env):
    """
    Transverse beam parameter tuning environment for the ARES Experimental Area.

    Magnets: AREAMQZM1, AREAMQZM2, AREAMCVM1, AREAMQZM3, AREAMCHM1
    Screen: AREABSCR1

    :param backend: Backend for communication with either a simulation or the control
        system.
    :param backend_args: Arguments for the backend. NOTE that these may be different
        for different backends.
    :param render_mode: Defines how the environment is rendered according to the
        Gymnasium documentation.
    :param action_mode: Choose weather actions set magnet settings directly (`"direct"`)
        or change magnet settings (`"delta"`).
    :param magnet_init_mode: Magnet initialisation on `reset`. Set to `None` for magnets
        to stay at their current settings, `"random"` to be set to random settings or an
        array of five values to set them to a constant value.
    :param max_quad_setting: Maximum allowed quadrupole setting. The real quadrupoles
        can be set from -72 to 72. These limits are imposed by the power supplies, but
        are unreasonably high to the task at hand. It might therefore make sense to
        choose a lower value.
    :param max_quad_delta: Limit of by how much quadrupole settings may be changed when
        `action_mode` is set to `"delta"`. This parameter is ignored when `action_mode`
        is set to `"direct"`.
    :param max_steerer_setting: Maximum allowed steerer setting. The real steerers can
        be set from -6.1782e-3 to 6.1782e-3. These limits are imposed by the power
        supplies.
    :param max_steerer_delta: Limit of by how much steerer settings may be changed when
        `action_mode` is set to `"delta"`. This parameter is ignored when `action_mode`
        is set to `"direct"`.
    :param target_beam_mode: Setting of target beam on `reset`. Set to "random" to
        generate a random target beam or to an array of four values to set it to a
        constant value.
    :param target_threshold: Distance from target beam parameters at which the episode
        may terminated successfully. Can be a single value or an array of four values
        for (mu_x, sigma_x, mu_y, sigma_y). The estimated accuracy the the screen is
        estimated to be 2e-5 m. Set to `None` to disable early termination.
    :param threshold_hold: Number of steps that all beam parameters difference must be
        below their thresolds before an episode is terminated as successful. A value of
        1 means that the episode is terminated as soon as all beam parameters are below
        their thresholds.
    :param unidirectional_quads: If `True`, quadrupoles are only allowed to be set to
        positive values. This might make learning or optimisation easier.
    :param clip_magnets: If `True`, magnet settings are clipped to their allowed ranges
        after each step.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        backend: Literal["cheetah"] = "cheetah",
        render_mode: Optional[Literal["human", "rgb_array"]] = None,
        action_mode: Literal["direct", "delta"] = "direct",
        magnet_init_mode: Optional[Union[Literal["random"], np.ndarray, list]] = None,
        max_quad_setting: float = 72.0,
        max_quad_delta: Optional[float] = None,
        max_steerer_setting: float = 6.1782e-3,
        max_steerer_delta: Optional[float] = None,
        target_beam_mode: Union[Literal["random"], np.ndarray, list] = "random",
        target_threshold: Optional[Union[float, np.ndarray, list]] = None,
        threshold_hold: int = 1,
        unidirectional_quads: bool = False,
        clip_magnets: bool = True,
        backend_args: dict = {},
    ) -> None:
        self.action_mode = action_mode
        self.magnet_init_mode = magnet_init_mode
        self.max_quad_delta = max_quad_delta
        self.max_steerer_delta = max_steerer_delta
        self.target_beam_mode = target_beam_mode
        self.target_threshold = target_threshold
        self.threshold_hold = threshold_hold
        self.unidirectional_quads = unidirectional_quads
        self.clip_magnets = clip_magnets

        # Create magnet space to be used by observation and action spaces
        if unidirectional_quads:
            self._magnet_space = spaces.Box(
                low=np.array(
                    [
                        0,
                        -max_quad_setting,
                        -max_steerer_setting,
                        0,
                        -max_steerer_setting,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        max_quad_setting,
                        0,
                        max_steerer_setting,
                        max_quad_setting,
                        max_steerer_setting,
                    ],
                    dtype=np.float32,
                ),
            )
        else:
            self._magnet_space = spaces.Box(
                low=np.array(
                    [
                        -max_quad_setting,
                        -max_quad_setting,
                        -max_steerer_setting,
                        -max_quad_setting,
                        -max_steerer_setting,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        max_quad_setting,
                        max_quad_setting,
                        max_steerer_setting,
                        max_quad_setting,
                        max_steerer_setting,
                    ],
                    dtype=np.float32,
                ),
            )

        # Create observation space
        self.observation_space = spaces.Dict(
            {
                "beam": spaces.Box(
                    low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                ),
                "magnets": self._magnet_space,
                "target": spaces.Box(
                    low=np.array([-2e-3, 0, -2e-3, 0], dtype=np.float32),
                    high=np.array([2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float32),
                ),
            }
        )

        # Create action space
        if self.action_mode == "direct":
            self.action_space = self._magnet_space
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array(
                    [
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                    ],
                    dtype=np.float32,
                ),
            )

        # Setup particle simulation or control system backend
        if backend == "cheetah":
            self.backend = CheetahBackend(**backend_args)
        else:
            raise ValueError(f'Invalid value "{backend}" for backend')

        # Utility variables
        self._threshold_counter = 0

        # Setup rendering according to Gymnasium manual
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        env_options, backend_options = self._preprocess_reset_options(options)

        self.backend.reset(options=backend_options)

        if "magnet_init" in env_options:
            self.backend.set_magnets(env_options["magnet_init"])
        elif isinstance(self.magnet_init_mode, (np.ndarray, list)):
            self.backend.set_magnets(self.magnet_init_mode)
        elif self.magnet_init_mode == "random":
            self.backend.set_magnets(self.observation_space["magnets"].sample())
        elif self.magnet_init_mode is None:
            pass  # Yes, his really is intended to do nothing

        if "target_beam" in env_options:
            self._target_beam = env_options["target_beam"]
        elif isinstance(self.target_beam_mode, np.ndarray):
            self._target_beam = self.target_beam_mode
        elif isinstance(self.target_beam_mode, list):
            self._target_beam = np.array(self.target_beam_mode)
        elif self.target_beam_mode == "random":
            self._target_beam = self.observation_space["target"].sample()

        # Update anything in the accelerator (mainly for running simulations)
        self.backend.update()

        # Set reward variables to None, so that _get_reward works properly
        self._beam_reward = None
        self._on_screen_reward = None
        self._magnet_change_reward = None

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._take_action(action)

        self.backend.update()  # Run the simulation

        terminated = self._get_terminated()
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _preprocess_reset_options(self, options: dict) -> tuple[dict, dict]:
        """
        Check that only valid options are passed and split the options into environment
        and backend options.

        NOTE: Backend options are not validated and should be validated by the backend
        itself.
        """
        if options is None:
            return {}, None

        valid_options = ["magnet_init", "target_beam", "backend_options"]
        for option in options:
            assert option in valid_options

        env_options = {k: v for k, v in options.items() if k != "backend_options"}
        backend_options = options.get("backend_options", None)

        return env_options, backend_options

    def _get_terminated(self):
        if self.target_threshold is None:
            return False

        # For readibility in computations below
        cb = self.backend.get_beam_parameters()
        tb = self._target_beam

        # Compute if done (beam within threshold for a certain number of steps)
        is_in_threshold = (np.abs(cb - tb) < self.target_threshold).all()
        self._threshold_counter = self._threshold_counter + 1 if is_in_threshold else 0
        terminated = self._threshold_counter >= self.threshold_hold

        return terminated

    def _get_obs(self):
        return {
            "beam": self.backend.get_beam_parameters().astype("float32"),
            "magnets": self.backend.get_magnets().astype("float32"),
            "target": self._target_beam.astype("float32"),
        }

    def _get_info(self):
        return {
            "binning": self.backend.get_binning(),
            "is_on_screen": self.backend.is_beam_on_screen(),
            "pixel_size": self.backend.get_pixel_size(),
            "screen_resolution": self.backend.get_screen_resolution(),
            "magnet_names": [
                "AREAMQZM1",
                "AREAMQZM2",
                "AREAMCVM1",
                "AREAMQZM3",
                "AREAMCHM1",
            ],
            "screen_name": "AREABSCR1",
            "beam_reward": self._beam_reward,
            "on_screen_reward": self._on_screen_reward,
            "magnet_change_reward": self._magnet_change_reward,
            "max_quad_setting": self.observation_space["magnets"].high[0],
            "backend_info": self.backend.get_info(),  # Info specific to the backend
        }

    def _take_action(self, action: np.ndarray) -> None:
        """Take `action` according to the environment's configuration."""
        self._previous_magnet_settings = self.backend.get_magnets()

        if self.action_mode == "direct":
            new_settings = action
            if self.clip_magnets:
                new_settings = self._clip_magnets_to_power_supply_limits(new_settings)
            self.backend.set_magnets(new_settings)
        elif self.action_mode == "delta":
            new_settings = self._previous_magnet_settings + action
            if self.clip_magnets:
                new_settings = self._clip_magnets_to_power_supply_limits(new_settings)
            self.backend.set_magnets(new_settings)
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

    def _clip_magnets_to_power_supply_limits(self, magnets: np.ndarray) -> np.ndarray:
        """Clip `magnets` to limits imposed by the magnets's power supplies."""
        return np.clip(
            magnets,
            self.observation_space["magnets"].low,
            self.observation_space["magnets"].high,
        )

    def _get_reward(self) -> float:
        """
        vvvvvvvvvvv YOU MAY MODIFY THIS METHOD TO IMPLEMENT YOUR OWN REWARD. vvvvvvvvvvv

        Computes the reward for the current step.

        You can make use of the following information to compute the reward:
         - self.backend.get_beam_parameters(): Returns a NumPy array with the current
              beam parameters (mu_x, sigma_x, mu_y, sigma_y).
            - self._target_beam: NumPy array with the target beam parameters (mu_x,
                sigma_x, mu_y, sigma_y).
            - self.backend.is_beam_on_screen(): Boolean indicating whether the beam is
                on the screen.
            - self.backend.get_magnets(): NumPy array with the current magnet settings
                as (k1_Q1, k1_Q2, angle_CV, k1_Q3, angle_CH).
            - self._previous_magnet_settings: NumPy array with the magnet settings
                before the current action was taken as (k1_Q1, k1_Q2, angle_CV, k1_Q3,
                angle_CH).

        You are allowed to make use of any other information available in the
        environment and backend, if you are so inclined to look through the code.
        """

        reward = 1.0  # Default reward

        return reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        binning = self.backend.get_binning()
        pixel_size = self.backend.get_pixel_size()
        resolution = self.backend.get_screen_resolution()

        # Read screen image and make 8-bit RGB
        img = self.backend.get_screen_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

        # Render beam image as if it were binning = 4
        render_resolution = (resolution * binning / 4).astype("int")
        img = cv2.resize(img, render_resolution)

        # Draw desired ellipse
        tb = self._target_beam
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(tb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(tb[1] / pixel_size_b4[0])
        e_pos_y = int(-tb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(tb[3] / pixel_size_b4[1])
        blue = (255, 204, 79)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, blue, 2
        )

        # Draw beam ellipse
        cb = self.backend.get_beam_parameters()
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(cb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(cb[1] / pixel_size_b4[0])
        e_pos_y = int(-cb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(cb[3] / pixel_size_b4[1])
        red = (0, 0, 255)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, red, 2
        )

        # Adjust aspect ratio from 1:1 pixels to 1:1 physical units on scintillating
        # screen
        new_width = int(img.shape[1] * pixel_size_b4[0] / pixel_size_b4[1])
        img = cv2.resize(img, (new_width, img.shape[0]))

        if self.render_mode == "human":
            cv2.imshow("Transverse Tuning", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.render_mode == "human":
            cv2.destroyWindow("Transverse Tuning")


class CheetahBackend(TransverseTuningBaseBackend):
    """
    Cheetah simulation backend to the ARES Experimental Area.

    :param incoming_mode: Setting for incoming beam parameters on reset. Can be
        `"random"` to generate random parameters or an array of 11 values to set them to
        a constant value.
    :param max_misalignment: Maximum misalignment of magnets and the diagnostic screen
        in meters when `misalignment_mode` is set to `"random"`. This parameter is
        ignored when `misalignment_mode` is set to a constant value.
    :param misalignment_mode: Setting for misalignment of magnets and the diagnostic
        screen on reset. Can be `"random"` to generate random misalignments or an array
        of 8 values to set them to a constant value.
    :param generate_screen_images: If `True`, screen images are generated in every step
        and recorded in the backend info. NOTE that this is very slow and requires a
        lot of memory. It should hence only be used when the images are actually
        needed.
    :param simulate_finite_screen: If `True`, the screen is assumed to be finite and
        false false beam parameters are returned when the beam is not on the screen.
        The false beam parameters are estimates of what would be measured on the real
        screen as a result of the camera vignetting when no beam is visible. NOTE that
        these false beam parameters would always be returned and therefore also be used
        for the reward computation.
    """

    def __init__(
        self,
        incoming_mode: Union[Literal["random"], np.ndarray, list] = "random",
        max_misalignment: float = 5e-4,
        misalignment_mode: Union[Literal["random"], np.ndarray, list] = "random",
        generate_screen_images: bool = False,
        simulate_finite_screen: bool = False,
    ) -> None:
        # Dynamic import for module only required by this backend
        global cheetah
        import cheetah

        if isinstance(incoming_mode, list):
            incoming_mode = np.array(incoming_mode)
        if isinstance(misalignment_mode, list):
            misalignment_mode = np.array(misalignment_mode)

        assert isinstance(incoming_mode, (str, np.ndarray))
        assert isinstance(misalignment_mode, (str, np.ndarray))
        if isinstance(misalignment_mode, np.ndarray):
            assert misalignment_mode.shape == (8,)

        self.incoming_mode = incoming_mode
        self.max_misalignment = max_misalignment
        self.misalignment_mode = misalignment_mode
        self.generate_screen_images = generate_screen_images
        self.simulate_finite_screen = simulate_finite_screen

        # Simulation setup
        self.segment = cheetah.Segment.from_lattice_json(
            Path(__file__).parent / "ea.json"
        )
        self.segment.AREABSCR1.is_active = True

        # Spaces for domain randomisation
        self.incoming_beam_space = spaces.Box(
            low=np.array(
                [
                    80e6,
                    -1e-3,
                    -1e-4,
                    -1e-3,
                    -1e-4,
                    1e-5,
                    1e-6,
                    1e-5,
                    1e-6,
                    1e-6,
                    1e-4,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
                dtype=np.float32,
            ),
        )

        self.misalignment_space = spaces.Box(
            low=-self.max_misalignment, high=self.max_misalignment, shape=(8,)
        )

    def is_beam_on_screen(self) -> bool:
        screen = self.segment.AREABSCR1
        beam_position = np.array(
            [screen.get_read_beam().mu_x, screen.get_read_beam().mu_y]
        )
        limits = np.array(screen.resolution) / 2 * np.array(screen.pixel_size)
        return np.all(np.abs(beam_position) < limits)

    def get_magnets(self) -> np.ndarray:
        return np.array(
            [
                self.segment.AREAMQZM1.k1,
                self.segment.AREAMQZM2.k1,
                self.segment.AREAMCVM1.angle,
                self.segment.AREAMQZM3.k1,
                self.segment.AREAMCHM1.angle,
            ]
        )

    def set_magnets(self, values: Union[np.ndarray, list]) -> None:
        self.segment.AREAMQZM1.k1 = torch.tensor(values[0], dtype=torch.float32)
        self.segment.AREAMQZM2.k1 = torch.tensor(values[1], dtype=torch.float32)
        self.segment.AREAMCVM1.angle = torch.tensor(values[2], dtype=torch.float32)
        self.segment.AREAMQZM3.k1 = torch.tensor(values[3], dtype=torch.float32)
        self.segment.AREAMCHM1.angle = torch.tensor(values[4], dtype=torch.float32)

    def reset(self, options=None) -> None:
        preprocessed_options = self._preprocess_reset_options(options)

        # Set up incoming beam
        if "incoming" in preprocessed_options:
            incoming_parameters = preprocessed_options["incoming"]
        elif isinstance(self.incoming_mode, np.ndarray):
            incoming_parameters = self.incoming_mode
        elif self.incoming_mode == "random":
            incoming_parameters = self.incoming_beam_space.sample()

        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=torch.tensor(incoming_parameters[0]),
            mu_x=torch.tensor(incoming_parameters[1]),
            mu_px=torch.tensor(incoming_parameters[2]),
            mu_y=torch.tensor(incoming_parameters[3]),
            mu_py=torch.tensor(incoming_parameters[4]),
            sigma_x=torch.tensor(incoming_parameters[5]),
            sigma_px=torch.tensor(incoming_parameters[6]),
            sigma_y=torch.tensor(incoming_parameters[7]),
            sigma_py=torch.tensor(incoming_parameters[8]),
            sigma_tau=torch.tensor(incoming_parameters[9]),
            sigma_p=torch.tensor(incoming_parameters[10]),
            dtype=torch.float32,
        )

        # Set up misalignments
        if "misalignments" in preprocessed_options:
            misalignments = preprocessed_options["misalignments"]
        elif isinstance(self.misalignment_mode, np.ndarray):
            misalignments = self.misalignment_mode
        elif self.misalignment_mode == "random":
            misalignments = self.misalignment_space.sample()

        self.segment.AREAMQZM1.misalignment = torch.tensor(
            misalignments[0:2], dtype=torch.float32
        )
        self.segment.AREAMQZM2.misalignment = torch.tensor(
            misalignments[2:4], dtype=torch.float32
        )
        self.segment.AREAMQZM3.misalignment = torch.tensor(
            misalignments[4:6], dtype=torch.float32
        )
        self.segment.AREABSCR1.misalignment = torch.tensor(
            misalignments[6:8], dtype=torch.float32
        )

    def _preprocess_reset_options(self, options: dict) -> dict:
        """
        Check that only valid options are passed and make it a dict if None was passed.
        """
        if options is None:
            return {}

        valid_options = ["incoming", "misalignments"]
        for option in options:
            assert option in valid_options

        return options

    def update(self) -> None:
        self.segment.track(self.incoming)

    def get_beam_parameters(self) -> np.ndarray:
        if self.simulate_finite_screen and not self.is_beam_on_screen():
            return np.array([0, 3.5, 0, 2.2])  # Estimates from real bo_sim data
        else:
            read_beam = self.segment.AREABSCR1.get_read_beam()
            return np.array(
                [read_beam.mu_x, read_beam.sigma_x, read_beam.mu_y, read_beam.sigma_y]
            )

    def get_incoming_parameters(self) -> np.ndarray:
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array(
            [
                self.incoming.energy,
                self.incoming.mu_x,
                self.incoming.mu_px,
                self.incoming.mu_y,
                self.incoming.mu_py,
                self.incoming.sigma_x,
                self.incoming.sigma_px,
                self.incoming.sigma_y,
                self.incoming.sigma_py,
                self.incoming.sigma_tau,
                self.incoming.sigma_p,
            ]
        )

    def get_misalignments(self) -> np.ndarray:
        return np.array(
            [
                self.segment.AREAMQZM1.misalignment[0],
                self.segment.AREAMQZM1.misalignment[1],
                self.segment.AREAMQZM2.misalignment[0],
                self.segment.AREAMQZM2.misalignment[1],
                self.segment.AREAMQZM3.misalignment[0],
                self.segment.AREAMQZM3.misalignment[1],
                self.segment.AREABSCR1.misalignment[0],
                self.segment.AREABSCR1.misalignment[1],
            ],
            dtype=np.float32,
        )

    def get_screen_image(self) -> np.ndarray:
        # Screen image to look like real image by dividing by goodlooking number and
        # scaling to 12 bits)
        return (self.segment.AREABSCR1.reading).numpy() / 1e9 * 2**12

    def get_binning(self) -> np.ndarray:
        return np.array(self.segment.AREABSCR1.binning)

    def get_screen_resolution(self) -> np.ndarray:
        return np.array(self.segment.AREABSCR1.resolution) / self.get_binning()

    def get_pixel_size(self) -> np.ndarray:
        return np.array(self.segment.AREABSCR1.pixel_size) * self.get_binning()

    def get_info(self) -> dict:
        info = {
            "incoming_beam": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }
        if self.generate_screen_images:
            info["screen_image"] = self.get_screen_image()

        return info

import os
import time  # TODO Think about which of these are only used by one backend
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Literal, Optional, Union

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from ..reward import combiners, transforms
from ..type_aliases import CombinerLiteral, TransformLiteral
from .base_backend import TransverseTuningBaseBackend

# TODO Pass incoming as Cheetah beam
# TODO Make sure random seeds are used correctly (incoming, misalignments, etc.)
# TODO Add reset option to generate beam images with Cheeath environment
# TODO Faster dummy pydoocs


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
    :param beam_param_transform: Reward transform for the beam parameters. Can be
        `"Linear"`, `"ClippedLinear"`, `"SoftPlus"`, `"NegExp"` or `"Sigmoid"`.
    :param beam_param_combiner: Reward combiner for the beam parameters. Can be
        `"Mean"`, `"Multiply"`, `"GeometricMean"`, `"Min"`, `"Max"`, `"LNorm"` or
        `"SmoothMax"`.
    :param beam_param_combiner_args: Arguments for the beam parameter combiner. NOTE
        that these may be different for different combiners.
    :param beam_param_combiner_weights: Weights for the beam parameter combiner.
    :param magnet_change_transform: Reward transform for the magnet changes. Can be
        `"Linear"`, `"ClippedLinear"`, `"SoftPlus"`, `"NegExp"` or `"Sigmoid"`.
    :param magnet_change_combiner: Reward combiner for the magnet changes. Can be
        `"Mean"`, `"Multiply"`, `"GeometricMean"`, `"Min"`, `"Max"`, `"LNorm"` or
        `"SmoothMax"`.
    :param magnet_change_combiner_args: Arguments for the magnet change combiner. NOTE
        that these may be different for different combiners.
    :param magnet_change_combiner_weights: Weights for the magnet change combiner.
    :param final_combiner: Reward combiner for the final reward. Can be `"Mean"`,
        `"Multiply"`, `"GeometricMean"`, `"Min"`, `"Max"`, `"LNorm"` or `"SmoothMax"`.
    :param final_combiner_args: Arguments for the final combiner. NOTE that these may
        be different for different combiners.
    :param final_combiner_weights: Weights for the final combiner.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        backend: Literal["cheetah", "doocs", "doocs_dummy"],
        render_mode: Optional[Literal["human", "rgb_array"]] = None,
        action_mode: Literal["direct", "delta"] = "direct",
        magnet_init_mode: Optional[Union[Literal["random"], np.ndarray, list]] = None,
        max_quad_setting: float = 72.0,
        max_quad_delta: Optional[float] = None,
        max_steerer_delta: Optional[float] = None,
        target_beam_mode: Union[Literal["random"], np.ndarray, list] = "random",
        target_threshold: Optional[Union[float, np.ndarray, list]] = None,
        threshold_hold: int = 1,
        unidirectional_quads: bool = False,
        clip_magnets: bool = True,
        beam_param_transform: TransformLiteral = "Sigmoid",
        beam_param_combiner: CombinerLiteral = "GeometricMean",
        beam_param_combiner_args: dict = {},
        beam_param_combiner_weights: list = [1, 1, 1, 1],
        magnet_change_transform: TransformLiteral = "Sigmoid",
        magnet_change_combiner: CombinerLiteral = "Mean",
        magnet_change_combiner_args: dict = {},
        magnet_change_combiner_weights: list = [1, 1, 1, 1, 1],
        final_combiner: CombinerLiteral = "SmoothMax",
        final_combiner_args: dict = {"alpha": -5},
        final_combiner_weights: list = [1, 1, 0.5],
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
                    [0, -max_quad_setting, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32
                ),
                high=np.array(
                    [max_quad_setting, 0, 6.1782e-3, max_quad_setting, 6.1782e-3],
                    dtype=np.float32,
                ),
            )
        else:
            # TODO Have max_quad_setting but don't limit to it
            self._magnet_space = spaces.Box(
                low=np.array(
                    [
                        -max_quad_setting,
                        -max_quad_setting,
                        -6.1782e-3,
                        -max_quad_setting,
                        -6.1782e-3,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        max_quad_setting,
                        max_quad_setting,
                        6.1782e-3,
                        max_quad_setting,
                        6.1782e-3,
                    ],
                    dtype=np.float32,
                ),
            )

        # Create observation space
        self.observation_space = spaces.Dict(
            {
                "beam": spaces.Box(
                    # low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                    # high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                    low=np.array([-5e-3, 0, -5e-3, 0], dtype=np.float32),
                    high=np.array([5e-3, 5e-3, 5e-3, 5e-3], dtype=np.float32),
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

        # Setup reward computation
        beam_param_transform_cls = getattr(transforms, beam_param_transform)
        beam_param_combiner_cls = getattr(combiners, beam_param_combiner)
        magnet_change_transform_cls = getattr(transforms, magnet_change_transform)
        magnet_change_combiner_cls = getattr(combiners, magnet_change_combiner)
        final_combiner_cls = getattr(combiners, final_combiner)

        self._abs_transform = transforms.Abs()
        self._beam_param_transform = beam_param_transform_cls(good=0.0, bad=4e-3)
        self._beam_param_combiner = beam_param_combiner_cls(**beam_param_combiner_args)
        self._magnet_change_transform = magnet_change_transform_cls(good=0.0, bad=1.0)
        self._magnet_change_combiner = magnet_change_combiner_cls(
            **magnet_change_combiner_args
        )
        self._final_combiner = final_combiner_cls(**final_combiner_args)

        self._beam_param_combiner_weights = beam_param_combiner_weights
        self._magnet_change_combiner_weights = magnet_change_combiner_weights
        self._final_combiner_weights = final_combiner_weights

        # Setup particle simulation or control system backend
        if backend == "cheetah":
            self.backend = CheetahBackend(**backend_args)
        elif backend == "doocs_dummy":
            self.backend = DOOCSBackend(use_dummy=True, **backend_args)
        elif backend == "doocs":
            self.backend = DOOCSBackend(use_dummy=False, **backend_args)
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
        self._take_action(action)  # TODO Clip magnets settings

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
        current_beam = self.backend.get_beam_parameters()
        target_beam = self._target_beam
        is_beam_on_screen = self.backend.is_beam_on_screen()
        magnet_changes = (
            self.backend.get_magnets() - self._previous_magnet_settings
        ) / np.maximum(
            np.abs(self.observation_space["magnets"].low),
            np.abs(self.observation_space["magnets"].high),
        )

        self._beam_reward = self._beam_param_combiner(
            self._beam_param_transform(self._abs_transform(current_beam - target_beam)),
            weights=self._beam_param_combiner_weights,
        )
        self._on_screen_reward = 1.0 if is_beam_on_screen else 0.0
        self._magnet_change_reward = self._magnet_change_combiner(
            self._magnet_change_transform(self._abs_transform(magnet_changes)),
            weights=self._magnet_change_combiner_weights,
        )
        reward = self._final_combiner(
            [self._beam_reward, self._on_screen_reward, self._magnet_change_reward],
            weights=self._final_combiner_weights,
        )

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

    @property
    def action_names(self):
        return ["Q1", "Q2", "CV", "Q3", "CH"]

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
        these fasle beam parameters would always be returned and therefore also be used
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
        # the lattice json file is in the same directory as this file
        self.segment = cheetah.Segment.from_lattice_json(
            os.path.join(os.path.dirname(__file__), "ARES_EA_Cheetah_Lattice.json")
        )

        self.segment.AREABSCR1.resolution = torch.tensor((2448, 2040))
        self.segment.AREABSCR1.pixel_size = torch.tensor((3.3198e-6, 2.4469e-6))
        self.segment.AREABSCR1.binning = torch.tensor(1)
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
            energy=torch.tensor(incoming_parameters[0], dtype=torch.float32),
            mu_x=torch.tensor(incoming_parameters[1], dtype=torch.float32),
            mu_xp=torch.tensor(incoming_parameters[2], dtype=torch.float32),
            mu_y=torch.tensor(incoming_parameters[3], dtype=torch.float32),
            mu_yp=torch.tensor(incoming_parameters[4], dtype=torch.float32),
            sigma_x=torch.tensor(incoming_parameters[5], dtype=torch.float32),
            sigma_xp=torch.tensor(incoming_parameters[6], dtype=torch.float32),
            sigma_y=torch.tensor(incoming_parameters[7], dtype=torch.float32),
            sigma_yp=torch.tensor(incoming_parameters[8], dtype=torch.float32),
            sigma_s=torch.tensor(incoming_parameters[9], dtype=torch.float32),
            sigma_p=torch.tensor(incoming_parameters[10], dtype=torch.float32),
        )

        # Set up misalignments
        if "misalignments" in preprocessed_options:
            misalignments = preprocessed_options["misalignments"]
        elif isinstance(self.misalignment_mode, np.ndarray):
            misalignments = self.misalignment_mode
        elif self.misalignment_mode == "random":
            misalignments = self.misalignment_space.sample()

        self.segment.AREAMQZM1.misalignment = torch.as_tensor(misalignments[0:2])
        self.segment.AREAMQZM2.misalignment = torch.as_tensor(misalignments[2:4])
        self.segment.AREAMQZM3.misalignment = torch.as_tensor(misalignments[4:6])
        self.segment.AREABSCR1.misalignment = torch.as_tensor(misalignments[6:8])

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
                self.incoming.mu_xp,
                self.incoming.mu_y,
                self.incoming.mu_yp,
                self.incoming.sigma_x,
                self.incoming.sigma_xp,
                self.incoming.sigma_y,
                self.incoming.sigma_yp,
                self.incoming.sigma_s,
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


class DOOCSBackend(TransverseTuningBaseBackend):
    """
    Backend for the ARES EA to communicate with the real accelerator through the DOOCS
    control system.

    :param use_dummy: If `True`, a dummy backend is used that does not require a
        connection to the real accelerator.
    """

    screen_channel = "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1"
    magnet_channels = [
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK",
    ]

    def __init__(self, use_dummy: bool) -> None:
        # Dynamic import for module only required by this backend
        global pydoocs
        if use_dummy:
            import dummypydoocs as pydoocs
        else:
            import pydoocs  # type: ignore

        self.beam_parameter_compute_failed = {"x": False, "y": False}
        self.reset_accelerator_was_just_called = False

    def is_beam_on_screen(self) -> bool:
        return not all(self.beam_parameter_compute_failed.values())

    def get_magnets(self) -> np.ndarray:
        return np.array(
            [
                pydoocs.read(f"{magnet_channel}.RBV")["data"]
                for magnet_channel in self.magnet_channels
            ]
        )

    def set_magnets(self, values: Union[np.ndarray, list]) -> None:
        with ThreadPoolExecutor(max_workers=len(self.magnet_channels)) as executor:
            for result in executor.map(self.set_magnet, self.magnet_channels, values):
                x = result  # noqa: F841

    def set_magnet(self, channel: str, value: float) -> None:
        """
        Set the value of a certain magnet. Returns only when the magnet has arrived at
        the set point.
        """
        setpoint_channel = channel + ".SP"
        busy_channel = channel.replace("STRENGTH", "BUSY").replace("KICK", "BUSY")
        ps_on_channel = channel.replace("STRENGTH", "PS_ON").replace("KICK", "PS_ON")

        pydoocs.write(setpoint_channel, value)
        time_of_write = datetime.now()

        time.sleep(3.0)  # Give magnets time to receive the command

        seconds_before_reanimation = 60
        is_busy = True
        is_ps_on = False
        while is_busy or not is_ps_on:
            is_busy = pydoocs.read(busy_channel)["data"]
            is_ps_on = pydoocs.read(ps_on_channel)["data"]

            time.sleep(0.1)

            # If the magnet is not responding, "wiggle" it back to life
            if datetime.now() - time_of_write > timedelta(
                seconds=seconds_before_reanimation
            ):
                # Don't wiggle steerers, just quadrupoles
                if "KICK" in channel:
                    continue

                self.try_to_reanimate_quadrupole(channel, value)

                time_of_write = datetime.now()  # Reset time of write
                seconds_before_reanimation = seconds_before_reanimation * 2

    def try_to_reanimate_quadrupole(self, channel: str, value: float) -> None:
        """
        The quadrupole magnets at ARES can freeze under certain conditions. This
        function attempts to unfreeze a quadrupole by first turning its power supply off
        and on again and then wiggling the setpoint up and down by 0.2 (k1).
        """
        setpoint_channel = channel + ".SP"
        readback_channel = channel + ".RBV"
        ps_on_channel = channel.replace("STRENGTH", "PS_ON").replace("KICK", "PS_ON")
        busy_channel = channel.replace("STRENGTH", "BUSY").replace("KICK", "BUSY")

        print(f"WARNING {datetime.now()}: Trying to reanimate {channel}.")

        # Turn off and on again (yes ... turn it off when it is off ...)
        print(f"    -> Truning off ({ps_on_channel} / {datetime.now()})")
        pydoocs.write(ps_on_channel, 0)
        is_ps_on = True
        readback = pydoocs.read(readback_channel)["data"]
        last_turn_off_time = datetime.now()
        while is_ps_on or np.abs(readback) > 0.3:
            time.sleep(0.3)
            is_ps_on = pydoocs.read(ps_on_channel)["data"]
            readback = pydoocs.read(readback_channel)["data"]
            if datetime.now() - last_turn_off_time > timedelta(seconds=120):
                print(f"        -> Trying to turn off AGAIN ({datetime.now()})")
                pydoocs.write(ps_on_channel, 0)
                last_turn_off_time = datetime.now()
        time.sleep(4.0)

        print(f"    -> Setpoint to 0.0 ({setpoint_channel} / {datetime.now()})")
        pydoocs.write(setpoint_channel, 0.0)
        time.sleep(10.0)

        print(f"    -> Turning back on ({ps_on_channel} / {datetime.now()})")
        pydoocs.write(ps_on_channel, 1)
        is_ps_on = False
        last_turn_on_time = datetime.now()
        while not is_ps_on:
            time.sleep(0.3)
            is_ps_on = pydoocs.read(ps_on_channel)["data"]
            if datetime.now() - last_turn_on_time > timedelta(seconds=60):
                print(f"        -> Trying to turn on AGAIN ({datetime.now()})")
                pydoocs.write(ps_on_channel, 1)
                last_turn_on_time = datetime.now()
        time.sleep(4.0)

        wiggle_value = value + np.sign(value) * 35.0
        print(
            f"    -> Wiggling to value {wiggle_value} ({setpoint_channel} /"
            f" {datetime.now()})"
        )
        pydoocs.write(setpoint_channel, wiggle_value)
        is_busy = True
        while is_busy:  # or not is_idle or not is_close:
            is_busy = pydoocs.read(busy_channel)["data"]

            time.sleep(0.5)

        print(
            f"    -> Returning from wiggle to {value} ({setpoint_channel} /"
            f" {datetime.now()})"
        )
        pydoocs.write(setpoint_channel, value)
        time.sleep(10.0)

    def reset(self, options=None) -> None:
        preprocessed_options = self._preprocess_reset_options(options)  # noqa: F841

        self.update()

        self.magnets_before_reset = self.get_magnets()
        self.screen_before_reset = self.get_screen_image()
        self.beam_before_reset = self.get_beam_parameters()

        # In order to record a screen image right after the accelerator was reset, this
        # flag is set so that we know to record the image the next time
        # `update_accelerator` is called.
        self.reset_accelerator_was_just_called = True

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

    def update(self):
        self.screen_image = self.capture_clean_screen_image()

        # Record the beam image just after reset (because there is no info on reset).
        # It will be included in `info` of the next step.
        if self.reset_accelerator_was_just_called:
            self.screen_after_reset = self.screen_image
            self.reset_accelerator_was_just_called = False

    def get_beam_parameters(self):
        img = self.get_screen_image()
        pixel_size = self.get_pixel_size()
        resolution = self.get_screen_resolution()

        parameters = {}
        for axis, direction in zip([0, 1], ["x", "y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(
                minfiltered, size=5, mode="nearest"
            )  # TODO rethink filters

            (half_values,) = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2

                # If (almost) all pixels are in FWHM, the beam might not be on screen
                self.beam_parameter_compute_failed[direction] = (
                    len(half_values) > 0.95 * resolution[axis]
                )
            else:
                fwhm_pixel = 42  # TODO figure out what to do with these
                center_pixel = 42

            parameters[f"mu_{direction}"] = (
                center_pixel - len(filtered) / 2
            ) * pixel_size[axis]
            parameters[f"sigma_{direction}"] = fwhm_pixel / 2.355 * pixel_size[axis]

        parameters["mu_y"] = -parameters["mu_y"]

        return np.array(
            [
                parameters["mu_x"],
                parameters["sigma_x"],
                parameters["mu_y"],
                parameters["sigma_y"],
            ]
        )

    def get_screen_image(self):
        return self.screen_image

    def get_binning(self):
        horizontal_binning_channel = self.screen_channel + "/BINNINGHORIZONTAL"
        vertical_binning_channel = self.screen_channel + "/BINNINGVERTICAL"
        return np.array(
            [
                pydoocs.read(horizontal_binning_channel)["data"],
                pydoocs.read(vertical_binning_channel)["data"],
            ]
        )

    def get_screen_resolution(self):
        width_channel = self.screen_channel + "/WIDTH"
        height_channel = self.screen_channel + "/HEIGHT"
        return np.array(
            [pydoocs.read(width_channel)["data"], pydoocs.read(height_channel)["data"]]
        )

    def get_pixel_size(self):
        x_pixel_size_channel = self.screen_channel + "/X.POLY_SCALE"
        y_pixel_size_channel = self.screen_channel + "/Y.POLY_SCALE"
        return (
            np.array(
                [
                    abs(pydoocs.read(x_pixel_size_channel)["data"][2]) / 1000,
                    abs(pydoocs.read(y_pixel_size_channel)["data"][2]) / 1000,
                ]
            )
            * self.get_binning()
        )

    def capture_clean_screen_image(self, average=5):
        """
        Capture a clean image of the beam from the screen using `average` images with
        beam on and `average` images of the background and then removing the background.

        Saves the image to a property of the object.
        """
        # Laser off
        self.set_cathode_laser(False)
        background_images = self.capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self.set_cathode_laser(True)
        screen_images = self.capture_interval(n=average, dt=0.1)
        median_beam = np.median(screen_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16 - 1)
        flipped = np.flipud(removed)

        return flipped.astype(np.uint16)

    def capture_interval(self, n, dt):
        """Capture `n` images from the screen and wait `dt` seconds in between them."""
        images = []
        for _ in range(n):
            images.append(self.capture_screen())
            time.sleep(dt)
        return np.array(images)

    def capture_screen(self):
        """Capture and image from the screen."""
        screen_image_channel = self.screen_channel + "/IMAGE_EXT_ZMQ"
        return pydoocs.read(screen_image_channel)["data"]

    def set_cathode_laser(self, setto: bool) -> None:
        """
        Sets the bool switch of the cathode laser event to `setto` and waits a second.
        """
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)

    def get_info(self) -> dict:
        # If magnets or the beam were recorded before reset, add them info on the first
        # step, so a generalised data recording wrapper captures them.
        info = {}

        # Screen image
        info["screen_image"] = self.get_screen_image()

        if hasattr(self, "magnets_before_reset"):
            info["magnets_before_reset"] = self.magnets_before_reset
            del self.magnets_before_reset
        if hasattr(self, "screen_before_reset"):
            info["screen_before_reset"] = self.screen_before_reset
            del self.screen_before_reset
        if hasattr(self, "beam_before_reset"):
            info["beam_before_reset"] = self.beam_before_reset
            del self.beam_before_reset

        if hasattr(self, "screen_after_reset"):
            info["screen_after_reset"] = self.screen_after_reset
            del self.screen_after_reset

        # Gain of camera for AREABSCR1
        camera_gain_channel = self.screen_channel + "/GAINRAW"
        info["camera_gain"] = pydoocs.read(camera_gain_channel)["data"]

        # Steerers upstream of Experimental Area
        for steerer in ["ARLIMCHM1", "ARLIMCVM1", "ARLIMCHM2", "ARLIMCVM2"]:
            response = pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{steerer}/KICK.RBV")
            info[steerer] = response["data"]

        # Gun solenoid
        info["gun_solenoid"] = pydoocs.read(
            "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/FIELD.RBV"
        )["data"]

        return info

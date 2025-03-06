import asyncio
import json
import logging
import os
import random
from collections import OrderedDict
from typing import Any, Dict, Optional, Set, Tuple

import cheetah
import gymnasium as gym
import numpy as np
import torch

from cheetah import Segment
from cheetah.utils.segment_3d_builder import Segment3DBuilder
from gymnasium import spaces

import ARESlatticeStage3v1_9 as ares_lattice
from rewarder import Rewarder

# Set logging level based on environment
debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Setup logging
logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
logger = logging.getLogger(__name__)


class BeamControlEnv(gym.Env):
    """
    A reinforcement learning environment for controlling a particle beam
    through an accelerator lattice. The environment consists of a sequence
    of beam segments, each allowing for fine-tuned control over beam
    alignment and focusing through an action space.

    Attributes:
        - ocelot_lattice (ARESLattice): Lattice structure for the accelerator.
        - lattice_segment (cheetah.Segment): Main segment of the beamline lattice.
        - beam_parameter_space (spaces.Box): Space defining the beam's initial parameters.
        - observation_space (spaces.Box): Space defining the range of observations.
        - action_space (spaces.Box): Space defining valid actions.
        - current_step (int): Counter for the current step in the episode.
        - target_distance (float): Target distance for beam alignment.
        - reward (float): Accumulated reward during the episodes
        - seed (int): This method sets seeds for numpy, random, and torch (for CPU and GPU)
                      to ensure reproducibility across random operations.

    Methods:
        - __init__: Initializes the environment, lattice structure, and WebSocket connection.
        - reset: Resets the environment to an initial state and returns the initial observation.
        - step: Takes an action, simulates beam dynamics, and returns the new state,
           reward, and info.
        - _simulate: Updates segment properties and tracks the beam through the lattice.
        - _reward: Calculates and returns the reward based on beam alignment and focusing.
        - _apply_magnet_settings: Applies new magnet settings based on action input.
        - _get_obs: Generates observations from the beam state.
        - _is_terminal: Checks if the current episode has reached a terminal state.
        - _is_beam_on_screen: Checks whether the beam position
           is within the visible screen boundaries.
    """

    def __init__(self, config: dict, websocket_manager=None) -> None:
        """
        Initialize BeamControlEnv.

        Args:
            config (dict): Configuration parameters for environment, training, and reward.
        """
        super(BeamControlEnv, self).__init__()

        # Basic Environment Configuration
        self.config = config
        self.seed = self.config["env_config"].get("seed", None)
        self.max_episode_steps = self.config["env_config"]["max_episode_steps"]
        self.render_mode = self.config["env_config"].get("render_mode", "human")
        self.host = self.config["env_config"].get("host", "localhost")
        self.port = self.config["env_config"].get("port", 8081)

        # WebSocket Manager
        self.websocket_manager = websocket_manager  # External WebSocket handler

        # Setup the main beamline lattice segment
        self.lattice_segment = cheetah.Segment.from_ocelot(
           ares_lattice.cell, warnings=False, device="cpu"
        ).subcell(
           "AREASOLA1", "AREABSCR1"
        )  # Add the section of focus

        self.lattice_segment.AREABSCR1.is_active = True  # Activate screen
        self.lattice_segment.AREABSCR1.binning = 1

        self.builder = Segment3DBuilder(self.lattice_segment)

        # Build and export the 3D scene
        self.builder.build_segment(
            output_filename="public/models/ares/scene.glb",
            is_export_enabled=False
        )

        # Track lattice component positions
        self.component_positions = torch.tensor(list(self.builder.component_positions.values()), dtype=torch.float32)

        # Define and generate lattice segments
        self.segments = OrderedDict()

        segment_definitions = {
            "AREAMQZM1": {"start": "AREASOLA1", "end": "AREAMQZM1"},
            "AREAMQZM2": {"start": "AREAMQZM1", "end": "AREAMQZM2"},
            "AREAMCVM1": {"start": "AREAMQZM2", "end": "AREAMCVM1"},
            "AREAMQZM3": {"start": "AREAMCVM1", "end": "AREAMQZM3"},
            "AREAMCHM1": {"start": "AREAMQZM3", "end": "AREAMCHM1"},
            "AREABSCR1": {"start": "AREAMCHM1", "end": "AREABSCR1"},
        }

        for segment_name, segment_info in segment_definitions.items():
            self.segments[segment_name] = self.lattice_segment.subcell(
                segment_info["start"], 
                segment_info["end"]
        )

        # Define screen
        self.screen_name = "AREABSCR1"
        self.screen = getattr(self.lattice_segment, self.screen_name)
        self.screen_resolution = ((2448, 2040),)
        self.screen_pixel_size = ((3.3198e-6, 2.4469e-6),)
        self.screen.binning = 1
        self.screen.is_active = True

        # Obtain screen Boundaries
        # The resolution of the screen in pixels (width, height),
        # i.e. 2448, 2040
        # The physical size of a single pixel in meters (width, height),
        # i.e. (3.5488e-06, 2.5003e-06
        self.screen_boundary = (
            self.get_screen_boundary()
        )  # e.g.  [0.00434373, 0.00255031]

        # Reset the tracking flag at the start of each episode
        self.beam_has_been_on_screen = False

        # Initialize screen reading
        self.screen_reading = np.zeros(
            tuple(self.lattice_segment.AREABSCR1.resolution), dtype=np.int32
        )
        # Beam parameter space
        self.beam_parameter_space = spaces.Box(
            low=np.array(
                [80e6, -1e-3, -1e-4, -1e-3, -1e-4, 1e-5, 1e-6, 1e-5, 1e-6, 1e-6, 1e-4],
                dtype=np.float32,
            ),
            high=np.array(
                [160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
                dtype=np.float32,
            ),
            shape=(11,),
            dtype=np.float32,
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-1e-3, -1e-3, 1e-5, 1e-5], dtype=np.float32),
            high=np.array([1e-3, 1e-3, 5e-4, 5e-4], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        # Action space
        # Quadrupoles have a range from -72 to 72 1/(m^2). (in operation we would not go above 30)
        # Steerers can range from -6.1782e-3 rad to 6.1782e-3 rad.
        self.action_space = spaces.Box(
            low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
            high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )

        # Control and Reward Initialization
        self.control_action = None
        self.reward_signals = (
            {}
        )  # Breakdown of reward components (e.g., {'beam_alignment': 0.5, 'beam_focus': 0.3})
        for reward_name, reward_config in self.config["reward_signals"].items():
            self.reward_signals[reward_name] = reward_config["weight"]

        self.rewarder = Rewarder(self.reward_signals)

        self.current_step = 0  # Episode step counter
        self.beam_has_been_on_screen = False
        self.info = {
            "component_positions": self.component_positions.tolist()
        }

        # Sample an ellipsoid beam distribution with attributes
        # (x, px, y, py, tau, sigma, 1) with N particles
        self.num_particles = 1_000 # Default: 100, 1_000, or 10_000
        self.incoming_particle_beam = None

        # Define visual scaling factors (s_x, s_y), estimated manually
        self.scale_x = 10  # 45  # simulated_width / physical_width
        self.scale_y = 10  # 30  # simulated_height / physical_height
        self.position_scale_factor = torch.tensor([
            self.scale_x, self.scale_y, 1
            ],
            dtype=torch.float32
        )  # z-direction does not get scaled
        self.beam_width_scale_factor = 1  # Scale beam width (default: 25, alt: 100000)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state and return an initial observation.

        Args:
            seed (Optional[int]): Seed for random number generation. Default is None.
            options (Optional[dict]): Additional options for reset. Default is None.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
        """
        # Reset episode state and termination flags
        self.current_step = 0    # Reset timestep counter
        self.terminated = False  # Natural episode end (e.g., task completion/failure)
        self.truncated = False   # Forced episode end (e.g., max steps reached)

        # Reset episode state
        self.current_step = 0  # Reset timestep counter

        # Reset reward state
        self.reward = 0.0  # Clear total step reward

        # Set the seed if provided
        if seed is None and self.seed is not None:
            seed = self.seed

        if seed is not None:
            self._set_global_seed(seed)

        # Sample initial parameters for the beam to be controlled
        self.beam_initial_parameters = self.beam_parameter_space.sample()

        # Sample initial parameters for the target (reference) beam
        self.target_beam_initial_parameters = self.beam_parameter_space.sample()

        # Create incoming beam using the initial parameters
        #self.incoming_beam = cheetah.ParticleBeam.from_parameters(
        #    energy=torch.tensor(self.beam_initial_parameters[0], dtype=torch.float32),
        #    mu_x=torch.tensor(self.beam_initial_parameters[1], dtype=torch.float32),
        #    mu_px=torch.tensor(self.beam_initial_parameters[2], dtype=torch.float32),
        #    mu_y=torch.tensor(self.beam_initial_parameters[3], dtype=torch.float32),
        #    mu_py=torch.tensor(self.beam_initial_parameters[4], dtype=torch.float32),
        #    sigma_x=torch.tensor(self.beam_initial_parameters[5], dtype=torch.float32),
        #    sigma_px=torch.tensor(self.beam_initial_parameters[6], dtype=torch.float32),
        #    sigma_y=torch.tensor(self.beam_initial_parameters[7], dtype=torch.float32),
        #    sigma_py=torch.tensor(self.beam_initial_parameters[8], dtype=torch.float32),
        #    sigma_tau=torch.tensor(
        #        self.beam_initial_parameters[9], dtype=torch.float32
        #    ),
        #    sigma_p=torch.tensor(self.beam_initial_parameters[10], dtype=torch.float32),
        #)

        self.incoming_beam = cheetah.ParticleBeam.from_astra(
            os.path.join(os.getcwd(),
            "simulation_controller",
            "resources/ACHIP_EA1_2021.1351.001"
            ),
            device="cpu",
        )

        # Create uniform particle beam (based-on random initial beam parameters)
        #self.incoming_particle_beam = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        #    num_particles=self.num_particles,
        #    radius_x=torch.tensor(0.001, dtype=torch.float32),    # Default: 0.01
        #    radius_y=torch.tensor(0.001, dtype=torch.float32),    # Default: 0.01
        #    radius_tau=torch.tensor(0.002, dtype=torch.float32),  # Default: 0.02 (radius of the beam in s-direction in the lab frame)
        #    sigma_px=torch.tensor(self.beam_initial_parameters[6], dtype=torch.float32),
        #    sigma_py=torch.tensor(self.beam_initial_parameters[8], dtype=torch.float32),
        #    sigma_p=torch.tensor(self.beam_initial_parameters[10], dtype=torch.float32),
        #    energy=torch.tensor(self.beam_initial_parameters[0], dtype=torch.float32),
        #    dtype=torch.float32,
        #    device="cpu",
        #)

        # TODO: Newly introduced to merge the pre-defined beam and the particle distribution beam
        #self.incoming_particle_beam.transformed_to(
        #    energy=self.incoming_beam.clone().float(),
        #    mu_x=self.incoming_beam.mu_x.clone().float(),
        #    mu_px=self.incoming_beam.mu_px.clone().float(),
        #    mu_y=self.incoming_beam.mu_y.clone().float(),
        #    mu_py=self.incoming_beam.mu_py.clone().float(),
        #    sigma_x=self.incoming_beam.sigma_x.clone().float(),
        #    sigma_px=self.incoming_beam.sigma_px.clone().float(),
        #    sigma_y=self.incoming_beam.sigma_y.clone().float(),
        #    sigma_py=self.incoming_beam.sigma_py.clone().float(),
        #    sigma_p=self.incoming_beam.sigma_p.clone().float(),
        #)

        self.incoming_particle_beam = cheetah.ParticleBeam.uniform_3d_ellipsoid(
            num_particles=self.num_particles,
            radius_x=torch.tensor(0.001, dtype=torch.float32),    # Default: 0.01
            radius_y=torch.tensor(0.001, dtype=torch.float32),    # Default: 0.01
            radius_tau=torch.tensor(0.002, dtype=torch.float32),  # Default: 0.02 (radius of the beam in s-direction in the lab frame)
            sigma_px=self.incoming_beam.sigma_px.clone().to('cpu').float(),
            sigma_py=self.incoming_beam.sigma_py.clone().to('cpu').float(),
            sigma_p=self.incoming_beam.sigma_p.clone().to('cpu').float(),
            energy=self.incoming_beam.energy.clone().to('cpu').float(),
            dtype=torch.float32,
            device="cpu",
        )

        # Track the lattice with the incoming beam
        self._simulate()

        # Get beam position and observation
        observation = self._get_obs()

        # Set target beam parameters
        self.target_beam = cheetah.ParameterBeam.from_parameters(
            energy=torch.tensor(
                self.target_beam_initial_parameters[0], dtype=torch.float32
            ),
            mu_x=torch.tensor(
                self.target_beam_initial_parameters[1], dtype=torch.float32
            ),
            mu_px=torch.tensor(
                self.target_beam_initial_parameters[2], dtype=torch.float32
            ),
            mu_y=torch.tensor(
                self.target_beam_initial_parameters[3], dtype=torch.float32
            ),
            mu_py=torch.tensor(
                self.target_beam_initial_parameters[4], dtype=torch.float32
            ),
            sigma_x=torch.tensor(
                self.target_beam_initial_parameters[5], dtype=torch.float32
            ),
            sigma_px=torch.tensor(
                self.target_beam_initial_parameters[6], dtype=torch.float32
            ),
            sigma_y=torch.tensor(
                self.target_beam_initial_parameters[7], dtype=torch.float32
            ),
            sigma_py=torch.tensor(
                self.target_beam_initial_parameters[8], dtype=torch.float32
            ),
            sigma_tau=torch.tensor(
                self.target_beam_initial_parameters[9], dtype=torch.float32
            ),
            sigma_p=torch.tensor(
                self.target_beam_initial_parameters[10], dtype=torch.float32
            ),
        )

        # Set target beam observation
        self.target_beam_observation = np.array(
            [
                self.target_beam.mu_x,
                self.target_beam.mu_y,
                self.target_beam.sigma_x,
                self.target_beam.sigma_y,
            ],
            dtype=np.float32,
        )

        # Update info dictionary with environment state
        self.info.update({
            "screen_boundary_x": float(self.screen_boundary[0]),
            "screen_boundary_y": float(self.screen_boundary[1]),
            "beam_has_been_on_screen": self.beam_has_been_on_screen,
            "reward": float(self.reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "mu_x": observation[0].item(),
            "mu_y": observation[1].item(),
            "sigma_x": observation[2].item(),
            "sigma_y": observation[3].item(),
            "bunch_count": self.current_step + 1,
        })

        return observation, self.info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a step in the environment based on the action provided.

        Args:
            action (np.ndarray): Array representing the action to be taken.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: Observation,
                reward, termination status, truncation status, and info dictionary.
        """
        # Get previous beam state
        previous_observation = self._get_obs()

        # Ensure actions are within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Simulate the beam dynamics through the lattice
        self._simulate(action)

        # Increment step count immediately after advancing the environment
        # and before checking termination conditions
        self.current_step += 1

        # Obtain current beam position and state
        observation = self._get_obs()

        # Compute reward
        self.reward = self._compute_reward(observation, previous_observation)

        # Update termination status
        self._is_terminal()

        # Update info dictionary with environment state
        self.info.update({
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "beam_has_been_on_screen": self.beam_has_been_on_screen,
            "distance_to_target": float(
                np.linalg.norm(observation - self.target_beam_observation)
            ),
            "mu_x": observation[0].item(),
            "mu_y": observation[1].item(),
            "sigma_x": observation[2].item(),
            "sigma_y": observation[3].item(),
            "bunch_count": self.current_step + 1,
        })

        return observation, self.reward, self.terminated, self.truncated, self.info

    def close(self) -> None:
        """
        Closes the WebSocket server and all client connections.
        """
        if self.websocket_manager:
            self.websocket_manager.close()
        super().close()

    def is_beam_on_screen(self) -> bool:
        """
        Determines whether the beam position is within the visible screen boundaries.

        This method performs the following steps:
        1. Retrieves the current beam state.
        2. Checks if the beam's position is within the screen boundaries.
        3. Updates the tracking flag if the beam enters the screen for the first time.

        Returns:
            bool: True if the beam is within the screen boundaries, False otherwise.
        """
        # Get current beam position
        read_beam = self.screen.get_read_beam()
        beam_position = np.array(
            [read_beam.mu_x.item(), read_beam.mu_y.item()]
        )

        # Check if the beam position is within the screen boundaries
        self.screen_boundary = self.get_screen_boundary()
        is_on_screen = np.all(np.abs(beam_position) < self.screen_boundary)

        # Update tracking flag if beam enters screen for the first time
        if is_on_screen and not self.beam_has_been_on_screen:
            self.beam_has_been_on_screen = True

        return is_on_screen

    async def render(self) -> None:
        """
        Renders simulation information and sends it via WebSocket to clients.
        """
        # Send the data to make it accessible to the visualizer
        if self.render_mode == "human" and self.websocket_manager:
            # Broadcast (i.e. sending) beam data to all connected WebSocket clients
            await self.websocket_manager.broadcast(self.info)
            # Add delay after broadcasting to allow animation to complete before sending new
            await asyncio.sleep(10.0)  # Adjust delay as needed (e.g., 250ms = 0.25s or 20.0s or 0.226s)

    def _set_global_seed(self, seed: int) -> None:
        """
        Set seed for all relevant libraries to ensure reproducibility.

        Args:
            seed (int): The seed to set.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Seed the environment's random number generator
        self.beam_parameter_space.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation from the environment's state.

        Returns:
            np.ndarray: Array of observations for the environment.
        """
        beam = self.lattice_segment.AREABSCR1.get_read_beam()

        return np.array(
            [
                beam.mu_x.item(),
                beam.mu_y.item(),
                beam.sigma_x.item(),
                beam.sigma_y.item(),
            ],
            dtype=np.float32,
        )

    def _compute_reward(
        self, observation: np.ndarray, previous_observation: np.ndarray
    ) -> float:
        """
        Computes the reward based on the beam's position and movement.

        Args:
            observation (np.ndarray): Current beam state.
            previous_observation (np.ndarray): Previous beam state.

        Returns:
            float: Reward value.
        """
        # Define target and secondary radii
        radius = 1e-5 * np.sqrt(2)
        secondary_radius = 2e-3

        # Check if the beam position is within the target beam
        is_within_target_area = bool(np.linalg.norm(observation[:2]) <= radius)

        # Reward shaping for successfully reaching the target area
        if is_within_target_area:
            return 100  # High reward for reaching the target
        elif np.linalg.norm(observation[:2]) <= secondary_radius:
            # Reward shaping for getting close
            return 10 * self.rewarder.compute_reward(
                observation,
                previous_observation,
                self.target_beam_observation,
                self.screen_boundary[0],  # half the height of diagnostic screen
                self.screen_boundary[1],  # half the width of diagnostic screen
            )
        elif self.is_beam_on_screen():
            # Reward shaping for being on screen but outside the target
            return self.rewarder.compute_reward(
                observation,
                previous_observation,
                self.target_beam_observation,
                self.screen_boundary[0],  # half the height of diagnostic screen
                self.screen_boundary[1],  # half width of diagnostic screen
            )
        else:
            return -1  # Penalty for going off-screen

        # Update info with reward metrics
        self.info.update(self.rewarder.get_info())

    def _is_terminal(self) -> None:
        """
        Checks if the current episode has reached a terminal state.

        This method determines whether the episode should end based on two conditions:
        1. Termination: The beam has appeared on screen at least once and is now off screen.
        2. Truncation: The maximum number of steps for the episode has been reached.

        The method updates the 'terminated' and 'truncated' flags accordingly and
        adds a 'termination_reason' to the info dictionary for logging purposes.
        """
        # Check for termination: beam has been on screen but is now off screen
        self.terminated = self.beam_has_been_on_screen and not self.is_beam_on_screen()

        # Check for truncation: maximum steps exceeded
        self.truncated = bool(self.current_step >= self.max_episode_steps)

        # Update info dictionary with termination reason
        if self.terminated:
            self.info["termination_reason"] = "beam_left_screen"
        elif self.truncated:
            self.info["termination_reason"] = "max_steps_reached"

    def _apply_magnet_settings(self, action: np.ndarray) -> None:
        """
        Apply new magnet settings to the lattice segments.

        Args:
            action (np.ndarray): Array of action values to set the magnetic field parameters.
        """
        # Apply delta adjustments to the magnet settings
        self.lattice_segment.AREAMQZM1.k1 = torch.tensor(
            action[0], dtype=torch.float32
        )  # Quadrupole
        self.lattice_segment.AREAMQZM2.k1 = torch.tensor(
            action[1], dtype=torch.float32
        )  # Quadrupole
        self.lattice_segment.AREAMCVM1.angle = torch.tensor(
            action[2], dtype=torch.float32
        )  # Steer
        self.lattice_segment.AREAMQZM3.k1 = torch.tensor(
            action[3], dtype=torch.float32
        )  # Quadrupole
        self.lattice_segment.AREAMCHM1.angle = torch.tensor(
            action[4], dtype=torch.float32
        )  # Steer

        # Clip the values using the specified range
        self.lattice_segment.AREAMQZM1.k1 = np.clip(
            self.lattice_segment.AREAMQZM1.k1,
            self.action_space.low[0],
            self.action_space.high[0],
        )
        self.lattice_segment.AREAMQZM2.k1 = np.clip(
            self.lattice_segment.AREAMQZM2.k1,
            self.action_space.low[1],
            self.action_space.high[1],
        )
        self.lattice_segment.AREAMCVM1.angle = np.clip(
            self.lattice_segment.AREAMCVM1.angle,
            self.action_space.low[2],
            self.action_space.high[2],
        )
        self.lattice_segment.AREAMQZM3.k1 = np.clip(
            self.lattice_segment.AREAMQZM3.k1,
            self.action_space.low[3],
            self.action_space.high[3],
        )
        self.lattice_segment.AREAMCHM1.angle = np.clip(
            self.lattice_segment.AREAMCHM1.angle,
            self.action_space.low[4],
            self.action_space.high[4],
        )

    def _simulate(self, action: Optional[np.ndarray] = None) -> None:
        """
        Calculate the positions of beam segments with dynamic angles
        Beam travels along x-axis, with position variations in yz-plane
        Allows for more dynamic z-axis movement
        """
        # Apply the new magnet settings
        if action is not None:
            self._apply_magnet_settings(action)

        # Data to be used to send data over WebSocket
        data = OrderedDict()

        lattice_sub_segment = self.segments["AREAMQZM1"]

        #incoming_beam = self.incoming_beam
        incoming_beam = self.incoming_particle_beam

        # Stack the x, y, and modified z positions for each particle
        positions = torch.stack([
            incoming_beam.x,
            incoming_beam.y,
            incoming_beam.tau + self.component_positions[0]
        ], dim=-1)  # dim=-1 stacks along the last dimension

        data["segment_0"] = {
            "segment_name": "AREASOLA1",
            "positions": positions.tolist(),
        }

        # Loop through the lattice sub-segments
        for i, (segment_name, lattice_sub_segment) in enumerate(self.segments.items(), 1):
            # Track the incoming beam with updated magnet settings through this segment,
            # returns ParticleBeam (particles size [32, 7])
            outgoing_beam = lattice_sub_segment.track(incoming_beam)

            # Track by recalculate particle beam with updated observation (TODO: still need for a epsilloid distribution??)
            self.incoming_particle_beam = self.incoming_particle_beam.transformed_to(
                energy=outgoing_beam.energy.clone().detach().float(),
                mu_x=outgoing_beam.mu_x.clone().detach().float(),
                mu_px=outgoing_beam.mu_px.clone().detach().float(),
                mu_y=outgoing_beam.mu_y.clone().detach().float(),
                mu_py=outgoing_beam.mu_py.clone().detach().float(),
                sigma_x=outgoing_beam.sigma_x.clone().detach().float(),
                sigma_px=outgoing_beam.sigma_px.clone().detach().float(),
                sigma_y=outgoing_beam.sigma_y.clone().detach().float(),
                sigma_py=outgoing_beam.sigma_py.clone().detach().float(),
                sigma_p=outgoing_beam.sigma_p.clone().detach().float(),
            )

            # Beam width in the xy-direction
            beam_width = torch.tensor([
                    1.0 if self.beam_width_scale_factor == 1.0 else self.beam_width_scale_factor * outgoing_beam.sigma_x.clone().detach().float(),
                    1.0 if self.beam_width_scale_factor == 1.0 else self.beam_width_scale_factor * outgoing_beam.sigma_y.clone().detach().float(),
                    1.0 if self.beam_width_scale_factor == 1.0 else self.beam_width_scale_factor * outgoing_beam.sigma_tau.clone().detach().float(),
                ],
                dtype=torch.float32
            )

            # Use the beam's center as the original center of geometry
            beam_center = torch.tensor([
                    self.incoming_particle_beam.mu_x,
                    self.incoming_particle_beam.mu_y,
                    self.incoming_particle_beam.mu_tau
                ],
                dtype=torch.float32
            )

            # Pair x and y values from columns 0 and 2 into (x, y) tuples.
            x = self.incoming_particle_beam.particles[:, 0]  # Column 0
            y = self.incoming_particle_beam.particles[:, 2]  # Column 2
            z = self.incoming_particle_beam.particles[:, 4]  # Column 4

            beam_vertices = torch.stack([x, y, z + self.builder.component_positions[segment_name]], dim=1)
            beam_vertices *= beam_width

            # Apply differential scaling transformation around the beam center,
            # the transformation is linear and directionally anisotropic 
            # (only affects x and y while keeping z unchanged).
            if segment_name == self.screen_name: # OR "AREAMCHM1" ??
                epsilon = 1e-6  # Small value to prevent near-zero effects,
                I = torch.ones_like(self.position_scale_factor)
                # We ensure that even when beam_center is close to zero, the transformation still applies
                positions = beam_vertices + (self.position_scale_factor - I) * beam_center
            else:
                positions = beam_vertices

            # Store segment data
            data[f"segment_{i}"] = {
                "segment_name": segment_name,
                "positions": positions.tolist(),
            }

            # Update the incoming beam for next lattice segement
            incoming_beam = outgoing_beam

        # Track the initial lattice segment to retrieve the beam position from the diagnostic screen later
        self._update()

        self.screen_reading = self.lattice_segment.AREABSCR1.reading

        # Update meta info to include particle reading from segments
        self.info.update({
            "segments": data,
            # Sum of the weighted particles distribution across screen pixels
            "screen_reading": self.screen_reading.tolist(),
        })

    def _update(self) -> None:
    	_ = self.lattice_segment.track(self.incoming_beam) # self.incoming_particle_beam

    def get_screen_boundary(self) -> np.ndarray:
        """
        Computes the screen boundary based on resolution and pixel size.

        The boundary is calculated as half of the screen resolution multiplied
        by the pixel size, giving the physical dimensions of the screen in meters.

        Returns:
            np.ndarray: The screen boundary as a 2D numpy array [width, height] in meters.
        """
        return np.array(self.screen.resolution) / 2 * np.array(self.screen.pixel_size)

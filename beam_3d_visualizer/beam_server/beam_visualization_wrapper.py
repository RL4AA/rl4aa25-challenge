import asyncio
import logging
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from cheetah.utils.segment_3d_builder import Segment3DBuilder
from gymnasium import Wrapper

from beam_3d_visualizer.beam_server.websocket_wrapper import WebSocketWrapper


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Define constants at module level
DEFAULT_HTTP_HOST = "localhost"
DEFAULT_HTTP_PORT = 5173
DEFAULT_NUM_PARTICLES = 1000
BEAM_SOURCE_COMPONENT = "AREASOLA1"
SCREEN_NAME = "AREABSCR1"
DEFAULT_SCREEN_RESOLUTION = (2448, 2040)
DEFAULT_SCREEN_PIXEL_SIZE = (3.3198e-6, 2.4469e-6)
DEFAULT_SCREEN_BINNING = 4
DEFAULT_SCALE_FACTOR = 100


class BeamVisualizationWrapper(Wrapper):
    """
    A Gym wrapper that encapsulates the beam simulation logic and manages the
    initialization of the JavaScript web application for 3D visualization.
    """

    def __init__(
        self,
        env: gym.Env,
        http_host: str = DEFAULT_HTTP_HOST,
        http_port: int = DEFAULT_HTTP_PORT,
        is_export_enabled: bool = False,
        num_particles: int = DEFAULT_NUM_PARTICLES,
    ):
        """
        Initialize the BeamVisualizationWrapper.

        Args:
            env (gym.Env): The underlying Gym environment (e.g., BeamControlEnv).
            http_host (str): Hostname for the JavaScript web application server.
            http_port (int): Port for the web application server.
            is_export_enabled (bool): Enable 3D scene export.
            num_particles (int): Number of particles to simulate in the beam.
        """
        # Internally wrap the environment with WebSocketWrapper
        env = WebSocketWrapper(env)
        super().__init__(env)

        # Basic configuration
        self.base_path = Path(__file__).resolve().parent
        self.http_host = http_host
        self.http_port = http_port
        self.num_particles = num_particles
        self.scale_factor = DEFAULT_SCALE_FACTOR
        self.current_step = 0
        self.web_process = None
        self.web_thread = None

        # Initialize state
        self.incoming_particle_beam = None
        self.last_action = np.zeros(5, dtype=np.float32)

        # Start the JavaScript web application
        self._start_web_application()

        # Set up 3D visualization
        self._initialize_3d_visualization(is_export_enabled)

        # Set up screen configuration
        self._initialize_screen()


    def _initialize_3d_visualization(self, is_export_enabled: bool) -> None:
        """
        Initialize the 3D visualization components.

        Args:
            is_export_enabled (bool): Whether to export the 3D scene.
        """
        # Define the output file path relative to the script's directory
        output_path = self.base_path.parent / "public" / "models" / "ares" / "scene.glb"

        # Set lattice segment
        self.segment = self.env.unwrapped.segment

        # Build and export the 3D scene
        self.builder = Segment3DBuilder(self.segment)
        self.builder.build_segment(
            output_filename=str(output_path),
            is_export_enabled=is_export_enabled,
        )

        # Note: For the purpose of beam animation, we consider BEAM_SOURCE_COMPONENT
        # as the origin of the particle beam source
        self.lattice_component_positions = OrderedDict({BEAM_SOURCE_COMPONENT: 0.0})
        self.lattice_component_positions.update(self.builder.component_positions)

        # Store position of lattice components to use in JS web-app beam animation
        self.component_positions = list(self.lattice_component_positions.values())

        # Data to be used to send data over WebSocket
        self.data = OrderedDict(
            {"component_positions": self.component_positions, "segments": {}}
        )

    def _initialize_screen(self) -> None:
        """Initialize the screen configuration for beam visualization."""
        # Define screen
        self.screen_name = SCREEN_NAME
        self.screen = getattr(self.segment, self.screen_name)
        self.screen_resolution = DEFAULT_SCREEN_RESOLUTION
        self.screen_pixel_size = DEFAULT_SCREEN_PIXEL_SIZE
        self.screen.binning = DEFAULT_SCREEN_BINNING
        self.screen.is_active = True

        # Obtain screen boundaries
        self.screen_boundary = self.get_screen_boundary()

        # Update visualization data with screen boundaries
        self.data.update(
            {
                "screen_boundary_x": float(self.screen_boundary[0]),
                "screen_boundary_y": float(self.screen_boundary[1]),
            }
        )

    @property
    def control_action(self):
        """
        Delegate control_action to WebSocketWrapper.

        Returns:
            The current control action from the WebSocketWrapper.
        """
        return self.env.control_action

    @control_action.setter
    def control_action(self, value):
        """
        Set control_action in WebSocketWrapper.

        Args:
            value: The control action value to set.
        """
        self.env.control_action = value

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment, reset last_action, and run the simulation.

        Args:
            seed (Optional[int]): Seed for random number generation.
            options (Optional[Dict]): Additional reset options.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info.
        """
        # Reset action state
        self.last_action = np.zeros(5, dtype=np.float32)
        self.current_step = 0

        # Reset the underlying environment
        observation, info = self.env.reset(seed=seed, options=options)

        # Initialize the particle beam
        self._initialize_particle_beam()

        # Run simulation
        self._simulate()

        return observation, info

    def _initialize_particle_beam(self) -> None:
        """
        Initialize the incoming particle beam for simulation.

        Raises:
            ValueError: If the incoming particle beam cannot be initialized.
        """
        # Try to get the beam from the backend if available
        if hasattr(self.env.unwrapped, "backend") and hasattr(
            self.env.unwrapped.backend, "incoming"
        ):
            self.incoming_particle_beam = (
                self.env.unwrapped.backend.incoming.as_particle_beam(
                    num_particles=self.num_particles
                )
            )
        # Otherwise get it from the incoming_beam attribute
        elif hasattr(self.env.unwrapped, "incoming_beam"):
            self.incoming_particle_beam = (
                self.env.unwrapped.incoming_beam.as_particle_beam(
                    num_particles=self.num_particles
                )
            )
        else:
            raise ValueError(
                "Cannot initialize incoming particle beam. Neither backend.incoming "
                "nor incoming_beam attributes found in the unwrapped environment."
            )

        if self.incoming_particle_beam is None:
            raise ValueError(
                "Incoming particle beam is None. Check beam initialization in BeamControlEnv."
            )


    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a step in the environment and run the simulation.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: Observation, reward,
                terminated, truncated, and info.
        """
        # Update last_action with the action being applied
        self.last_action = action.copy()

        # Execute step in the underlying environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Run simulation with the new state
        self._simulate()

        return observation, reward, terminated, truncated, info

    async def render(self):
        """
        Render the environment by preparing simulation data and delegating
        to the inner wrapper.

        Note: The simulation data is already updated in step() or reset(),
        so we don't need to call _simulate() again here.
        """
        # Update WebSocket broadcasting data
        self.env.data = self.data

        # Delegate to the inner wrapper (WebSocketWrapper) for broadcasting
        if hasattr(self.env, "render"):
            await self.env.render()

    def close(self):
        """
        Close the wrapper and terminate the web application process.
        """
        # Terminate the web application process if it exists
        if self.web_process:
            try:
                self.web_process.terminate()
                self.web_process.wait(timeout=5)
                logger.info("Terminated JavaScript web application process.")
            except subprocess.TimeoutExpired:
                logger.warning("Forcibly killing web application process...")
                self.web_process.kill()

        # Close the underlying environment
        super().close()

    def get_screen_boundary(self) -> np.ndarray:
        """
        Computes the screen boundary based on resolution and pixel size.

        The boundary is calculated as half of the screen resolution multiplied
        by the pixel size, giving the physical dimensions of the screen
        in meters.

        Returns:
            np.ndarray: The screen boundary as a 2D numpy array [width, height]
            in meters.
        """
        return np.array(self.screen.resolution) / 2 * np.array(self.screen.pixel_size)

    def _start_web_application(self):
        """
        Start the JavaScript web application (Vite development server)
        in a background thread.
        """
        def run_web_server():
            try:
                # Start Vite development server
                cmd = [
                    "npx",
                    "vite",
                    "--host",
                    self.http_host,
                    "--port",
                    str(self.http_port),
                ]
                self.web_process = subprocess.Popen(
                    cmd,
                    cwd=self.base_path.parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Log output for debugging
                for line in self.web_process.stdout:
                    logger.info(f"Vite stdout: {line.strip()}")
                for line in self.web_process.stderr:
                    logger.error(f"Vite stderr: {line.strip()}")

            except Exception as e:
                logger.error(f"Failed to start web application: {e}")
                # Consider raising the exception here for better error handling

        # Start the web server in a background thread
        self.web_thread = threading.Thread(target=run_web_server, daemon=True)
        self.web_thread.start()

        # Give the server a moment to start
        logger.info(
            f"Started JavaScript web application on "
            f"http://{self.http_host}:{self.http_port}"
        )

    def _simulate(self) -> None:
        """
        Calculate the positions of beam segments with dynamic angles.

        This method tracks the particle beam through each element in the segment,
        computing the positions of particles at each step. The beam travels along
        the x-axis, with position variations in the yz-plane. The simulation
        data is stored in self.data for later use in visualization.
        """
        # Reset segments data for this simulation step
        self.data["segments"] = {}
        segment_index = 0

        # Track beam through each lattice element
        for element in self.segment.elements:
            if element.name in list(self.lattice_component_positions.keys()):
                # Track beam through this element
                outgoing_beam = element.track(self.incoming_particle_beam)
                logger.debug(
                    f"Tracked beam through element {element.name}: {outgoing_beam}"
                )

                # Extract particle positions
                x = outgoing_beam.particles[:, 0]  # Column 0
                y = outgoing_beam.particles[:, 2]  # Column 2
                z = outgoing_beam.particles[:, 4]  # Column 4

                # Shift beam particles 3D position in reference to segment component
                positions = torch.stack(
                    [x, y, z + self.lattice_component_positions[element.name]], dim=1
                )

                # Compute the mean position of the bunch
                mean_position = positions.mean(dim=0, keepdim=True)

                # Spread out positions without altering the mean
                positions = (
                    positions - mean_position
                ) * self.scale_factor + mean_position

                # Store segment data
                self.data["segments"][f"segment_{segment_index}"] = {
                    "segment_name": element.name,
                    "positions": positions.tolist(),
                }

                # Update segment index
                segment_index += 1

                # Update the incoming beam for the next lattice segment
                self.incoming_particle_beam = outgoing_beam

        # Get screen pixel reading
        screen_reading = self.env.unwrapped.segment.AREABSCR1.reading
        self.current_step += 1

        # Update meta info to include particle reading from segments
        self.data.update(
            {
                "screen_reading": screen_reading.tolist(),
                "bunch_count": self.current_step,
            }
        )

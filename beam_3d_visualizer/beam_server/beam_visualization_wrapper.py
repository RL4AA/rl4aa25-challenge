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
from beam_3d_visualizer.beam_server.websocket_wrapper import WebSocketWrapper
from cheetah.utils.segment_3d_builder import Segment3DBuilder
from gymnasium import Wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class BeamVisualizationWrapper(Wrapper):
    """
    A Gym wrapper that encapsulates the beam simulation logic and manages the
    initialization of the JavaScript web application for 3D visualization.
    """

    def __init__(
        self,
        env: gym.Env,
        http_host: str = "localhost",
        http_port: int = 5173,
        is_export_enabled: bool = False,
        num_particles: int = 1000,
    ):
        """
        Initialize the BeamVisualizationWrapper.

        Args:
            env (gym.Env): The underlying Gym environment (e.g., BeamControlEnv).
            web_app_dir (Path): Directory containing the JavaScript web application.
            host (str): Hostname for the web application server.
            port (int): Port for the web application server.
            is_export_enabled (boolean): Enable 3D scene export.
        """
        # Internally wrap the environment with WebSocketWrapper
        env = WebSocketWrapper(env)
        super().__init__(env)

        base_path = Path(__file__).resolve().parent
        print(f"TODO: base_path: {base_path}")

        self.web_app_dir = base_path
        self.http_host = http_host
        self.http_port = http_port
        self.web_process = None
        self.web_thread = None

        # Start the JavaScript web application
        self._start_web_application()

        self.incoming_particle_beam = None

        # Define the output file path relative to the script's directory
        output_path = base_path.parent / "public" / "models" / "ares" / "scene.glb"

        # Set lattice segment
        self.segment = self.env.unwrapped.segment

        # Build and export the 3D scene
        self.builder = Segment3DBuilder(self.segment)
        self.builder.build_segment(
            output_filename=str(output_path),
            is_export_enabled=is_export_enabled,
        )

        # Note: For the purpose of beam animation, we consider "AREASOLA1"
        # as the origin of the particle beam source
        self.lattice_component_positions = OrderedDict({"AREASOLA1": 0.0})
        self.lattice_component_positions.update(self.builder.component_positions)

        # Store position of lattice components to use in JS web-app beam animation
        self.component_positions = list(self.lattice_component_positions.values())

        # Data to be used to send data over WebSocket
        self.data = OrderedDict(
            {"component_positions": self.component_positions, "segments": {}}
        )

        # Define screen
        self.screen_name = "AREABSCR1"
        self.screen = getattr(self.segment, self.screen_name)
        self.screen_resolution = (2448, 2040)
        self.screen_pixel_size = (3.3198e-6, 2.4469e-6)
        self.screen.binning = 4  # default: 1
        self.screen.is_active = True

        # Obtain screen Boundaries
        # The resolution of the screen in pixels (width, height),
        # i.e. 2448, 2040
        # The physical size of a single pixel in meters (width, height),
        # i.e. (3.5488e-06, 2.5003e-06
        self.screen_boundary = (
            self.get_screen_boundary()
        )  # e.g.  [0.00434373, 0.00255031]

        self.data.update(
            {
                "screen_boundary_x": float(self.screen_boundary[0]),
                "screen_boundary_y": float(self.screen_boundary[1]),
            }
        )

        self.current_step = 0  # Episode step counter
        self.num_particles = num_particles

        self.last_action = np.array(
            [0, 0, 0, 0, 0], dtype=np.float32
        )  # Initialize last_action

        self.scale_factor = 100

    @property
    def control_action(self):
        """Delegate control_action to WebSocketWrapper."""
        return self.env.control_action

    @control_action.setter
    def control_action(self, value):
        """Set control_action in WebSocketWrapper."""
        self.env.control_action = value

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
                    cwd=self.web_app_dir.parent,
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

        self.web_thread = threading.Thread(target=run_web_server, daemon=True)
        self.web_thread.start()
        logger.info(
            f"Started JavaScript web application on\
            http://{self.http_host}:{self.http_port}"
        )

    def _simulate(self) -> None:
        """
        Calculate the positions of beam segments with dynamic angles.
        Beam travels along x-axis, with position variations in yz-plane.
        Allows for more dynamic z-axis movement.
        """
        segment_index = 0

        for element in self.segment.elements:
            if element.name in list(self.lattice_component_positions.keys()):
                outgoing_beam = element.track(self.incoming_particle_beam)
                logger.debug(
                    f"Tracked beam through element {element.name}: {outgoing_beam}"
                )

                # Pair x and y values from columns 0 and 2 into (x, y) tuples.
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
                "bunch_count": self.current_step + 1,
            }
        )

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
        self.last_action = np.array(
            [0, 0, 0, 0, 0], dtype=np.float32
        )  # Reset last_action

        observation, info = self.env.reset(seed=seed, options=options)

        self.incoming_particle_beam = self.env.unwrapped.incoming_beam.as_particle_beam(
            num_particles=self.num_particles
        )

        if hasattr(self.env.unwrapped, "backend") and hasattr(
            self.env.unwrapped.backend, "incoming"
        ):
            self.incoming_particle_beam = (
                self.env.unwrapped.backend.incoming.as_particle_beam(
                    num_particles=self.num_particles
                )
            )
        else:
            self.incoming_particle_beam = (
                self.env.unwrapped.incoming_beam.as_particle_beam(
                    num_particles=self.num_particles
                )
            )

        if self.incoming_particle_beam is None:
            raise ValueError(
                "Incoming particle beam is None.\
                Check beam initialization in BeamControlEnv."
            )

        self._simulate()

        return observation, info

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
        self.last_action = action

        observation, reward, terminated, truncated, info = self.env.step(action)

        self._simulate()

        return observation, reward, terminated, truncated, info

    async def render(self):
        """
        Render the environment by preparing simulation data and delegating
        to the inner wrapper.
        """
        # Optionally, perform any rendering-related tasks specific to this wrapper
        # For example, ensure simulation data is up-to-date
        self._simulate()  # Ensure simulation data is fresh before rendering

        # Update WebSocket broadcasting data
        self.env.data = self.data

        # Delegate to the inner wrapper (WebSocketWrapper) for broadcasting
        if hasattr(self.env, "render"):  # WebSocketWrapper
            await self.env.render()

    def close(self):
        """
        Close the wrapper and terminate the web application process.
        """
        if self.web_process:
            self.web_process.terminate()
            self.web_process.wait()
            logger.info("Terminated JavaScript web application process.")
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


async def main_with_wrappers():
    """Example main function to demonstrate usage of the wrapper."""
    import argparse

    import yaml
    from beam_control_env import BeamControlEnv

    from websocket_wrapper import WebSocketWrapper

    parser = argparse.ArgumentParser(
        description="Run the BeamControlEnv simulation with wrappers."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="env_configs.yml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--web-app-dir",
        type=str,
        default="path/to/your/js/web/app",
        help="Directory containing the JavaScript web application",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Initialize environment and wrap it
    env = BeamControlEnv(config=config)
    env = WebSocketWrapper(env)  # Handle WebSocket communication
    env = BeamVisualizationWrapper(
        env,
        web_app_dir=Path(args.web_app_dir),
        host=config["env_config"].get("host", "localhost"),
        port=5173,  # Vite default port
    )

    # Run simulation
    observation, info = env.reset()
    done = False

    while not done:
        if not env.env.connected:  # Check WebSocketWrapper's connected status
            logger.info("Waiting for WebSocket client to connect...")
            await asyncio.sleep(0.5)
            continue

        await env.env.render()  # Render and broadcast via WebSocketWrapper
        action = np.zeros(5, dtype=np.float32)  # Default action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    asyncio.run(main_with_wrappers())

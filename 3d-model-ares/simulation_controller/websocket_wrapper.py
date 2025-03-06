import argparse
import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import websockets
import yaml
from beam_control_env import BeamControlEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class WebSocketWrapper(gym.Wrapper):
    """
    A Gym wrapper that integrates WebSocket functionality with BeamControlEnv.
    Manages WebSocket server and client communication internally.
    """

    def __init__(
        self,
        env: gym.Env,
    ):
        """
        Initialize the WebSocketWrapper.

        Args:
            env (gym.Env): The BeamControlEnv instance to wrap
        """
        super().__init__(env)

        # Store host and port
        self.host = self.env.unwrapped.host
        self.port = self.env.unwrapped.port

        # WebSocket management attributes
        self.clients = set()
        self.connected = False
        self.server = None
        self.control_action = None

        # Replace the environment's websocket_manager with this wrapper
        if hasattr(self.env.unwrapped, "websocket_manager"):
            self.env.unwrapped.websocket_manager = self

        # Start the WebSocket server in a separate thread
        self._start_websocket_server()

    def _start_websocket_server(self):
        """Start the WebSocket server in a background thread."""

        def run_server():
            asyncio.run(self._run_server())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"WebSocket server thread started on ws://{self.host}:{self.port}")

    async def _run_server(self):
        """Run the WebSocket server."""
        self.server = await websockets.serve(self._handle_client, self.host, self.port)
        logger.info(f"WebSocket server running on ws://{self.host}:{self.port}")
        await self.server.wait_closed()

    async def _handle_client(
        self, websocket: websockets.WebSocketServerProtocol, path: str = None
    ):
        """Handle incoming WebSocket connections and messages."""
        self.connected = True
        self.clients.add(websocket)
        logger.info("WebSocket connection established.")

        try:
            async for message in websocket:
                data = json.loads(message)
                logger.debug(f"Received data: {data}")

                if "controls" in data:
                    self.control_action = np.array(
                        list(data["controls"].values()), dtype=np.float32
                    )
                    logger.debug(f"Received control action: {self.control_action}")
        except json.JSONDecodeError:
            logger.error("Error: Received invalid JSON data.")
        except asyncio.exceptions.CancelledError:
            logger.info("WebSocket task was cancelled.")
            raise
        except websockets.ConnectionClosed:
            logger.info("WebSocket connection closed by client.")
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)
            if not self.clients:
                self.connected = False
            logger.info("Client cleanup completed.")

    async def broadcast(self, message: Dict):
        """Broadcast a message to all connected clients."""
        if not self.clients:
            return

        disconnected_clients = set()
        tasks = []

        for client in self.clients:
            try:
                tasks.append(client.send(json.dumps(message)))
            except websockets.ConnectionClosed:
                disconnected_clients.add(client)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.clients.difference_update(disconnected_clients)
        if disconnected_clients:
            logger.info(f"Removed {len(disconnected_clients)} disconnected clients")

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a step in the environment, using WebSocket control action if available.
        """
        if self.control_action is not None:
            action = self.control_action
            self.control_action = None  # Clear after use

        return self.env.step(action)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        return self.env.reset(seed=seed, options=options)

    def close(self):
        """Close the environment and WebSocket server."""
        if self.server:
            self.connected = False
            self.clients.clear()
            self.server.close()
            logger.info("WebSocket server closed.")
        super().close()

    async def render(self):
        """Render the environment and broadcast data via WebSocket."""
        if hasattr(self.env, "render"):
            await self.env.render()


# Example usage in main.py
async def main_with_wrapper():
    parser = argparse.ArgumentParser(
        description="Run the BeamControlEnv simulation with WebSocket wrapper."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="env_configs.yml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    host = config["env_config"].get("host", "localhost")
    port = config["env_config"].get("port", 8081)

    # Initialize environment and wrap it
    env = BeamControlEnv(config=config)  # No websocket_manager needed
    env = WebSocketWrapper(env, host=host, port=port)

    # Run simulation
    observation, info = env.reset()
    done = False

    while not done:
        if not env.connected:
            logger.info("Waiting for WebSocket client to connect...")
            await asyncio.sleep(0.5)
            continue

        await env.render()
        action = np.zeros(5, dtype=np.float32)  # Default action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    asyncio.run(main_with_wrapper())

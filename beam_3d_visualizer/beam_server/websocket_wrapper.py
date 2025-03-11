import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Define constants at module level
DEFAULT_WS_HOST = "localhost"
DEFAULT_WS_PORT = 8081
DEFAULT_CONNECTION_TIMEOUT = 1.0


class WebSocketWrapper(gym.Wrapper):
    """
    A Gym wrapper that enables WebSocket integration for communication
    with a Gym-based environment.
    Manages WebSocket server and client communication internally.
    """

    def __init__(
        self,
        env: gym.Env,
        ws_host: str = DEFAULT_WS_HOST,
        ws_port: int = DEFAULT_WS_PORT,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
    ):
        """
        Initialize the WebSocketWrapper.

        Args:
            env (gym.Env): The underlying Gym environment.
            ws_host (str): WebSocket server hostname.
            ws_port (int): WebSocket server port.
            connection_timeout (float): Timeout for WebSocket connections in seconds.
        """
        super().__init__(env)

        # Store host and port, with defaults if not defined in env.unwrapped
        self.ws_host = getattr(self.env.unwrapped, "host", ws_host)
        self.ws_port = getattr(self.env.unwrapped, "port", ws_port)
        self.connection_timeout = connection_timeout

        # Validate port number
        if not isinstance(self.ws_port, int) or not (1 <= self.ws_port <= 65535):
            logger.warning(
                f"Invalid port number {self.ws_port}. Defaulting to {self.ws_port}."
            )

        # WebSocket management attributes
        self.clients = set()
        self.connected = False
        self.server = None

        # Data to be transmitted to the JavaScript web application
        self.data = None

        self._control_action = np.array(
            [0, 0, 0, 0, 0], dtype=np.float32
        )  # "no-op" action
        self.last_action = None  # Not used here, but included for completeness

        # Start the WebSocket server in a separate thread
        self._lock = threading.Lock()
        self._start_websocket_server()

    @property
    def control_action(self):
        """Get the current control action."""
        return self._control_action

    @control_action.setter
    def control_action(self, value):
        """Set the current control action."""
        self._control_action = value

    def _start_websocket_server(self):
        """Start the WebSocket server in a background thread."""
        def run_server():
            asyncio.run(self._run_server())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info(
            f"WebSocket server thread started on ws://{self.ws_host}:{self.ws_port}"
        )

    async def _run_server(self):
        """Run the WebSocket server."""
        self.server = await websockets.serve(
            self._handle_client, self.ws_host, self.ws_port
        )
        logger.info(f"WebSocket server running on ws://{self.ws_host}:{self.ws_port}")
        await self.server.wait_closed()

    async def _handle_client(
        self, websocket: websockets.WebSocketServerProtocol, path: str = None
    ):
        """Handle incoming WebSocket connections and messages."""
        with self._lock:
            self.connected = True
            self.clients.add(websocket)
        logger.info("WebSocket connection established.")

        try:
            async for message in websocket:
                try:
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
            with self._lock:
                self.clients.discard(websocket)
                if not self.clients:
                    self.connected = False
            logger.info("Client cleanup completed.")

    async def broadcast(self, message: Dict):
        """Broadcast a message to all connected clients."""
        if not self.clients:
            return

        tasks = [client.send(json.dumps(message)) for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a step in the environment using WebSocket control action if available.
        """
        if self.control_action is not None:
            action = self.control_action
            self.control_action = None  # Clear after use

        return self.env.step(action)

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
            await self.env.render()  # Let the env prepare its state
            await self.broadcast(self.data)  # Broadcast the updated info

            # Add delay after broadcasting to allow animation to complete
            # before sending new
            await asyncio.sleep(1.0)

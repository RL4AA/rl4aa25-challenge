import argparse
import asyncio
import collections
import logging
import time

import numpy as np
import yaml
from beam_control_env import BeamControlEnv
from stable_baselines3.common.utils import set_random_seed
from websocket_wrapper import WebSocketWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def run_simulation(env: WebSocketWrapper):
    """Run the simulation loop, stepping the environment with control actions."""
    logger.info("\n--- Starting simulation loop ---")

    step_count = 0
    done = False

    # Queue to buffer incoming actions
    action_queue = collections.deque(maxlen=10)  # Stores up to 10 recent actions

    # Initialize last_action with default values
    last_action = np.zeros(5, dtype=np.float32)

    # Timestamp to track when the last action was received
    last_action_time = time.time()

    # Reset the environment
    observation, _ = env.reset()

    while not done:
        if not env.connected:
            logger.info("Waiting for WebSocket client to connect...")
            await asyncio.sleep(0.5)  # Small delay to prevent CPU overload
            continue

        # Render and broadcast data to clients
        await env.render()

        # Check if we have a new control action from WebSocket
        if env.control_action is not None:
            action_queue.append(env.control_action.copy())  # Store in queue
            logger.info(f"Queued new action: {env.control_action}")
            env.control_action = None  # Clear after use
            last_action_time = time.time()  # Update last action timestamp

        # Check if we have actions in the queue
        if action_queue:
            last_action = action_queue.popleft()  # Take the oldest action
        elif time.time() - last_action_time > 2:
            # If no new actions for too long, issue a warning
            logger.warning(
                "No new control action received for 2 seconds. Using last action."
            )

        # Step through the environment with the action
        observation, reward, terminated, truncated, info = env.step(last_action)

        # Update step count
        step_count += 1

        logger.info(
            "Step %d: Action = %s, Reward = %s, Observation = %s",
            step_count,
            last_action,
            reward,
            observation,
        )

        done = terminated or truncated

    env.close()
    logger.info("Simulation completed.")


async def main():
    """Main entry point to set up the environment and start the simulation."""
    parser = argparse.ArgumentParser(description="Run the BeamControlEnv simulation.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="env_configs.yml",
        help="Path to the YAML configuration file (default: env_configs.yml)",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set random seed if provided
    if config["env_config"].get("seed") is not None:
        set_random_seed(config["env_config"]["seed"])

    # Initialize the environment and wrap it with WebSocketWrapper
    env = BeamControlEnv(config=config)
    env = WebSocketWrapper(env)

    # Run the simulation
    await run_simulation(env)

    logger.info("Simulation shutdown completed.")


if __name__ == "__main__":
    asyncio.run(main())

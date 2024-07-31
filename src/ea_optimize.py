from concurrent.futures import ThreadPoolExecutor

# import dummypydoocs as pydoocs
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack, RescaleAction, TimeLimit
from stable_baselines3 import PPO, TD3

from .environments import ea
from .utils import load_config
from .wrappers import (
    ARESeLog,
    NotVecNormalize,
    PolishedDonkeyCompatibility,
    RecordEpisode,
    RescaleObservation,
    TQDMWrapper,
)


def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=4e-5,
    target_mu_y_threshold=4e-5,
    target_sigma_x_threshold=4e-5,
    target_sigma_y_threshold=4e-5,
    max_steps=50,
    model_name="vital-breeze-514",
    logbook=False,
    data_log_dir=None,
    progress_bar=False,
    callback=None,
    backend="cheetah",
    initial_magnets=None,
):
    """
    Optimise beam in ARES EA using a reinforcement learning agent.
    """
    config = load_config(f"../models/ea/ppo/{model_name}/config")

    # Load the model
    model = PPO.load(f"../models/ea/ppo/{model_name}/model")

    callback = setup_callback(callback)

    # Create the environment
    env = ea.TransverseTuning(
        backend=backend,
        backend_args={"generate_screen_images": True} if backend == "cheetah" else {},
        action_mode=config["action_mode"],
        magnet_init_mode=(
            config["magnet_init_mode"] if initial_magnets is None else initial_magnets
        ),
        max_quad_setting=config["max_quad_setting"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=np.array(
            [target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]
        ),
        target_threshold=np.array(
            [
                target_mu_x_threshold,
                target_sigma_x_threshold,
                target_mu_y_threshold,
                target_sigma_y_threshold,
            ]
        ),
        clip_magnets=False,  # Should happen by itself on the machine
        beam_param_transform=config["beam_param_transform"],
        beam_param_combiner=config["beam_param_combiner"],
        beam_param_combiner_args=config["beam_param_combiner_args"],
        beam_param_combiner_weights=config["beam_param_combiner_weights"],
        final_combiner=config["final_combiner"],
        final_combiner_args=config["final_combiner_args"],
        final_combiner_weights=config["final_combiner_weights"],
    )
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    if progress_bar:
        env = TQDMWrapper(env)
    if callback is not None:
        env = OptimizeFunctionCallback(env, callback)
    if data_log_dir is not None:
        env = RecordEpisode(env, save_dir=data_log_dir)
    if logbook:
        env = ARESeLog(env, agent_name=model_name)
    if config["normalize_observation"] and not config["running_obs_norm"]:
        env = RescaleObservation(env, -1, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    env = FlattenObservation(env)
    if config["frame_stack"] > 1:
        env = FrameStack(env, config["frame_stack"])

    callback.env = env

    # Actual optimisation
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


def optimize_donkey(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=4e-5,
    target_mu_y_threshold=4e-5,
    target_sigma_x_threshold=4e-5,
    target_sigma_y_threshold=4e-5,
    max_steps=50,
    model_name="polished-donkey-996",
    logbook=False,
    data_log_dir=None,
    progress_bar=False,
    callback=None,
    backend="cheetah",
):
    """
    Function used for optimisation during operation.

    Note: Current version only works for polished-donkey-996.
    """
    # config = read_from_yaml(f"models/{model}/config")
    assert (
        model_name == "polished-donkey-996"
    ), "Current version only works for polished-donkey-996."

    # Load the model
    model = TD3.load(f"../models/ea/legacy/{model_name}/model")

    callback = setup_callback(callback)

    # Create the environment
    env = ea.TransverseTuning(
        backend=backend,
        backend_args={"generate_screen_images": True} if backend == "cheetah" else {},
        action_mode="delta",
        magnet_init_mode=np.array([10, -10, 0, 10, 0]),
        max_quad_delta=30 * 0.1,
        max_steerer_delta=6e-3 * 0.1,
        target_beam_mode=np.array(
            [target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]
        ),
        target_threshold=np.array(
            [
                target_mu_x_threshold,
                target_sigma_x_threshold,
                target_mu_y_threshold,
                target_sigma_y_threshold,
            ]
        ),
        clip_magnets=False,  # Should happen by itself on the machine
    )
    if max_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_steps)
    if progress_bar:
        env = TQDMWrapper(env)
    if callback is not None:
        env = OptimizeFunctionCallback(env, callback)
    if data_log_dir is not None:
        env = RecordEpisode(env, save_dir=data_log_dir)
    if logbook:
        env = ARESeLog(env, agent_name=model_name)
    # env = RecordVideo(env, f"recordings_real/{datetime.now():%Y%m%d%H%M}")
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"../models/ea/legacy/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    callback.env = env

    # Actual optimisation
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


def optimize_async(*args, **kwargs):
    """Run `optimize without blocking."""
    executor = ThreadPoolExecutor(max_workers=1)
    # executor.submit(optimize, *args, **kwargs)
    kwargs["model_name"] = "polished-donkey-996"
    executor.submit(optimize_donkey, *args, **kwargs)


def setup_callback(callback):
    """
    Prepare the callback for the actual optimisation run and return a callback that
    works exactly as expected.
    """
    if callback is None:
        callback = BaseCallback()
    elif isinstance(callback, list):
        callback = CallbackList(callback)
    return callback


class BaseCallback:
    """
    Base for callbacks to pass into `optimize` function and get information at different
    points of the optimisation.
    Provides access to the environment via `self.env`.
    """

    def __init__(self):
        self.env = None

    def environment_reset(self, obs):
        """Called after the environment's `reset` method has been called."""
        pass

    def environment_step(self, obs, reward, done, info):
        """
        Called after every call to the environment's `step` function.
        Return `True` tostop optimisation.
        """
        return False

    def environment_close(self):
        """Called after the optimization was finished."""
        pass


class CallbackList(BaseCallback):
    """Combines multiple callbacks into one."""

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value
        for callback in self.callbacks:
            callback.env = self._env

    def environment_reset(self, obs):
        for callback in self.callbacks:
            callback.environment_reset(obs)

    def environment_step(self, obs, reward, done, info):
        return any(
            [
                callback.environment_step(obs, reward, done, info)
                for callback in self.callbacks
            ]
        )

    def environment_close(self):
        for callback in self.callbacks:
            callback.environment_close()


class TestCallback(BaseCallback):
    """
    Very simple callback for testing. Prints method name and arguments whenever callback
    is called.
    """

    def environment_reset(self, obs):
        print(
            f"""environment_reset
    -> {obs = }"""
        )

    def environment_step(self, obs, reward, done, info):
        print(
            f"""environment_step
    -> {obs = }
    -> {reward = }
    -> {done = }
    -> {info = }"""
        )
        return False

    def environment_close(self):
        print("""environment_close""")


class OptimizeFunctionCallback(gym.Wrapper):
    """Wrapper to send screen image, beam parameters and optimisation end to GUI."""

    def __init__(self, env, callback):
        super().__init__(env)
        self.callback = callback

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.callback.environment_reset(observation)
        return observation, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        done = done or self.callback.environment_step(obs, reward, done, info)
        return obs, reward, done, truncated, info

    def close(self):
        super().close()
        self.callback.environment_close()

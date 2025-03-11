from functools import partial

import gymnasium as gym
import torch.nn as nn
import wandb
from gymnasium.wrappers import (
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from rl_zoo3 import linear_schedule
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from src.wrappers import LogTaskStatistics, PolishedDonkeyReward

from ..environments import ea
from ..utils import save_config
from ..wrappers import PlotEpisode, RescaleObservation


def main() -> None:
    config = {
        # Environment
        "action_mode": "delta",
        "max_quad_setting": 30.0,
        "max_quad_delta": 30.0,
        "max_steerer_delta": 6.1782e-3,
        "magnet_init_mode": "random",
        "incoming_mode": "random",
        "misalignment_mode": "random",
        "max_misalignment": 5e-4,
        "target_beam_mode": "random",
        "threshold_hold": 1,
        "clip_magnets": True,
        # Reward (also environment)
        "beam_param_transform": "ClippedLinear",
        "beam_param_combiner": "Mean",
        "beam_param_combiner_args": {},
        "beam_param_combiner_weights": [1, 1, 1, 1],
        "magnet_change_transform": "Sigmoid",
        "magnet_change_combiner": "Mean",
        "magnet_change_combiner_args": {},
        "magnet_change_combiner_weights": [1, 1, 1, 1, 1],
        "final_combiner": "Mean",
        "final_combiner_args": {},
        "final_combiner_weights": [3, 0.5, 0.5],
        # Wrappers
        "frame_stack": 1,  # 1 means no frame stacking
        "normalize_observation": True,
        "running_obs_norm": False,
        "normalize_reward": False,  # Not really needed because normalised by design
        "rescale_action": True,
        "target_threshold": None,  # 2e-5 m is estimated screen resolution
        "max_episode_steps": 50,
        "polished_donkey_reward": False,
        # RL algorithm
        "n_envs": 40,
        "learning_rate": 0.0003,
        "lr_schedule": "constant",  # Can be "constant" or "linear"
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 40 * 50,  # (1, "episode"),
        "gradient_steps": -1,
        "action_noise": None,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "total_timesteps": 1_000_000,
        # Policy
        "net_arch": "small",  # Can be "small" or "medium"
        "activation_fn": "ReLU",  # Tanh, ReLU, GELU
        "n_critics": 2,
        # SB3 config
        "sb3_device": "auto",
        "vec_env": "subproc",
    }

    train(config)


def train(config: dict) -> None:
    # Setup wandb
    wandb.init(
        project="ares-ea-v3",
        entity="msk-ipc",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        dir=".wandb",
    )
    config = dict(wandb.config)
    config["run_name"] = wandb.run.name

    # Setup environments
    if config["vec_env"] == "dummy":
        vec_env = DummyVecEnv(
            [partial(make_env, config) for _ in range(config["n_envs"])]
        )
    elif config["vec_env"] == "subproc":
        vec_env = SubprocVecEnv(
            [partial(make_env, config) for _ in range(config["n_envs"])]
        )
    else:
        raise ValueError(f"Invalid value \"{config['vec_env']}\" for dummy")
    eval_vec_env = DummyVecEnv(
        [partial(make_env, config, plot_episode=True, log_task_statistics=True)]
    )

    if (config["normalize_observation"] and config["running_obs_norm"]) or config[
        "normalize_reward"
    ]:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config["normalize_observation"] and config["running_obs_norm"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
        )
        eval_vec_env = VecNormalize(
            eval_vec_env,
            norm_obs=config["normalize_observation"] and config["running_obs_norm"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
            training=False,
        )

    # Setup learning rate schedule if needed
    if config["lr_schedule"] == "linear":
        config["learning_rate"] = linear_schedule(config["learning_rate"])

    # Train
    model = TD3(
        "MlpPolicy",
        vec_env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        gamma=config["gamma"],
        train_freq=(
            tuple(config["train_freq"])
            if isinstance(config["train_freq"], list)
            else config["train_freq"]
        ),
        gradient_steps=config["gradient_steps"],
        action_noise=config["action_noise"],
        policy_delay=config["policy_delay"],
        target_policy_noise=config["target_policy_noise"],
        target_noise_clip=config["target_noise_clip"],
        policy_kwargs={
            "activation_fn": getattr(nn, config["activation_fn"]),
            "net_arch": {  # From rl_zoo3
                "small": {"pi": [64, 64], "qf": [64, 64]},
                "medium": {"pi": [256, 256], "qf": [256, 256]},
            }[config["net_arch"]],
            "n_critics": config["n_critics"],
        },
        device=config["sb3_device"],
        tensorboard_log=f"log/{config['run_name']}",
        verbose=1,
    )

    eval_callback = EvalCallback(eval_vec_env, eval_freq=1_000, n_eval_episodes=5)
    wandb_callback = WandbCallback()

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[eval_callback, wandb_callback],
    )

    model.save(f"models/ea/td3/{wandb.run.name}/model")
    if (config["normalize_observation"] and config["running_obs_norm"]) or config[
        "normalize_reward"
    ]:
        vec_env.save(f"models/ea/td3/{wandb.run.name}/vec_normalize.pkl")
    save_config(config, f"models/ea/td3/{wandb.run.name}/config")


def make_env(
    config: dict,
    record_video: bool = False,
    plot_episode: bool = False,
    log_task_statistics: bool = False,
) -> gym.Env:
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": config["incoming_mode"],
            "misalignment_mode": config["misalignment_mode"],
            "max_misalignment": config["max_misalignment"],
            "generate_screen_images": plot_episode,
        },
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        max_quad_setting=config["max_quad_setting"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=config["target_beam_mode"],
        target_threshold=config["target_threshold"],
        threshold_hold=config["threshold_hold"],
        clip_magnets=config["clip_magnets"],
        beam_param_transform=config["beam_param_transform"],
        beam_param_combiner=config["beam_param_combiner"],
        beam_param_combiner_args=config["beam_param_combiner_args"],
        beam_param_combiner_weights=config["beam_param_combiner_weights"],
        magnet_change_transform=config["magnet_change_transform"],
        magnet_change_combiner=config["magnet_change_combiner"],
        magnet_change_combiner_args=config["magnet_change_combiner_args"],
        magnet_change_combiner_weights=config["magnet_change_combiner_weights"],
        final_combiner=config["final_combiner"],
        final_combiner_args=config["final_combiner_args"],
        final_combiner_weights=config["final_combiner_weights"],
        render_mode="rgb_array",
    )
    env = TimeLimit(env, config["max_episode_steps"])
    if plot_episode:
        env = PlotEpisode(
            env,
            save_dir=f"plots/{config['run_name']}",
            episode_trigger=lambda x: x % 5 == 0,  # Once per (5x) evaluation
            log_to_wandb=True,
        )
    if log_task_statistics:
        env = LogTaskStatistics(env)
    if config["normalize_observation"] and not config["running_obs_norm"]:
        env = RescaleObservation(env, -1, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    if config["polished_donkey_reward"]:
        env = PolishedDonkeyReward(env)
    env = FlattenObservation(env)
    if config["frame_stack"] > 1:
        env = FrameStack(env, config["frame_stack"])
    env = Monitor(env)
    if record_video:
        env = RecordVideo(
            env,
            video_folder=f"recordings/{config['run_name']}",
            episode_trigger=lambda x: x % 5 == 0,  # Once per (5x) evaluation
        )
    return env


if __name__ == "__main__":
    main()

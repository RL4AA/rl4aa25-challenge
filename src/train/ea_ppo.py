from functools import partial

import gymnasium as gym
import numpy as np
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from ..environments import ea
from ..utils import save_config
from ..wrappers import LogTaskStatistics, PlotEpisode, RescaleObservation


def main() -> None:
    config = {
        # Environment
        "action_mode": "delta",
        "max_quad_setting": 30.0,
        "max_quad_delta": 30.0,
        "max_steerer_delta": 6.1782e-3,
        "magnet_init_mode": np.array([10.0, -10.0, 0.0, 10.0, 0.0]),
        "incoming_mode": "random",
        "misalignment_mode": "random",
        "max_misalignment": 5e-4,
        "target_beam_mode": np.zeros(4),
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
        "batch_size": 64,
        "learning_rate": 0.0003,
        "lr_schedule": "constant",  # Can be "constant" or "linear"
        "gamma": 0.99,
        "n_envs": 40,
        "n_steps": 64,
        "ent_coef": 0.0,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,  # None,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "total_timesteps": 5_000_000,
        # Policy
        "net_arch": "small",  # Can be "small" or "medium"
        "activation_fn": "Tanh",  # Tanh, ReLU, GELU
        "ortho_init": True,  # True, False
        "log_std_init": 0.0,
        # SB3 config
        "sb3_device": "auto",
        "vec_env": "subproc",
    }

    train(config)


def train(config: dict) -> None:
    # Setup wandb
    wandb.init(
        entity="msk-ipc",
        project="rl4aa-tutorial-2025-dev",
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
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        use_sde=config["use_sde"],
        sde_sample_freq=config["sde_sample_freq"],
        target_kl=config["target_kl"],
        policy_kwargs={
            "activation_fn": getattr(nn, config["activation_fn"]),
            "net_arch": {  # From rl_zoo3
                "small": {"pi": [64, 64], "vf": [64, 64]},
                "medium": {"pi": [256, 256], "vf": [256, 256]},
            }[config["net_arch"]],
            "ortho_init": config["ortho_init"],
            "log_std_init": config["log_std_init"],
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

    model.save(f"models/ea/ppo/{wandb.run.name}/model")
    if (config["normalize_observation"] and config["running_obs_norm"]) or config[
        "normalize_reward"
    ]:
        vec_env.save(f"models/ea/ppo/{wandb.run.name}/vec_normalize.pkl")
    save_config(config, f"models/ea/ppo/{wandb.run.name}/config")

    vec_env.close()
    eval_vec_env.close()


def make_env(
    config: dict,
    record_video: bool = False,
    plot_episode: bool = False,
    log_task_statistics: bool = False,
) -> gym.Env:
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "generate_screen_images": plot_episode,
            "incoming_mode": config["incoming_mode"],
            "max_misalignment": config["max_misalignment"],
            "misalignment_mode": config["misalignment_mode"],
            "simulate_finite_screen": config["simulate_finite_screen"],
        },
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        max_quad_setting=config["max_quad_setting"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=config["target_beam_mode"],
        target_threshold=config["target_threshold"],
        # threshold_hold=config["threshold_hold"],
        clip_magnets=config["clip_magnets"],
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
    if config["normalize_observation"]:
        env = RescaleObservation(env, -1, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
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

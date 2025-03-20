from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, FrameStack, RescaleAction, TimeLimit
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm

from src.environments import ea
from src.trial import Trial, load_trials
from src.wrappers import RecordEpisode, RescaleObservation


def try_problem(
    trial_index: int,
    trial: Trial,
    model: BaseAlgorithm,
    config: dict,
    write_data: bool = True,
) -> None:
    policy_name = config["run_name"]

    # Create the environment
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": trial.incoming_beam,
            "max_misalignment": 5e-4,
            "misalignment_mode": trial.misalignments,
            "simulate_finite_screen": False,
        },
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        max_quad_setting=config["max_quad_setting"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=np.zeros(4),
        target_threshold=None,
        threshold_hold=5,
        clip_magnets=True,
    )
    env = TimeLimit(env, 150)
    if write_data:
        env = RecordEpisode(
            env,
            save_dir=(f"data/evaluate_policy/{policy_name}/problem_{trial_index:03d}"),
        )
    if config["normalize_observation"]:
        env = RescaleObservation(env, -1, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    env = FlattenObservation(env)
    if config["frame_stack"] > 1:
        env = FrameStack(env, config["frame_stack"])

    # Actual optimisation
    observation, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


def generate_trials(num: int, seed: int | None = None) -> list[Trial]:
    assert num > 0

    if seed is not None:
        np.random.seed(seed)

    target_beam_space = spaces.Box(
        low=np.array([-2e-3, 0, -2e-3, 0], dtype=np.float32),
        high=np.array([2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float32),
    )
    incoming_beam_space = spaces.Box(
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
    misalignment_space = spaces.Box(low=-5e-4, high=5e-4, shape=(8,), dtype=np.float32)
    initial_magnet_space = spaces.Box(
        low=np.array([-30, -30, -6.1782e-3, -30, -6.1782e-3]),
        high=np.array([30, 30, 6.1782e-3, 30, 6.1782e-3], dtype=np.float32),
    )

    trials = [
        Trial(
            target_beam=target_beam_space.sample(),
            incoming_beam=incoming_beam_space.sample(),
            misalignments=misalignment_space.sample(),
            initial_magnets=initial_magnet_space.sample,
        )
        for _ in range(num)
    ]

    return trials


def evaluate_policy(
    model: BaseAlgorithm, config: dict, write_data: bool = True, seed: int | None = None
) -> None:
    trials = generate_trials(num=20, seed=seed)

    for i, trial in enumerate(trials):
        try_problem(i, trial, model, config, write_data)

    # with ProcessPoolExecutor() as executor:
    #     _ = tqdm(
    #         executor.map(
    #             try_problem,
    #             range(len(trials)),
    #             trials,
    #             [model] * len(trials),
    #             [config] * len(trials),
    #             [write_data] * len(trials),
    #         ),
    #         total=len(trials),
    #     )


def main():
    trials = load_trials(Path("data/trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()

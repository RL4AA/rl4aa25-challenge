from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gymnasium.wrappers import (
    FilterObservation,
    FlattenObservation,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3 import TD3
from tqdm import tqdm

from src.environments import ea
from src.trial import Trial, load_trials
from src.wrappers import NotVecNormalize, PolishedDonkeyCompatibility, RecordEpisode


def try_problem(trial_index: int, trial: Trial, write_data: bool = True) -> None:
    model_name = "still-deluge-998"

    # Load the model
    model = TD3.load(f"models/ea/legacy/{model_name}/model")

    # Create the environment
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": trial.incoming_beam,
            "misalignment_mode": trial.misalignments,
        },
        action_mode="delta",
        magnet_init_mode=np.array([10, -10, 0, 10, 0]),
        max_quad_delta=30 * 0.1,
        max_steerer_delta=6e-3 * 0.1,
        target_beam_mode=trial.target_beam,
        target_threshold=None,
        threshold_hold=5,
        clip_magnets=False,
    )
    env = TimeLimit(env, 150)
    if write_data:
        env = RecordEpisode(
            env,
            save_dir=(f"data/dissertation/rl_still_deluge/problem_{trial_index:03d}"),
        )
    env = FilterObservation(env, ["beam", "magnets", "target"])
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"models/ea/legacy/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    # Actual optimisation
    observation, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


def main():
    trials = load_trials(Path("data/trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()

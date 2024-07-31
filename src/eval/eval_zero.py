from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

from src.environments import ea
from src.trial import Trial, load_trials
from src.wrappers import RecordEpisode


def try_problem(trial_index: int, trial: Trial, write_data: bool = True) -> None:
    # Create the environment
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": trial.incoming_beam,
            "misalignment_mode": trial.misalignments,
        },
        action_mode="delta",
        magnet_init_mode=np.zeros(5),
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
            save_dir=(f"data/dissertation/zero/problem_{trial_index:03d}"),
        )

    # Actual optimisation
    observation, info = env.reset()
    done = False
    while not done:
        action = np.zeros(5)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


def main():
    trials = load_trials(Path("data/trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()

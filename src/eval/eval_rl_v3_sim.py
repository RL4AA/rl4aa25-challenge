from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from gymnasium.wrappers import FlattenObservation, FrameStack, RescaleAction, TimeLimit
from stable_baselines3 import PPO
from tqdm import tqdm

from src.environments import ea
from src.trial import Trial, load_trials
from src.utils import load_config
from src.wrappers import RecordEpisode, RescaleObservation


def try_problem(trial_index: int, trial: Trial, write_data: bool = True) -> None:
    # model_name = "denim-spaceship-145"
    # model_name = "atomic-sweep-282"
    # model_name = "balmy-sweep-115"
    # model_name = "colorful-sponge-28"
    # model_name = "spring-music-454"
    model_name = "feasible-plasma-476"
    # model_name = "smooth-planet-477"

    # Load the model
    model = PPO.load(f"models/ea/ppo/{model_name}/model")
    config = load_config(f"models/ea/ppo/{model_name}/config")

    # Create the environment
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": trial.incoming_beam,
            "misalignment_mode": trial.misalignments,
        },
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=trial.target_beam,
        target_threshold=None,
        threshold_hold=5,
        clip_magnets=True,
        beam_param_transform=config["beam_param_transform"],
        beam_param_combiner=config["beam_param_combiner"],
        beam_param_combiner_args=config["beam_param_combiner_args"],
        beam_param_combiner_weights=config["beam_param_combiner_weights"],
        final_combiner=config["final_combiner"],
        final_combiner_args=config["final_combiner_args"],
        final_combiner_weights=config["final_combiner_weights"],
    )
    env = TimeLimit(env, 150)
    if write_data:
        env = RecordEpisode(
            env,
            save_dir=(
                f"data/bo_vs_rl/simulation/rl_feasible_plasma/problem_{trial_index:03d}"
            ),
        )
    if config["normalize_observation"] and not config["running_obs_norm"]:
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


def main():
    trials = load_trials(Path("data/trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()

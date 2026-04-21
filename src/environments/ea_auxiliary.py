from gymnasium.wrappers import FlattenObservation, RescaleAction, TimeLimit

from ..wrappers import RecordEpisode, RescaleObservation
from . import ea


def make_eval_env(config: dict):
    wrapper_config = config["env_wrapper"]
    env = ea.TransverseTuning(**config["env"])

    env = TimeLimit(env, max_episode_steps=wrapper_config["max_episode_steps"])
    env = RecordEpisode(env)
    if wrapper_config["normalize_observation"]:
        env = RescaleObservation(env, 0, 1)
    if wrapper_config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    env = FlattenObservation(env)
    return env

from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_checker import check_env

from src.environments import ea_auxiliary
from src.wrappers import FlattenAction


def test_check_env_cheetah():
    """Test SB3's `check_env` on the environment using the Cheetah backend."""
    env = ea_auxiliary.TransverseTuning(backend="cheetah")
    env = FlattenAction(env)
    env = RescaleAction(env, -1, 1)  # Prevents SB3 action space scale warning
    check_env(env)

import numpy as np

from src.environments import bc, ea
from src.wrappers import ReorderMagnets


def test_ea_larl():
    """
    Test if the magnet order works on when using a LARL policy on the EA environment.
    """
    reset_magnets = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    env = ea.TransverseTuning(
        backend="cheetah",
        action_mode="direct",
        magnet_init_mode=reset_magnets[[0, 1, 3, 2, 4]],
        clip_magnets=False,
    )
    env = ReorderMagnets(
        env,
        env_magnet_order=["Q1", "Q2", "CV", "Q3", "CH"],
        policy_magnet_order=["Q1", "Q2", "Q3", "CV", "CH"],
    )

    reset_obs, _ = env.reset()
    assert np.allclose(reset_obs["magnets"], reset_magnets)

    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    obs, _, _, _, _ = env.step(action)

    # Check that action and observation follow the same order from the policy's view
    assert np.allclose(obs["magnets"], action)

    # Check that the settings are written to the correct magnets
    assert action[0] == env.unwrapped.backend.segment.AREAMQZM1.k1
    assert action[1] == env.unwrapped.backend.segment.AREAMQZM2.k1
    assert action[2] == env.unwrapped.backend.segment.AREAMQZM3.k1
    assert action[3] == env.unwrapped.backend.segment.AREAMCVM1.angle
    assert action[4] == env.unwrapped.backend.segment.AREAMCHM1.angle


def test_bc_larl():
    """
    Test if the magnet order works on when using a LARL policy on the BC environment.
    """
    reset_magnets = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    env = bc.TransverseTuning(
        backend="cheetah",
        action_mode="direct",
        magnet_init_mode=reset_magnets[[0, 1, 3, 4, 2]],
        clip_magnets=False,
    )
    env = ReorderMagnets(
        env,
        env_magnet_order=["Q1", "Q2", "CV", "CH", "Q3"],
        policy_magnet_order=["Q1", "Q2", "Q3", "CV", "CH"],
    )

    reset_obs, _ = env.reset()
    assert np.allclose(reset_obs["magnets"], reset_magnets)

    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    obs, _, _, _, _ = env.step(action)

    # Check that action and observation follow the same order from the policy's view
    assert np.allclose(obs["magnets"], action)

    # Check that the settings are written to the correct magnets
    assert action[0] == env.unwrapped.backend.segment.ARMRMQZM4.k1
    assert action[1] == env.unwrapped.backend.segment.ARMRMQZM5.k1
    assert action[2] == env.unwrapped.backend.segment.ARMRMQZM6.k1
    assert action[3] == env.unwrapped.backend.segment.ARMRMCVM5.angle
    assert action[4] == env.unwrapped.backend.segment.ARMRMCHM5.angle

import numpy as np
import pytest

from src.environments import ea


@pytest.mark.parametrize("section", [pytest.param(ea, marks=pytest.mark.ea)])
@pytest.mark.skip(reason="Random seeds are not fixed yet")
def test_seed(section):
    """
    Test that using a fixed seed produces reproducible initial magnet settings and
    target beams, while different seeds produce different values.
    """
    env = section.TransverseTuning(
        backend="cheetah",
        backend_args={"incoming_mode": "random", "misalignment_mode": "random"},
    )
    _, info_ref = env.reset(seed=42)
    _, info_same = env.reset(seed=42)
    _, info_diff = env.reset(seed=24)

    # Incoming beam
    assert all(info_ref["incoming"] == info_same["incoming"])
    assert all(info_ref["incoming"] != info_diff["incoming"])

    # Misalignments
    assert all(info_ref["misalignments"] == info_same["misalignments"])
    assert all(info_ref["misalignments"] != info_diff["misalignments"])


@pytest.mark.parametrize(
    "section, misalignments", [pytest.param(ea, np.zeros(8), marks=pytest.mark.ea)]
)
def test_cheetah_fixed_incoming_mode_array_list(section, misalignments):
    """
    Test that if fixed values are set for `incoming` using an np.ndarray or a list
    results in the same values behaviour.
    """
    incoming_example = [
        1.47126736e08,
        2.34137638e-04,
        8.38419946e-05,
        8.83190442e-05,
        1.25623028e-05,
        3.93863709e-04,
        1.36142835e-05,
        9.13740369e-05,
        4.38809075e-05,
        3.61393245e-06,
        7.29136518e-04,
    ]

    env_array = section.TransverseTuning(
        backend="cheetah",
        magnet_init_mode=None,
        backend_args={
            "incoming_mode": np.array(incoming_example),
            "misalignment_mode": misalignments,
        },
    )
    observation_array_first_reset, _ = env_array.reset()
    _, _, _, _, _ = env_array.step(np.zeros(env_array.action_space.shape))
    observation_array_second_reset, _ = env_array.reset()

    env_list = section.TransverseTuning(
        backend="cheetah",
        magnet_init_mode=None,
        backend_args={
            "incoming_mode": incoming_example,
            "misalignment_mode": misalignments,
        },
    )
    observation_list_first_reset, _ = env_list.reset()
    _, _, _, _, _ = env_list.step(np.zeros(env_list.action_space.shape))
    observation_list_second_reset, _ = env_list.reset()

    assert np.allclose(
        observation_array_first_reset["beam"], observation_array_second_reset["beam"]
    )
    assert np.allclose(
        observation_array_first_reset["beam"], observation_list_first_reset["beam"]
    )
    assert np.allclose(
        observation_array_first_reset["beam"], observation_list_second_reset["beam"]
    )


@pytest.mark.parametrize(
    "section, settings",
    [
        pytest.param(
            ea, [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5], marks=pytest.mark.ea
        ),
    ],
)
def test_cheetah_fixed_misalignment_mode_array_list(section, settings):
    """
    Test that if fixed values are set for the misalignments using an np.ndarray or a
    list results in the same values behaviour.
    """
    incoming_example = [
        1.47126736e08,
        2.34137638e-04,
        8.38419946e-05,
        8.83190442e-05,
        1.25623028e-05,
        3.93863709e-04,
        1.36142835e-05,
        9.13740369e-05,
        4.38809075e-05,
        3.61393245e-06,
        7.29136518e-04,
    ]

    env_array = section.TransverseTuning(
        backend="cheetah",
        magnet_init_mode=np.array([10, -10, 0, 10, 0]),
        backend_args={
            "incoming_mode": np.array(incoming_example),
            "misalignment_mode": np.array(settings),
        },
    )
    observation_array_first_reset, _ = env_array.reset()
    _, _, _, _, _ = env_array.step(np.zeros(env_array.action_space.shape))
    observation_array_second_reset, _ = env_array.reset()

    env_list = section.TransverseTuning(
        backend="cheetah",
        magnet_init_mode=np.array([10, -10, 0, 10, 0]),
        backend_args={
            "incoming_mode": np.array(incoming_example),
            "misalignment_mode": settings,
        },
    )
    observation_list_first_reset, _ = env_list.reset()
    _, _, _, _, _ = env_list.step(np.zeros(env_list.action_space.shape))
    observation_list_second_reset, _ = env_list.reset()

    assert np.allclose(
        observation_array_first_reset["beam"], observation_array_second_reset["beam"]
    )
    assert np.allclose(
        observation_array_first_reset["beam"], observation_list_first_reset["beam"]
    )
    assert np.allclose(
        observation_array_first_reset["beam"], observation_list_second_reset["beam"]
    )

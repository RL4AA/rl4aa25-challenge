import inspect

import numpy as np
import pytest
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_checker import check_env

from src.environments import bc, dl, ea, sh
from src.reward import combiners, transforms


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_check_env_cheetah(section):
    """Test SB3's `check_env` on all environments using their Cheetah backends."""
    env = section.TransverseTuning(backend="cheetah")
    env = RescaleAction(env, -1, 1)  # Prevents SB3 action space scale warning
    check_env(env)


@pytest.mark.ocelot
@pytest.mark.ea
def test_check_env_ocelot():
    """
    Test SB3's `check_env` on all environments using their Ocelot backends.

    NOTE Only the EA environment currently has an Ocelot backend.
    """
    env = ea.TransverseTuning(backend="ocelot")
    env = RescaleAction(env, -1, 1)  # Prevents SB3 action space scale warning
    check_env(env)


@pytest.mark.doocs
@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_check_env_doocs(section):
    """Test SB3's `check_env` on all environments using their DOOCS backends."""
    env = section.TransverseTuning(backend="doocs_dummy")
    env = RescaleAction(env, -1, 1)  # Prevents SB3 action space scale warning
    check_env(env)


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_mandatory_backend_argument(section):
    """Test that the `backend` argument is mandatory."""
    with pytest.raises(TypeError):
        section.TransverseTuning(
            # backend="cheetah"
        )


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_passing_backend_args(section):
    """
    Test that backend_args are passed through the environment to the backend correctly.
    """
    incoming_mode = np.array(
        [
            160e6,
            1e-3,
            1e-4,
            1e-3,
            1e-4,
            5e-4,
            5e-5,
            5e-4,
            5e-5,
            5e-5,
            1e-3,
        ]
    )
    if section in [ea, bc]:  # EA and BC with 3 quadrupoles + screen
        misalignment_mode = np.array([-1e-4, 1e-4, -1e-5, 1e-5, 3e-4, 0, -3e-4, 9e-5])
    else:  # DL and SH with 2 quadrupoles + screen
        misalignment_mode = np.array([-1e-4, 1e-4, -1e-5, 1e-5, -3e-4, 9e-5])

    simulate_finite_screen = True

    env = section.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": incoming_mode,
            "misalignment_mode": misalignment_mode,
            "simulate_finite_screen": simulate_finite_screen,
        },
    )

    # Test that config is passed through to backend
    assert all(env.unwrapped.backend.incoming_mode == incoming_mode)
    assert all(env.unwrapped.backend.misalignment_mode == misalignment_mode)
    assert env.unwrapped.backend.simulate_finite_screen == simulate_finite_screen

    # Test that configs are used correctly
    _, _ = env.reset()
    incoming_parameters = np.array(
        [
            env.unwrapped.backend.incoming.parameters["energy"],
            env.unwrapped.backend.incoming.parameters["mu_x"],
            env.unwrapped.backend.incoming.parameters["mu_xp"],
            env.unwrapped.backend.incoming.parameters["mu_y"],
            env.unwrapped.backend.incoming.parameters["mu_yp"],
            env.unwrapped.backend.incoming.parameters["sigma_x"],
            env.unwrapped.backend.incoming.parameters["sigma_xp"],
            env.unwrapped.backend.incoming.parameters["sigma_y"],
            env.unwrapped.backend.incoming.parameters["sigma_yp"],
            env.unwrapped.backend.incoming.parameters["sigma_s"],
            env.unwrapped.backend.incoming.parameters["sigma_p"],
        ]
    )

    assert np.allclose(incoming_parameters, incoming_mode)
    assert np.allclose(env.unwrapped.backend.get_misalignments(), misalignment_mode)


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_public_members(section):
    """
    Make sure that all and only intended members are exposed to the user (named withouth
    leading underscore).
    """
    gymnasium_public_members = [
        "reset",
        "step",
        "render",
        "close",
        "action_space",
        "observation_space",
        "metadata",
        "np_random",
        "render_mode",
        "reward_range",
        "spec",
        "unwrapped",
        "get_wrapper_attr",
    ]
    custom_public_members = [
        "backend",
        "action_mode",
        "magnet_init_mode",
        "max_quad_delta",
        "max_steerer_delta",
        "target_beam_mode",
        "target_threshold",
        "threshold_hold",
        "unidirectional_quads",
        "clip_magnets",
    ]
    allowed_public_members = gymnasium_public_members + custom_public_members

    env = section.TransverseTuning(backend="cheetah")
    _, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())

    members = dir(env)
    public_members = [m for m in members if not m.startswith("_")]

    for member in public_members:
        assert member in allowed_public_members

        # Remove member from list of allowed members
        allowed_public_members.remove(member)


def test_same_members():
    """
    Test that all environment classes have the same members. This should give at least
    some indication, when one implementation differs in priciple from the others.
    """
    sections = [ea, dl, bc, sh]

    members = []
    for section in sections:
        env = section.TransverseTuning(backend="cheetah")
        _, _ = env.reset()
        _, _, _, _, _ = env.step(env.action_space.sample())
        members.append(dir(env))

    for member in members[0]:
        for other_members in members[1:]:
            assert member in other_members


def test_same_argspec():
    """
    To make sure that all four section environments are implemented in the same way,
    test that their `__init__` methods have the same arguments.
    """
    sections = [ea, dl, bc, sh]

    argspecs = [
        inspect.getfullargspec(section.TransverseTuning) for section in sections
    ]

    print(f"{argspecs[0].args = }")
    for argspec in argspecs[1:]:
        print(f"{argspec.args = }")
        assert argspec.args == argspecs[0].args


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
@pytest.mark.skip(reason="Random seeds are not fixed yet")
def test_seed(section):
    """
    Test that using a fixed seed produces reproducible initial magnet settings and
    target beams, while different seeds produce different values.
    """
    env = section.TransverseTuning(
        backend="cheetah", magnet_init_mode="random", target_beam_mode="random"
    )
    observation_ref, _ = env.reset(seed=42)
    observation_same, _ = env.reset(seed=42)
    observation_diff, _ = env.reset(seed=24)

    # Magnet settings
    assert all(observation_ref["magnets"] == observation_same["magnets"])
    assert all(observation_ref["magnets"] != observation_diff["magnets"])

    # Target beams
    assert all(observation_ref["target"] == observation_same["target"])
    assert all(observation_ref["target"] != observation_diff["target"])


@pytest.mark.doocs
@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_doocs_before_reset_infos(section):
    """
    Test that the DOOCS backend returns magnet settings, beam image and beam parameters
    from before the reset, when a magnet ititialisation is configured.
    """
    env = section.TransverseTuning(backend="doocs_dummy", magnet_init_mode="random")
    _, info = env.reset()
    backend_info = info["backend_info"]

    assert "magnets_before_reset" in backend_info
    assert isinstance(backend_info["magnets_before_reset"], np.ndarray)

    assert "screen_before_reset" in backend_info
    assert isinstance(backend_info["screen_before_reset"], np.ndarray)

    assert "beam_before_reset" in backend_info
    assert isinstance(backend_info["beam_before_reset"], np.ndarray)


@pytest.mark.doocs
@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_doocs_before_reset_infos_disappear(section):
    """
    Test that magnet settings, beam image and beam parameters from before the reset,
    disappear from `info` after the first step, when a magnet ititialisation is
    configured.
    """
    env = section.TransverseTuning(backend="doocs_dummy", magnet_init_mode="random")
    _, _ = env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    backend_info = info["backend_info"]

    assert "magnets_before_reset" not in backend_info
    assert "screen_before_reset" not in backend_info
    assert "beam_before_reset" not in backend_info


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_transform_combiner_passing(section):
    """
    Test at an example that the beam parameter transform, beam parameter combiner and
    final combiner are correctly setup according to the arguments passed to the
    environment.
    """
    env = section.TransverseTuning(
        backend="cheetah",
        beam_param_transform="Sigmoid",
        beam_param_combiner="SmoothMax",
        beam_param_combiner_args={"alpha": -5},
        beam_param_combiner_weights=[1, 1, 1, 1],
        magnet_change_transform="Sigmoid",
        magnet_change_combiner="Mean",
        magnet_change_combiner_args={},
        magnet_change_combiner_weights=[1, 1, 1, 1, 1],
        final_combiner="SmoothMax",
        final_combiner_args={"alpha": -5},
        final_combiner_weights=[1, 1],
    )

    # Test beam parameter transform
    assert isinstance(env._beam_param_transform, transforms.Sigmoid)

    # Test beam parameter combiner
    assert isinstance(env._beam_param_combiner, combiners.SmoothMax)
    assert env._beam_param_combiner.alpha == -5
    assert env._beam_param_combiner_weights == [1, 1, 1, 1]

    # Test magnet change transform
    assert isinstance(env._magnet_change_transform, transforms.Sigmoid)

    # Test magnet change combiner
    assert isinstance(env._magnet_change_combiner, combiners.Mean)
    assert env._magnet_change_combiner_weights == [1, 1, 1, 1, 1]

    # Test final combiner
    assert isinstance(env._final_combiner, combiners.SmoothMax)
    assert env._final_combiner.alpha == -5
    assert env._final_combiner_weights == [1, 1]


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_magnet_clipping_direct(section):
    """
    Test that magnet settings are clipped to the allowed range when the action mode is
    set to "direct".
    """
    env = section.TransverseTuning(
        backend="cheetah",
        action_mode="direct",
        clip_magnets=True,
    )
    min_magnet_settings = env.observation_space["magnets"].low
    max_magnet_settings = env.observation_space["magnets"].high

    _, _ = env.reset()
    observation, _, _, _, _ = env.step(max_magnet_settings * 2)

    assert all(observation["magnets"] >= min_magnet_settings)
    assert all(observation["magnets"] <= max_magnet_settings)


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_magnet_clipping_delta(section):
    """
    Test that magnet settings are clipped to the allowed range when the action mode is
    set to "delta".
    """
    env = section.TransverseTuning(
        backend="cheetah",
        action_mode="direct",
        clip_magnets=True,
    )
    min_magnet_settings = env.observation_space["magnets"].low
    max_magnet_settings = env.observation_space["magnets"].high

    env.reset(options={"magnet_init": max_magnet_settings * 0.5})
    observation, _, _, _, _ = env.step(max_magnet_settings)

    assert all(observation["magnets"] >= min_magnet_settings)
    assert all(observation["magnets"] <= max_magnet_settings)


@pytest.mark.parametrize(
    "section, settings",
    [
        pytest.param(ea, [1.0, 2.0, 1e-3, 3.0, 2e-3], marks=pytest.mark.ea),
        pytest.param(dl, [1e-3, 2e-3, 1.0, 2.0], marks=pytest.mark.dl),
        pytest.param(bc, [1.0, 2.0, 1e-3, 2e-3, 3.0], marks=pytest.mark.bc),
        pytest.param(sh, [1e-3, 1.0, 2e-3, 2.0], marks=pytest.mark.sh),
    ],
)
def test_fixed_magnet_init_mode_array(section, settings):
    """
    Test that if fixed values are set for `magnet_init_mode`, the magnets are in fact
    set to these values. This tests checks two consecutive resets. It considers the
    initials values to be set as a NumPy array.
    """
    env = section.TransverseTuning(
        backend="cheetah", magnet_init_mode=np.array(settings)
    )
    observation_first_reset, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())
    observation_second_reset, _ = env.reset()

    assert np.allclose(observation_first_reset["magnets"], np.array(settings))
    assert np.allclose(observation_second_reset["magnets"], np.array(settings))


@pytest.mark.parametrize(
    "section, settings",
    [
        pytest.param(ea, [1.0, 2.0, 1e-3, 3.0, 2e-3], marks=pytest.mark.ea),
        pytest.param(dl, [1e-3, 2e-3, 1.0, 2.0], marks=pytest.mark.dl),
        pytest.param(bc, [1.0, 2.0, 1e-3, 2e-3, 3.0], marks=pytest.mark.bc),
        pytest.param(sh, [1e-3, 1.0, 2e-3, 2.0], marks=pytest.mark.sh),
    ],
)
def test_fixed_magnet_init_mode_list(section, settings):
    """
    Test that if fixed values are set for `magnet_init_mode`, the magnets are in fact
    set to these values. This tests checks two consecutive resets. It considers the
    initials values to be set as a Python list.
    """
    env = section.TransverseTuning(backend="cheetah", magnet_init_mode=settings)
    observation_first_reset, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())
    observation_second_reset, _ = env.reset()

    assert np.allclose(observation_first_reset["magnets"], np.array(settings))
    assert np.allclose(observation_second_reset["magnets"], np.array(settings))


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_fixed_target_beam_mode_array(section):
    """
    Test that if fixed values are set for `target_beam_mode`, the target beam is in fact
    set to these values. This tests checks two consecutive resets. It considers the
    values to be set as a NumPy array.
    """
    settings = [2e-5, 3e-5, 4e-5, 5e-5]

    env = section.TransverseTuning(
        backend="cheetah", target_beam_mode=np.array(settings)
    )
    observation_first_reset, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())
    observation_second_reset, _ = env.reset()

    assert np.allclose(observation_first_reset["target"], np.array(settings))
    assert np.allclose(observation_second_reset["target"], np.array(settings))


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_fixed_target_beam_mode_list(section):
    """
    Test that if fixed values are set for `target_beam_mode`, the target beam is in fact
    set to these values. This tests checks two consecutive resets. It considers the
    values to be set as a Python list.
    """
    settings = [2e-5, 3e-5, 4e-5, 5e-5]

    env = section.TransverseTuning(backend="cheetah", target_beam_mode=settings)
    observation_first_reset, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())
    observation_second_reset, _ = env.reset()

    assert np.allclose(observation_first_reset["target"], np.array(settings))
    assert np.allclose(observation_second_reset["target"], np.array(settings))


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_fixed_target_threshold_array(section):
    """
    Test that if fixed values are set for `target_threshold`, the target threshold is in
    fact set to these values. This tests considers the values to be set as a NumPy
    array.
    """
    settings = [2e-5, 3e-5, 4e-5, 5e-5]

    env = section.TransverseTuning(
        backend="cheetah", target_threshold=np.array(settings)
    )
    observation_first_reset, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())
    _, _, _, _, _ = env.step(env.action_space.sample())
    observation_second_reset, _ = env.reset()

    assert np.allclose(env.target_threshold, np.array(settings))


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(ea, marks=pytest.mark.ea),
        pytest.param(dl, marks=pytest.mark.dl),
        pytest.param(bc, marks=pytest.mark.bc),
        pytest.param(sh, marks=pytest.mark.sh),
    ],
)
def test_fixed_target_threshold_list(section):
    """
    Test that if fixed values are set for `target_threshold`, the target threshold is in
    fact set to these values. This tests considers the values to be set as a list.
    """
    settings = [2e-5, 3e-5, 4e-5, 5e-5]

    env = section.TransverseTuning(backend="cheetah", target_threshold=settings)
    observation_first_reset, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())
    _, _, _, _, _ = env.step(env.action_space.sample())
    observation_second_reset, _ = env.reset()

    assert np.allclose(env.target_threshold, np.array(settings))

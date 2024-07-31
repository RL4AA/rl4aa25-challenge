import pytest
from gymnasium.wrappers import RecordVideo, RescaleAction, TimeLimit
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import unwrap_wrapper

from src.environments import ea
from src.wrappers import ARESeLog

# TODO Test that episode trigger behaves like RecordVideo


@pytest.mark.parametrize("section", [ea])
@pytest.mark.parametrize(
    "backend", ["cheetah", pytest.param("doocs_dummy", marks=pytest.mark.doocs)]
)
def test_check_env(section, backend):
    """Test that the `ARESeLog` wrapper throws no exceptions under `check_env`."""
    env = section.TransverseTuning(
        backend=backend,
        backend_args={"generate_screen_images": True} if backend == "cheetah" else {},
    )
    env = ARESeLog(env)
    env = RescaleAction(env, -1, 1)

    check_env(env)


def test_trigger_like_record_video(tmp_path):
    """
    Test that, given the same trigger function, the `ARESeLog` wrapper records the
    same episodes as the `RecordVideo` wrapper from Gymnasium.
    """
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={"generate_screen_images": True},
        render_mode="rgb_array",
    )
    env = TimeLimit(env, 10)
    env = ARESeLog(env, episode_trigger=lambda x: x % 5 == 0)
    env = RecordVideo(
        env,
        video_folder=str(tmp_path / "recordings"),
        episode_trigger=lambda x: x % 5 == 0,
    )

    plot_episode = unwrap_wrapper(env, ARESeLog)
    record_video = unwrap_wrapper(env, RecordVideo)

    for i in range(10):
        _, _ = env.reset()
        assert plot_episode.episode_id == i
        assert record_video.episode_id == i

        assert plot_episode.is_recording == (i % 5 == 0)
        assert plot_episode.is_recording == record_video.recording

        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


def test_episode_id_advanced(tmp_path):
    """
    Test that the episode ID advances in the same way as it does in the `RecordVideo`
    wrapper from Gymnasium.
    """
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={"generate_screen_images": True},
        render_mode="rgb_array",
    )
    env = TimeLimit(env, 10)
    env = ARESeLog(env, episode_trigger=lambda x: x % 5 == 0)
    env = RecordVideo(
        env,
        video_folder=str(tmp_path / "recordings"),
        episode_trigger=lambda x: x % 5 == 0,
    )

    plot_episode = unwrap_wrapper(env, ARESeLog)
    record_video = unwrap_wrapper(env, RecordVideo)

    # Test normal case where episode was terminated or truncated
    for _ in range(10):
        _, _ = env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
        assert plot_episode.episode_id == record_video.episode_id

    # Test abnormal case where episode is just run for some steps and then reset
    for _ in range(10):
        _, _ = env.reset()
        for _ in range(5):
            _, _, _, _, _ = env.step(env.action_space.sample())
        assert plot_episode.episode_id == record_video.episode_id

    # To supress unnecessary warnings
    env.close()

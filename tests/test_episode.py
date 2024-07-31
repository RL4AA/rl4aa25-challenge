from gymnasium.wrappers import TimeLimit

from src.environments import ea
from src.eval import Episode
from src.wrappers import RecordEpisode


def test_save_then_load_equal(tmp_path):
    """
    Test that if you save an an episode and then load it again, you get the same episde.
    """
    # Sart by generating an episode
    env = ea.TransverseTuning(backend="cheetah")
    env = TimeLimit(env, 10)
    env = RecordEpisode(
        env, save_dir=str(tmp_path), name_prefix="test_save_then_load_equal"
    )
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()

    # Load the episode
    episode_before = Episode.load(tmp_path / "test_save_then_load_equal_1.pkl")

    # Save the episode
    episode_before.save(tmp_path / "test_save_then_load_equal_1b.pkl")

    # Load the episode again
    episode_after = Episode.load(tmp_path / "test_save_then_load_equal_1b.pkl")

    # Check that the episodes are equal
    assert episode_after == episode_before

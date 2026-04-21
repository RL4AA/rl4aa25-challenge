from pathlib import Path

from src.eval import eval_bo_rl_opponent_sim, eval_rl_v3_ea_sim
from src.trial import load_trials


def test_polished_donkey():
    """Test that the simulation evaluation for polished donkey works."""
    trial = load_trials(Path("data/trials.yaml"))[0]

    eval_bo_rl_opponent_sim.try_problem(trial_index=0, trial=trial, write_data=False)


def test_v3_agent():
    """Test that the simulation evaluation for v3 agent works."""
    trial = load_trials(Path("data/trials.yaml"))[0]

    eval_rl_v3_ea_sim.try_problem(trial_index=0, trial=trial, write_data=False)

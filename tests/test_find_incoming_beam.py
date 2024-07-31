from pathlib import Path

import cheetah
import numpy as np
import pytest
import torch

from src.eval.eval_bo_sim_quad_aligned import find_quad_aligned_incoming_beam_parameters
from src.trial import load_trials


def test_find_incoming_beam_for_zero_misalignment():
    trial = load_trials(Path("data/trials.yaml"))[0]
    trial.misalignments = np.zeros(8)

    new_incoming_beam = find_quad_aligned_incoming_beam_parameters(trial)

    assert all(new_incoming_beam[[1, 2, 3, 4]] == 0)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_find_incoming_beam_for_same_offset_misalignment(direction):
    trial = load_trials(Path("data/trials.yaml"))[0]
    trial.misalignments = direction * np.ones(8)

    new_incoming_beam = find_quad_aligned_incoming_beam_parameters(trial)

    correct_offset = all(new_incoming_beam[[1, 3]] == direction)
    no_transverse_momentum = all(new_incoming_beam[[2, 4]] == 0)
    assert correct_offset and no_transverse_momentum


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_find_incoming_beam_for_increasing_offset_misalignment(direction):
    trial = load_trials(Path("data/trials.yaml"))[0]
    trial.misalignments = direction * np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0])

    new_incoming_beam = find_quad_aligned_incoming_beam_parameters(trial)

    transverse_momentums = new_incoming_beam[[2, 4]]
    assert all(transverse_momentums > 0 if direction == 1 else transverse_momentums < 0)


def test_find_incoming_beam_slope_to_momentum():
    trial = load_trials(Path("data/trials.yaml"))[0]
    trial.misalignments = np.array(
        [
            200e-6,
            200e-6,
            200e-6 / 0.24 * 0.79,
            200e-6 / 0.24 * 0.79,
            200e-6 / 0.24 * 1.33,
            200e-6 / 0.24 * 1.33,
            0,
            0,
        ]
    )
    assumed_slope = 200e-6 / 0.24

    new_beam_parameters = find_quad_aligned_incoming_beam_parameters(trial)

    cheetah_beam = cheetah.ParameterBeam.from_parameters(
        energy=torch.tensor(new_beam_parameters[0], dtype=torch.float32),
        mu_x=torch.tensor(new_beam_parameters[1], dtype=torch.float32),
        mu_xp=torch.tensor(new_beam_parameters[2], dtype=torch.float32),
        mu_y=torch.tensor(new_beam_parameters[3], dtype=torch.float32),
        mu_yp=torch.tensor(new_beam_parameters[4], dtype=torch.float32),
        sigma_x=torch.tensor(new_beam_parameters[5], dtype=torch.float32),
        sigma_xp=torch.tensor(new_beam_parameters[6], dtype=torch.float32),
        sigma_y=torch.tensor(new_beam_parameters[7], dtype=torch.float32),
        sigma_yp=torch.tensor(new_beam_parameters[8], dtype=torch.float32),
        sigma_s=torch.tensor(new_beam_parameters[9], dtype=torch.float32),
        sigma_p=torch.tensor(new_beam_parameters[10], dtype=torch.float32),
    )
    drifted_beam = cheetah.Drift(length=torch.tensor(1.0))(cheetah_beam)

    offset_zero = np.isclose(cheetah_beam.mu_x, 0) and np.isclose(cheetah_beam.mu_y, 0)
    slope_matches_momentum = np.isclose(
        drifted_beam.mu_x, assumed_slope
    ) and np.isclose(drifted_beam.mu_y, assumed_slope)

    assert offset_zero and slope_matches_momentum

"""
Tests for combiners.

NOTE: This file was copied and modified from Deepmind's fusion paper code:
https://github.com/deepmind/deepmind-research/blob/master/fusion_tcv/combiners_test.py
"""

import math

import numpy as np
import pytest

from src.reward import combiners

NAN = float("nan")


def test_errors():
    c = combiners.Mean()
    with pytest.raises(ValueError):
        c([0, 1], [1])
    with pytest.raises(ValueError):
        c([0, 1], [1, 2, 3])
    with pytest.raises(ValueError):
        c([0, 1], [-1, 2])


def test_mean():
    c = combiners.Mean()
    assert c([0, 2, 4]) == 2
    assert c([0, 0.5, 1]) == 0.5
    assert c([0, 0.5, 1], [0, 0, 1]) == 1
    assert c([0, 1], [1, 3]) == 0.75
    assert c([0, NAN], [1, 3]) == 0
    assert math.isnan(c([NAN, NAN], [1, 3]))


def test_geometric_mean():
    c = combiners.GeometricMean()
    assert c([0.5, 0]) == 0
    assert np.isclose(c([0.3]), 0.3)
    assert c([4, 4]) == 4
    assert c([0.5, 0.5]) == 0.5
    assert c([0.5, 0.5], [1, 3]) == 0.5
    assert c([0.5, 1], [1, 2]) == 0.5 ** (1 / 3)
    assert c([0.5, 1], [2, 1]) == 0.5 ** (2 / 3)
    assert c([0.5, 0], [2, 0]) == 0.5
    assert c([0.5, 0, 0], [2, 1, 0]) == 0
    assert c([0.5, NAN, 0], [2, 1, 0]) == 0.5
    assert math.isnan(c([NAN, NAN], [1, 3]))


def test_multiply():
    c = combiners.Multiply()
    assert c([0.5, 0]) == 0
    assert np.isclose(c([0.3]), 0.3)
    assert c([0.5, 0.5]) == 0.25
    assert c([0.5, 0.5], [1, 3]) == 0.0625
    assert c([0.5, 1], [1, 2]) == 0.5
    assert c([0.5, 1], [2, 1]) == 0.25
    assert c([0.5, 0], [2, 0]) == 0.25
    assert c([0.5, 0, 0], [2, 1, 0]) == 0
    assert c([0.5, NAN], [1, 1]) == 0.5
    assert math.isnan(c([NAN, NAN], [1, 3]))


def test_min():
    c = combiners.Min()
    assert c([0, 1]) == 0
    assert c([0.5, 1]) == 0.5
    assert c([1, 0.75]) == 0.75
    assert c([1, 3]) == 1
    assert c([1, 1, 3], [0, 1, 1]) == 1
    assert c([NAN, 3]) == 3
    assert math.isnan(c([NAN, NAN], [1, 3]))


def test_max():
    c = combiners.Max()
    assert c([0, 1]) == 1
    assert c([0.5, 1]) == 1
    assert c([1, 0.75]) == 1
    assert c([1, 3]) == 3
    assert c([1, 1, 3], [0, 1, 1]) == 3
    assert c([NAN, 3]) == 3
    assert math.isnan(c([NAN, NAN], [1, 3]))


def test_lnorm():
    c = combiners.LNorm(1)
    assert c([0, 2, 4]) == 2
    assert c([0, 0.5, 1]) == 0.5
    assert c([3, 4]) == 7 / 2
    assert c([0, 2, 4], [1, 1, 0]) == 1
    assert c([0, 2, NAN]) == 1
    assert math.isnan(c([NAN, NAN], [1, 3]))

    c = combiners.LNorm(1, normalized=False)
    assert c([0, 2, 4]) == 6
    assert c([0, 0.5, 1]) == 1.5
    assert c([3, 4]) == 7

    c = combiners.LNorm(2)
    assert c([3, 4]) == 5 / 2**0.5

    c = combiners.LNorm(2, normalized=False)
    assert c([3, 4]) == 5

    c = combiners.LNorm(math.inf)
    assert np.isclose(c([3, 4]), 4)

    c = combiners.LNorm(math.inf, normalized=False)
    assert np.isclose(c([3, 4]), 4)


def test_smoothmax():
    # Max
    c = combiners.SmoothMax(math.inf)
    assert c([0, 1]) == 1
    assert c([0.5, 1]) == 1
    assert c([1, 0.75]) == 1
    assert c([1, 3]) == 3

    # Smooth Max
    c = combiners.SmoothMax(1)
    assert np.isclose(c([0, 1]), 0.7310585786300049)

    # Mean
    c = combiners.SmoothMax(0)
    assert c([0, 2, 4]) == 2
    assert c([0, 0.5, 1]) == 0.5
    assert c([0, 0.5, 1], [0, 0, 1]) == 1
    assert c([0, 2, NAN]) == 1
    assert c([0, 2, NAN], [0, 1, 1]) == 2
    assert np.isclose(c([0, 1], [1, 3]), 0.75)
    assert math.isnan(c([NAN, NAN], [1, 3]))

    # Smooth Min
    c = combiners.SmoothMax(-1)
    assert c([0, 1]) == 0.2689414213699951

    # Min
    c = combiners.SmoothMax(-math.inf)
    assert c([0, 1]) == 0
    assert c([0.5, 1]) == 0.5
    assert c([1, 0.75]) == 0.75
    assert c([1, 3]) == 1

"""
Tests for transforms.

NOTE: This file was copied and modified from Deepmind's fusion paper code:
https://github.com/deepmind/deepmind-research/blob/master/fusion_tcv/transforms_test.py
"""

import math

import numpy as np

from src.reward import transforms

NAN = float("nan")


def test_clip():
    assert transforms.clip(-1, 0, 1) == 0
    assert transforms.clip(5, 0, 1) == 1
    assert transforms.clip(0.5, 0, 1) == 0.5
    assert math.isnan(transforms.clip(NAN, 0, 1))


def test_scale():
    assert transforms.scale(0, 0, 0.5, 0, 1) == 0
    assert transforms.scale(0.125, 0, 0.5, 0, 1) == 0.25
    assert transforms.scale(0.25, 0, 0.5, 0, 1) == 0.5
    assert transforms.scale(0.5, 0, 0.5, 0, 1) == 1
    assert transforms.scale(1, 0, 0.5, 0, 1) == 2
    assert transforms.scale(-1, 0, 0.5, 0, 1) == -2
    assert transforms.scale(0.5, 1, 0, 0, 1) == 0.5
    assert transforms.scale(0.25, 1, 0, 0, 1) == 0.75

    assert transforms.scale(0, 0, 1, -4, 4) == -4
    assert transforms.scale(0.25, 0, 1, -4, 4) == -2
    assert transforms.scale(0.5, 0, 1, -4, 4) == 0
    assert transforms.scale(0.75, 0, 1, -4, 4) == 2
    assert transforms.scale(1, 0, 1, -4, 4) == 4

    assert math.isnan(transforms.scale(NAN, 0, 1, -4, 4))


def test_logistic():
    assert transforms.logistic(-50) < 0.000001
    assert transforms.logistic(-5) < 0.01
    assert transforms.logistic(0) == 0.5
    assert transforms.logistic(5) > 0.99
    assert transforms.logistic(50) > 0.999999
    assert transforms.logistic(0.8) == math.tanh(0.4) / 2 + 0.5
    assert math.isnan(transforms.logistic(NAN))


def test_exp_scaled():
    t = transforms.NegExp(good=0, bad=1)
    assert math.isnan(t([NAN])[0])
    assert np.isclose(t([0])[0], 1)
    assert np.isclose(t([1])[0], 0.1)
    assert t([50])[0] < 0.000001

    t = transforms.NegExp(good=10, bad=30)
    assert np.isclose(t([0])[0], 1)
    assert np.isclose(t([10])[0], 1)
    assert t([3000])[0] < 0.000001

    t = transforms.NegExp(good=30, bad=10)
    assert np.isclose(t([50])[0], 1)
    assert np.isclose(t([30])[0], 1)
    assert np.isclose(t([10])[0], 0.1)
    assert t([-90])[0] < 0.00001


def test_neg():
    t = transforms.Neg()
    assert t([-5, -3, 0, 1, 4]) == [5, 3, 0, -1, -4]
    assert math.isnan(t([NAN])[0])


def test_abs():
    t = transforms.Abs()
    assert t([-5, -3, 0, 1, 4]) == [5, 3, 0, 1, 4]
    assert math.isnan(t([NAN])[0])


def test_pow():
    t = transforms.Pow(2)
    assert t([-5, -3, 0, 1, 4]) == [25, 9, 0, 1, 16]
    assert math.isnan(t([NAN])[0])


def test_log():
    t = transforms.Log()
    # NOTE In the original of the line below it said something like precision = 4
    assert np.isclose(t([math.exp(2)])[0], 2)
    assert math.isnan(t([NAN])[0])


def test_clipped_linear():
    t = transforms.ClippedLinear(good=0.1, bad=0.3)
    assert np.isclose(t([0])[0], 1)
    assert np.isclose(t([0.05])[0], 1)
    assert np.isclose(t([0.1])[0], 1)
    assert np.isclose(t([0.15])[0], 0.75)
    assert np.isclose(t([0.2])[0], 0.5)
    assert np.isclose(t([0.25])[0], 0.25)
    assert np.isclose(t([0.3])[0], 0)
    assert np.isclose(t([0.4])[0], 0)
    assert math.isnan(t([NAN])[0])

    t = transforms.ClippedLinear(good=1, bad=0.5)
    assert np.isclose(t([1.5])[0], 1)
    assert np.isclose(t([1])[0], 1)
    assert np.isclose(t([0.75])[0], 0.5)
    assert np.isclose(t([0.5])[0], 0)
    assert np.isclose(t([0.25])[0], 0)


def test_softplus():
    t = transforms.SoftPlus(good=0.1, bad=0.3)
    assert t([0])[0] == 1
    assert t([0.1])[0] == 1
    assert np.isclose(t([0.3])[0], 0.1)
    assert t([0.5])[0] < 0.01
    assert math.isnan(t([NAN])[0])

    t = transforms.SoftPlus(good=1, bad=0.5)
    assert t([1.5])[0] == 1
    assert t([1])[0] == 1
    assert np.isclose(t([0.5])[0], 0.1)
    assert t([0.1])[0] < 0.01


def test_sigmoid():
    t = transforms.Sigmoid(good=0.1, bad=0.3)
    assert t([0])[0] > 0.99
    assert np.isclose(t([0.1])[0], 0.95)
    assert np.isclose(t([0.2])[0], 0.5)
    assert np.isclose(t([0.3])[0], 0.05)
    assert t([0.4])[0] < 0.01
    assert math.isnan(t([NAN])[0])

    t = transforms.Sigmoid(good=1, bad=0.5)
    assert t([1.5])[0] > 0.99
    assert np.isclose(t([1])[0], 0.95)
    assert np.isclose(t([0.75])[0], 0.5)
    assert np.isclose(t([0.5])[0], 0.05)
    assert t([0.25])[0] < 0.01


def test_equal():
    t = transforms.Equal()
    assert t([0])[0] == 1
    assert t([0.001])[0] == 0
    assert math.isnan(t([NAN])[0])

    t = transforms.Equal(not_equal_val=0.5)
    assert t([0])[0] == 1
    assert t([0.001])[0] == 0.5

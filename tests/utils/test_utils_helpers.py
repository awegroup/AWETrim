"""Unit tests for awetrim.utils.utils.

Covers the CasADi vector helpers (evaluated numerically). The EKF-file reader
that used to live here is awetrim.flight_data (tests/flight_data/).
"""

import casadi as ca
import numpy as np
import pytest

from awetrim.utils.utils import (
    calculate_angle_2vec,
    skew_symmetric,
)


def _np(expr):
    return np.asarray(ca.DM(expr)).squeeze()


def test_skew_symmetric_matrix_layout():
    S = _np(skew_symmetric(ca.DM([1.0, 2.0, 3.0])))
    expected = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    assert S == pytest.approx(expected)


def test_skew_symmetric_is_antisymmetric():
    S = _np(skew_symmetric(ca.DM([0.5, -1.5, 2.0])))
    assert S == pytest.approx(-S.T)


def test_skew_symmetric_times_self_is_zero():
    v = ca.DM([1.0, 2.0, 3.0])
    assert _np(skew_symmetric(v) @ v) == pytest.approx(np.zeros(3))


def test_angle_between_orthogonal_vectors_is_half_pi():
    angle = float(ca.DM(calculate_angle_2vec(ca.DM([1.0, 0.0, 0.0]), ca.DM([0.0, 1.0, 0.0]))))
    assert angle == pytest.approx(np.pi / 2)


def test_angle_between_parallel_vectors_is_zero():
    angle = float(ca.DM(calculate_angle_2vec(ca.DM([2.0, 0.0, 0.0]), ca.DM([5.0, 0.0, 0.0]))))
    assert angle == pytest.approx(0.0, abs=1e-7)


def test_angle_is_scale_invariant():
    a, b = ca.DM([1.0, 1.0, 0.0]), ca.DM([0.0, 1.0, 0.0])
    angle = float(ca.DM(calculate_angle_2vec(a, b)))
    angle_scaled = float(ca.DM(calculate_angle_2vec(10.0 * a, 0.1 * b)))
    assert angle == pytest.approx(angle_scaled)
    assert angle == pytest.approx(np.pi / 4)


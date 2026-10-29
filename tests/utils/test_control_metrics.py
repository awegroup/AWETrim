"""Unit tests for :mod:`awetrim.utils.control_metrics`."""

import numpy as np

from awetrim.utils.control_metrics import (
    STEERING_DEADBAND,
    count_steering_reversals,
)


def test_count_steering_reversals_ignores_deadband_chatter():
    u = np.array([0.1, 0.01, -0.01, 0.1, -0.1, 0.005, -0.1])
    # inside the deadband the sign is dropped: +, +, -, - -> one reversal
    assert count_steering_reversals(u, deadband=0.02, periodic=False) == 1
    # with no deadband every sign change counts
    assert count_steering_reversals(u, deadband=0.0, periodic=False) == 5


def test_count_steering_reversals_counts_the_seam_only_when_periodic():
    u = np.array([0.1, 0.1, -0.1, -0.1])
    assert count_steering_reversals(u, periodic=False) == 1
    assert count_steering_reversals(u, periodic=True) == 2
    same_sign_ends = np.array([0.1, -0.1, 0.1])
    assert count_steering_reversals(same_sign_ends, periodic=True) == 2


def test_count_steering_reversals_all_centred_is_zero():
    assert count_steering_reversals(np.zeros(10)) == 0
    assert count_steering_reversals(np.full(10, 0.5 * STEERING_DEADBAND)) == 0
    assert count_steering_reversals([]) == 0
    assert count_steering_reversals(np.full(10, 0.1)) == 0

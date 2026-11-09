"""awetrim.timeseries.phase: the step-box and mean-power helpers shared by the reel-out and cycle scripts."""

import numpy as np
import pytest

from awetrim.timeseries.phase import step_box_active, trajectory_mean_power


def test_box_active_at_its_edge_only():
    before = {"C_phi": np.zeros(3)}
    assert step_box_active(before, {"C_phi": np.array([0.0, 0.295, 0.0])}, {"C_phi": 0.3})
    assert not step_box_active(before, {"C_phi": np.array([0.0, 0.2, 0.0])}, {"C_phi": 0.3})


def test_box_frozen_elements_are_ignored():
    before = {"C_phi": np.zeros(2)}
    after = {"C_phi": np.array([0.5, 0.0])}
    assert not step_box_active(before, after, {"C_phi": [0.0, 0.3]})


def test_mean_power():
    traj = {"s": [0.0, 1.0, 2.0], "s_dot": [1.0, 1.0, 1.0],
            "tension_tether_ground": [1000.0, 1000.0, 1000.0], "speed_radial": [1.0, 2.0, 3.0]}
    assert trajectory_mean_power(traj) == pytest.approx(2000.0)
    assert trajectory_mean_power({"s": [0.0]}) is None

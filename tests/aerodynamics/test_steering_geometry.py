# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the geometric bridle-steering model (roll from delta).

Pure geometry + signature assertions; no VSM solve involved.
"""

import inspect

import numpy as np
import pytest

from awetrim.aerodynamics.vsm_quasi_steady import (
    roll_angle_from_steering_delta,
    solve_vsm_quasi_steady_trim,
    steering_delta_limit,
    turn_radius_vs_steering_delta,
)

H, B = 8.54, 8.25  # LEI-V3 bridle triangle (tip mid-chords vs bridle point)


def test_zero_delta_gives_zero_roll():
    assert roll_angle_from_steering_delta(H, B, 0.0) == 0.0


def test_sign_convention_positive_delta_negative_roll():
    # delta = (L_left - L_right)/2 > 0 (left line longer) -> negative theta.
    assert roll_angle_from_steering_delta(H, B, 0.3) < 0.0
    assert roll_angle_from_steering_delta(H, B, -0.3) > 0.0
    # Odd function.
    assert roll_angle_from_steering_delta(H, B, 0.3) == pytest.approx(
        -roll_angle_from_steering_delta(H, B, -0.3)
    )


def test_small_delta_gain_matches_analytic():
    c = H**2 + (B / 2.0) ** 2
    gain = 2.0 * np.sqrt(c) / (H * B)  # rad/m
    d = 1e-4
    theta = np.deg2rad(roll_angle_from_steering_delta(H, B, d))
    assert abs(theta) / d == pytest.approx(gain, rel=1e-4)


def test_v3_scale_roll_angles():
    # ~15.4 deg per metre for the V3 triangle; delta=0.5 m -> ~7.7 deg.
    assert roll_angle_from_steering_delta(H, B, 0.5) == pytest.approx(-7.73, abs=0.05)


def test_delta_is_clamped_inside_geometric_limit():
    lim = steering_delta_limit(H, B)
    assert lim > 0.0
    # Far beyond the limit: no NaN, angle saturates near the max tilt.
    theta_beyond = roll_angle_from_steering_delta(H, B, 10.0 * lim)
    assert np.isfinite(theta_beyond)
    assert abs(theta_beyond) <= 90.0


def test_trim_accepts_prescribed_roll():
    sig = inspect.signature(solve_vsm_quasi_steady_trim)
    assert "prescribed_roll_deg" in sig.parameters
    assert sig.parameters["prescribed_roll_deg"].default is None


def test_turn_map_signature_requires_triangle_geometry():
    sig = inspect.signature(turn_radius_vs_steering_delta)
    for name in ("steering_h", "steering_b", "tip_midpoint"):
        assert name in sig.parameters
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

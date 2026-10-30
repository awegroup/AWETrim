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

"""Lift and drag of a ROM kite at a given power tape and apparent wind.

The point evaluation behind the server's ``POST /aero_forces``: a client
(e.g. another kite simulator) compares its own forces with the ROM's at the
same power-tape length ``u_p`` and apparent wind speed ``v_a``.

Without an angle of attack the ROM's own GRAVITY-FREE TRIM sets it: with no
weight and no inertia the bridle resultant balances the wing force, so the
bridle pitch is ``alpha_b = atan2(C_D, C_L)`` and the wing angle of attack
solves ``alpha = atan2(C_D(alpha), C_L(alpha)) - theta_b(u_p)`` (paper Eq. 1).
This is the centre-window state the LEI-V3 ROMs were identified on; the KCU
drag acts at the bridle point and does not enter it. The lowest root below
the stall is taken, as in the quasi-steady solver.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import casadi as ca
import numpy as np
from scipy.optimize import brentq

from awetrim.aerodynamics.kcu_drag import force_drag_kcu

#: Bracket of the trim's angle-of-attack scan [deg] and its step.
ALPHA_SCAN_MIN_DEG = -10.0
ALPHA_SCAN_STEP_DEG = 0.25
ALPHA_SCAN_MAX_DEG_NO_STALL = 30.0


def _coefficient_function(kite) -> ca.Function:
    """``(alpha, u_p, u_s, v_a) -> (C_L, C_D)`` of the kite's ROM."""
    alpha, u_p, u_s, v_a = (ca.MX.sym(n) for n in ("alpha", "u_p", "u_s", "v_a"))
    c_l, c_d = kite.aerodynamic_force_coefficients_at(alpha, u_p, u_s, v_a)[:2]
    return ca.Function("rom_coefficients", [alpha, u_p, u_s, v_a], [c_l, c_d])


def _angle_pitch_tether(kite, input_depower: float) -> float:
    """Numeric theta_b(u_p) [rad]; the ROM must state both pitch params."""
    theta_0 = kite.angle_pitch_tether_0
    slope = kite.slope_angle_pitch_tether_depower
    if isinstance(theta_0, ca.MX) or isinstance(slope, ca.MX):
        raise ValueError(
            "the ROM states no angle_pitch_tether_0 / "
            "slope_angle_pitch_tether_depower: no trim angle of attack"
        )
    return float(theta_0) + float(slope) * float(input_depower)


def _trim_angle_of_attack(coefficients, theta_b, u_p, u_s, v_a, alpha_max) -> float:
    """Lowest root of ``atan2(C_D, C_L) - theta_b - alpha`` below ``alpha_max``."""

    def residual(alpha):
        c_l, c_d = (float(c) for c in coefficients(alpha, u_p, u_s, v_a))
        return math.atan2(c_d, c_l) - theta_b - alpha

    grid = np.arange(
        math.radians(ALPHA_SCAN_MIN_DEG),
        alpha_max + 1e-12,
        math.radians(ALPHA_SCAN_STEP_DEG),
    )
    values = [residual(a) for a in grid]
    for a0, a1, r0, r1 in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if r0 == 0.0:
            return float(a0)
        if r0 * r1 < 0.0:
            return float(brentq(residual, a0, a1, xtol=1e-10))
    raise ValueError(
        f"no gravity-free trim between {ALPHA_SCAN_MIN_DEG} deg and "
        f"{math.degrees(alpha_max):.2f} deg at u_p = {u_p} m"
    )


def _kcu_force(kite, speed_apparent_wind, force_wing_dir, rho) -> np.ndarray:
    """KCU drag in the (drag, -, lift) frame, its axis along the tether.

    In a gravity-free trim the tether carries wing + KCU force, so the axis
    depends on the KCU force itself; a few fixed-point passes settle it.
    """
    model = kite.kcu_drag_model
    if model is None:
        return np.zeros(3)
    velocity = np.array([speed_apparent_wind, 0.0, 0.0])
    force = np.zeros(3)
    for _ in range(5):
        axis = force_wing_dir + force
        force = force_drag_kcu(
            velocity, axis, rho, model.cd_area_axial, model.cd_area_broadside
        )
    return force


def rom_aero_forces(
    kite,
    input_depower: float,
    speed_apparent_wind: float,
    angle_of_attack: Optional[float] = None,
    input_steering: float = 0.0,
    rho: Optional[float] = None,
) -> Dict[str, Any]:
    """Lift and drag [N] of ``kite`` (a :class:`~awetrim.system.kite.Kite`).

    ``input_depower`` is u_p (power-tape length [m] on the LEI-V3),
    ``speed_apparent_wind`` v_a [m/s], ``angle_of_attack`` the wing angle of
    attack [rad] (``None`` = the gravity-free trim, see the module docstring),
    ``input_steering`` u_s, which only enters through the coefficients here,
    and ``rho`` the air density (``None`` = the kite's). Lift is perpendicular
    to v_a, drag along it. ``lift``/``drag`` are the wing's (bridle lines
    included); ``drag_kcu``/``lift_kcu`` are the KCU's when the ROM keeps it
    out of C_D (else zero); ``*_total`` are their sums.
    """
    u_p = float(input_depower)
    u_s = float(input_steering)
    v_a = float(speed_apparent_wind)
    if v_a <= 0.0:
        raise ValueError("speed_apparent_wind must be positive")
    rho = float(kite.rho if rho is None else rho)

    coefficients = _coefficient_function(kite)
    trimmed = angle_of_attack is None
    theta_b = None
    if trimmed:
        theta_b = _angle_pitch_tether(kite, u_p)
        alpha_max = float(
            kite.aero_params.get(
                "angle_of_attack_stall", math.radians(ALPHA_SCAN_MAX_DEG_NO_STALL)
            )
        )
        alpha = _trim_angle_of_attack(coefficients, theta_b, u_p, u_s, v_a, alpha_max)
    else:
        alpha = float(angle_of_attack)
        try:
            theta_b = _angle_pitch_tether(kite, u_p)
        except ValueError:
            theta_b = None

    c_l, c_d = (float(c) for c in coefficients(alpha, u_p, u_s, v_a))
    dynamic_pressure_area = 0.5 * rho * v_a**2 * kite.area_wing
    lift = dynamic_pressure_area * c_l
    drag = dynamic_pressure_area * c_d
    force_kcu = _kcu_force(kite, v_a, np.array([drag, 0.0, lift]), rho)

    return {
        "angle_of_attack": alpha,
        "angle_pitch_tether": theta_b,
        "angle_pitch_bridle": None if theta_b is None else alpha + theta_b,
        "trimmed": trimmed,
        "lift_coefficient": c_l,
        "drag_coefficient": c_d,
        "lift": lift,
        "drag": drag,
        "lift_kcu": float(force_kcu[2]),
        "drag_kcu": float(force_kcu[0]),
        "lift_total": lift + float(force_kcu[2]),
        "drag_total": drag + float(force_kcu[0]),
        "reference_area": float(kite.area_wing),
        "rho": rho,
        "kcu_drag_in_coefficients": bool(kite.kcu_drag_in_coefficients),
    }

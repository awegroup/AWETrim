"""Tests for force and acceleration computations on Kite.

Covers properties tested here that are not in test_kite.py:
  - acceleration_rotation_course  (Coriolis/centripetal cross-product)
"""

import casadi as ca
import numpy as np

from awetrim.system.kite import Kite
from awetrim.system.system_model import SystemModel
from awetrim.environment.Wind import Wind


_AERO_INPUT = {"model": "coeffs", "params": {"CD0": 0.05}, "coefficients": {}}

_INVISCID_INPUT = {
    "model": "inviscid",
    "params": {
        "oswald_efficiency": 0.9,
        "aspect_ratio": 8.0,
        "CD0": 0.05,
        "angle_pitch_tether_0": 0.0,
        "slope_angle_pitch_tether_depower": 0.0,
    },
}


def _make_kite() -> Kite:
    return Kite(mass_wing=10.0, area_wing=5.0, aero_input=_AERO_INPUT)


def test_inviscid_side_force_uses_unrotated_lift():
    """For the inviscid model C_L = C_L0·cos(roll) and C_S = C_L0·sin(roll),
    so C_L·sin(roll) - C_S·cos(roll) == 0 for any state. (Regression: C_S was
    built from the already-rotated C_L, giving C_L0·cos·sin.)"""
    kite = Kite(
        mass_wing=10.0,
        area_wing=5.0,
        aero_input=_INVISCID_INPUT,
        steering_control="roll",
    )
    wind = Wind("uniform", direction_wind=0.0, speed_wind_ref=10.0)
    model = SystemModel(kite=kite, quasi_steady=True, wind_model=wind)

    C_L, _C_D, C_S = model.kite.aerodynamic_force_coefficients_for(model)
    roll = model.input_steering * model.kite.gain_roll_steering
    residual = C_L * ca.sin(roll) - C_S * ca.cos(roll)

    syms = ca.symvar(residual)
    resid_fn = ca.Function("resid", syms, [residual])
    rng = np.random.default_rng(0)
    values = [float(rng.uniform(0.5, 1.5)) for _ in syms]
    assert abs(float(resid_fn(*values))) < 1e-9


def test_acceleration_rotation_course_equals_cross_product():
    """acceleration_rotation_course = cross(velocity_rotation_course_frame, velocity_kite).

    When velocity_rotation_course_frame and velocity_kite are set to constant
    vectors, the property must match the cross product of those same vectors.
    """
    kite = _make_kite()

    omega = ca.vertcat(0.1, 0.2, 0.3)
    vk = ca.vertcat(1.0, 0.5, 0.2)

    kite.velocity_rotation_course_frame = omega
    kite.velocity_kite = vk

    expected = ca.cross(omega, vk)
    residual = ca.norm_2(kite.acceleration_rotation_course - expected)

    f = ca.Function("residual", [], [residual])
    result = f()
    value = list(result.values())[0] if isinstance(result, dict) else result[0]
    assert float(value) < 1e-10

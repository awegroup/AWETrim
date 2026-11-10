"""ROM steering_law turn_rate: the yaw-moment turn law replaces the lateral row,
the roll is solved from the lateral force balance."""

import casadi as ca
import numpy as np
import pytest

from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.system_model import SystemModel

TURN_PARAMS = {"gain_turn_steering": 0.4, "turn_inertia_coefficient": 0.02}


def _model(steering_law="turn_rate", **params) -> SystemModel:
    aero = {"model": "coeffs", "steering_law": steering_law,
            "params": {"CD0": 0.08, "CL0": 0.9, "angle_pitch_tether_0": -0.3,
                       "slope_angle_pitch_tether_depower": 0.2,
                       "gain_roll_steering": -0.4, **TURN_PARAMS, **params},
            "coefficients": {}}
    kite = Kite(mass_wing=13.0, mass_kcu=8.0, area_wing=19.75, aero_input=aero,
                steering_control="asymmetric")
    return SystemModel(kite=kite, quasi_steady=True,
                       wind_model=Wind("uniform", direction_wind=0.0, speed_wind_ref=8.0))


def _numeric(model, exprs, seed=1):
    """Evaluate ``exprs`` at one random, physical-ish point of their symbols."""
    syms = ca.symvar(ca.vertcat(*exprs))
    f = ca.Function("f", syms, list(exprs))
    rng = np.random.default_rng(seed)
    ranges = {"distance_radial": (200, 300), "angle_elevation": (0.3, 0.6),
              "angle_azimuth": (-0.3, 0.3), "angle_course": (0.5, 2.5),
              "speed_tangential": (18, 26), "speed_radial": (0.5, 2.0),
              "timeder_angle_course": (-0.5, 0.5), "tension_tether_ground": (3000, 5000),
              "input_depower": (1.6, 2.2), "input_steering": (-0.1, 0.1)}
    values = [rng.uniform(*ranges.get(s.name(), (0.0, 0.1))) for s in syms]
    outs = f(*values)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    return [np.asarray(v, dtype=float).ravel() for v in outs], dict(zip(
        [s.name() for s in syms], values))


def test_unknown_steering_law_is_rejected():
    with pytest.raises(ValueError, match="steering_law"):
        _model("roll_rate")


def test_turn_rate_needs_its_params():
    with pytest.raises(ValueError, match="gain_turn_steering"):
        aero = {"model": "coeffs", "steering_law": "turn_rate",
                "params": {"CD0": 0.08}, "coefficients": {}}
        Kite(mass_wing=1.0, area_wing=1.0, aero_input=aero, steering_control="asymmetric")


def test_roll_gain_residual_is_the_force_balance():
    model = _model("roll_gain")
    residual = model.force_residual
    full = -(model.kite.mass_wing + model.kite.mass_kcu) * model.acceleration + model.force_external
    (diff,), _ = _numeric(model, [residual - full])
    assert np.allclose(diff, 0.0)


def test_roll_is_an_unknown_and_the_residual_gains_the_turn_law():
    """[chi, n, r] force rows, then the turn law; angle_roll a default unknown
    that the lift direction (hence the force rows) depends on."""
    model = _model()
    mass = model.kite.mass_wing + model.kite.mass_kcu
    full = -mass * model.acceleration + model.force_external
    assert model.force_residual.numel() == 4
    assert "angle_roll" in model.default_unknown_vars
    assert ca.depends_on(full, model.angle_roll)
    (residual, force), _ = _numeric(model, [model.force_residual, full])
    assert np.allclose(residual[:3], force)


def test_roll_gain_model_has_no_roll_unknown():
    model = _model("roll_gain")
    assert "angle_roll" not in model.default_unknown_vars
    assert model.force_residual.numel() == 3
    with pytest.raises(AttributeError):
        model.angle_roll


def test_turn_law_row():
    """Row 3 is m (w v_a + b (m v_tau w + F_g,n) - a v_a^2 u_s), w = -a_n / v_tau."""
    model = _model()
    kite = model.kite
    mass = kite.mass_wing + kite.mass_kcu
    va = ca.norm_2(model.velocity_apparent_wind)
    w = -model.acceleration[1] / (model.speed_tangential + 1e-6)
    gravity_n = model.expression("force_gravity")[1]
    expected = mass * (w * va + 0.02 * (mass * model.speed_tangential * w + gravity_n)
                       - 0.4 * va**2 * model.input_steering)
    (row, ref), _ = _numeric(model, [model.force_residual[3], expected])
    assert row[0] == pytest.approx(ref[0], rel=1e-6)


def test_turn_rate_is_course_rate_less_transport():
    """w = chi_dot - (v_tau / r) tan(beta) sin(chi): the turn about the tether."""
    model = _model()
    (w,), x = _numeric(model, [model.kite.turn_rate_for(model)])
    transport = (x["speed_tangential"] / x["distance_radial"]
                 * np.tan(x["angle_elevation"] + 1e-6) * np.sin(x["angle_course"]))
    expected = x["timeder_angle_course"] - transport
    assert w[0] == pytest.approx(expected * x["speed_tangential"] / (x["speed_tangential"] + 1e-6),
                                 rel=1e-6)


def test_complete_law_gravity_terms_match_eq_33():
    """The k5 (cos chi cos beta) and k6 (sin beta) terms of Eq. 33, written with
    the ROM's weight vector F_g = -m g [cos chi cos beta, sin chi cos beta,
    sin beta]: w (v_a + b m v_tau) = a v_a^2 u_s + m g (b s_chi c_beta +
    c5 c_chi c_beta + c6 s_beta)."""
    model = _model(turn_gravity_course_coefficient=0.003, turn_gravity_radial_coefficient=-0.001)
    kite = model.kite
    mass = kite.mass_wing + kite.mass_kcu
    va = ca.norm_2(model.velocity_apparent_wind)
    w = kite.turn_rate_for(model)
    beta, chi, g = model.angle_elevation, model.angle_course, 9.81
    rhs = (0.4 * va**2 * model.input_steering
           + mass * g * (0.02 * ca.sin(chi) * ca.cos(beta) + 0.003 * ca.cos(chi) * ca.cos(beta)
                         - 0.001 * ca.sin(beta)))
    expected = mass * (w * (va + 0.02 * mass * model.speed_tangential) - rhs)
    (row, ref), _ = _numeric(model, [model.force_residual[3], expected])
    assert row[0] == pytest.approx(ref[0], rel=1e-5)


def test_inertia_coefficient_depower_and_density():
    """b(u_p) = b0 + b1 u_p ..., and b, c5, c6 scale with rho_ref / rho (k2 ~ rho)."""
    model = _model(turn_inertia_coefficient_depower=[0.005], turn_density_reference=1.0)
    kite = model.kite
    b = kite.turn_inertia_coefficient_for(model)
    assert ca.depends_on(b, model.input_depower)
    (value,), x = _numeric(model, [b])
    assert value[0] == pytest.approx((0.02 + 0.005 * x["input_depower"]) * 1.0 / kite.rho)


def test_turn_gain_depends_on_depower_and_steering_squared():
    model = _model(gain_turn_steering_depower=[0.1], gain_turn_steering_squared=[-1.0])
    gain = model.kite.gain_turn_steering_for(model)
    assert ca.depends_on(gain, model.input_depower)
    assert ca.depends_on(gain, model.input_steering)


def test_steering_authority_scale_acts_on_the_turn_gain():
    model = _model(gain_turn_steering_depower=[0.1], gain_turn_steering_squared=[-1.0])
    base = model.kite.gain_turn_steering_for(model)
    model.kite.gain_roll_scale = 1.5
    scaled = model.kite.gain_turn_steering_for(model)
    (b, s), _ = _numeric(model, [base, scaled])
    assert s[0] == pytest.approx(1.5 * b[0])

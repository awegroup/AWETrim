"""ROM parameter naming on Kite: paper-sign pitch, roll gain, reference area, v_a terms."""

import casadi as ca
import pytest

from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite, canonical_aero_params
from awetrim.system.system_model import SystemModel


def _kite(params, coefficients=None, **kwargs) -> Kite:
    aero_input = {
        "model": "coeffs",
        "params": {"CD0": 0.05, **params},
        "coefficients": coefficients or {},
        **kwargs,
    }
    return Kite(
        mass_wing=10.0, area_wing=5.0, aero_input=aero_input,
        steering_control="asymmetric",
    )


def _model(kite: Kite) -> SystemModel:
    wind = Wind("uniform", direction_wind=0.0, speed_wind_ref=10.0)
    return SystemModel(kite=kite, quasi_steady=True, wind_model=wind)


def test_legacy_pitch_keys_map_to_paper_sign():
    with pytest.warns(DeprecationWarning):
        params = canonical_aero_params(
            {"angle_pitch_depower_0": 0.4, "delta_pitch_depower": -0.3}
        )
    assert params == {
        "angle_pitch_tether_0": -0.4,
        "slope_angle_pitch_tether_depower": 0.3,
    }


def test_both_spellings_of_one_parameter_are_rejected():
    with pytest.raises(ValueError):
        canonical_aero_params({"angle_pitch_depower_0": 0.1, "angle_pitch_tether_0": -0.1})


def _evaluate(expr, seed=0):
    """Evaluate an MX expression at random values of all its free symbols."""
    import numpy as np

    syms = ca.symvar(expr)
    if not syms:
        return float(ca.evalf(expr))
    f = ca.Function("f", syms, [expr])
    rng = np.random.default_rng(seed)
    return float(f(*[float(rng.uniform(0.5, 1.5)) for _ in syms]))


def test_angle_of_attack_subtracts_theta_b():
    """alpha_w = alpha_b - theta_b(u_p): the u_p sensitivity is -slope."""
    model = _model(_kite({"angle_pitch_tether_0": 0.1,
                          "slope_angle_pitch_tether_depower": 0.3}))
    alpha = model.kite.angle_of_attack_for(model)
    assert not ca.depends_on(model.kite.pitch_bridle_for(model), model.input_depower)
    assert _evaluate(ca.jacobian(alpha, model.input_depower)) == pytest.approx(-0.3)


def test_bridle_angles_are_taken_in_the_apparent_wind_frame():
    """pitch_bridle_for is paper Eq. D11 and roll_bridle_for Eqs. D6-D8 about
    v_a itself, also in sideslip (v_a with a normal component). Until
    2026-10-06 an Euler-angle frame with a mirrored yaw sign sat twice the
    sideslip off v_a and put alpha_b 2-3 deg off in turns."""
    import numpy as np

    # A constant, positive lift: the lift direction (hence its roll) is
    # defined for L > 0 only.
    # A KCU mass gives the bridle resultant a weight and inertia part, so it
    # is not purely radial (the bare-wing default has F_b = 0, which would
    # make this test blind).
    kite = Kite(mass_wing=10.0, mass_kcu=5.0, area_wing=5.0, steering_control="asymmetric",
                aero_input={"model": "coeffs",
                            "params": {"CD0": 0.05, "CL0": 1.0, "angle_pitch_tether_0": 0.1,
                                       "slope_angle_pitch_tether_depower": 0.3,
                                       "gain_roll_steering": -0.6},
                            "coefficients": {}})
    model = _model(kite)
    va = model.velocity_apparent_wind
    f_b = kite.force_bridle_for(model)
    f_w = kite.force_aerodynamic_wing(model)
    stacked = ca.vertcat(va, f_b, f_w, kite.pitch_bridle_for(model),
                         kite.roll_bridle_for(model), kite.angle_roll_aerodynamic_for(model))
    syms = ca.symvar(stacked)
    f = ca.Function("f", syms, [stacked])
    rng = np.random.default_rng(1)
    for _ in range(5):
        out = np.asarray(f(*[float(rng.uniform(0.5, 1.5)) for _ in syms])).ravel()
        v, fb, fw = out[0:3], out[3:6], out[6:9]
        pitch, roll, roll_steer = out[9], out[10], out[11]
        e_va = v / np.linalg.norm(v)
        assert abs(e_va[1]) > 0.05  # a genuine sideslip, or the test is blind
        e_n = np.cross([0.0, 0.0, 1.0], v)
        e_n /= np.linalg.norm(e_n)
        e_r = np.array([0.0, 0.0, 1.0]) - e_va[2] * e_va
        e_r /= np.linalg.norm(e_r)
        # Eq. D11: alpha_b = atan2(F_b . e_chi', -F_b . e_r'), e_chi' = -v_a/|v_a|.
        assert pitch == pytest.approx(np.arctan2(fb @ -e_va, -(fb @ e_r)), abs=1e-6)
        # Eqs. D6-D8: roll of the resultant about v_a, zero along e_r'.
        roll_of = lambda force: np.arctan2(-(force @ e_n), force @ e_r)  # noqa: E731
        assert roll == pytest.approx(roll_of(-fb), abs=1e-6)
        # The wing's lift (its force less the drag along v_a) is rolled by
        # k u_s from the bridle resultant.
        lift = fw - (fw @ e_va) * e_va
        assert roll_of(lift) == pytest.approx(roll + roll_steer, abs=1e-6)


def test_legacy_and_new_files_build_the_same_angle_of_attack():
    with pytest.warns(DeprecationWarning):
        legacy = _model(_kite({"angle_pitch_depower_0": 0.4,
                               "delta_pitch_depower": -0.3}))
    new = _model(_kite({"angle_pitch_tether_0": -0.4,
                        "slope_angle_pitch_tether_depower": 0.3}))
    for model in (legacy, new):
        theta_b = ca.Function("t", [model.input_depower],
                              [model.kite.angle_pitch_tether_for(model)])
        assert float(theta_b(1.7)) == pytest.approx(-0.4 + 0.3 * 1.7)
        alpha = model.kite.angle_of_attack_for(model)
        offset = alpha - model.kite.pitch_bridle_for(model)
        assert _evaluate(ca.substitute(offset, model.input_depower, ca.MX(1.7))) == (
            pytest.approx(0.4 - 0.3 * 1.7)
        )


def test_explicit_roll_gain_wins_over_side_force_coefficient():
    coefficients = {"CS": [{"var": "u_s", "coef": 0.34}]}
    assert _kite({}, coefficients).gain_roll_steering == pytest.approx(-0.34)
    kite = _kite({"gain_roll_steering": 0.15}, coefficients)
    assert kite.gain_roll_steering == pytest.approx(0.15)
    assert kite.k_steering == kite.gain_roll_steering


def test_reference_area_overrides_the_system_area():
    assert _kite({}).area_wing == pytest.approx(5.0)
    assert _kite({}, reference_area=17.2119).area_wing == pytest.approx(17.2119)


def test_v_a_terms_enter_the_lift_coefficient():
    model = _model(_kite({}, {"CL": [{"var": "v_a", "coef": 0.01}]}))
    cl = model.kite.aerodynamic_force_coefficients_for(model)[0]
    assert ca.depends_on(cl, model.speed_tangential)


def test_without_v_a_terms_the_coefficient_ignores_the_speed():
    model = _model(_kite({}, {"CL": [{"var": "u_p", "coef": 0.1}]}))
    cl = model.kite.aerodynamic_force_coefficients_for(model)[0]
    assert not ca.depends_on(cl, model.speed_tangential)
    assert ca.depends_on(cl, model.input_depower)


def _kite_with_kcu(**aero_kwargs) -> Kite:
    aero_input = {"model": "coeffs", "params": {"CD0": 0.05}, "coefficients": {},
                  **aero_kwargs}
    return Kite(mass_wing=10.0, area_wing=5.0, aero_input=aero_input,
                length_kcu=1.0, diameter_kcu=0.5642, diameter_turbine=0.5,
                steering_control="asymmetric")


def test_kcu_drag_stays_inside_the_coefficients_by_default():
    model = _model(_kite_with_kcu())
    assert model.kite.kcu_drag_model is None
    assert _evaluate(ca.norm_2(model.kite.force_drag_kcu_for(model))) == 0.0


def test_explicit_kcu_drag_joins_the_aero_force_and_the_bridle_resultant():
    model = _model(_kite_with_kcu(kcu_drag_in_coefficients=False))
    kite = model.kite
    assert kite.kcu_drag_model.cd_area_broadside == pytest.approx(0.5458, abs=1e-3)
    drag = kite.force_drag_kcu_for(model)
    gap = kite.force_aerodynamic(model) - kite.force_aerodynamic_wing(model) - drag
    assert _evaluate(ca.norm_2(gap)) == pytest.approx(0.0, abs=1e-6)
    assert _evaluate(ca.norm_2(drag)) > 0.0
    # The KCU hangs at the bridle point: its drag loads the bridle resultant.
    bridle_gap = (kite.force_bridle_for(model) - model.force_tether_at_kite
                  - model.force_gravity_kcu + kite.mass_kcu * model.acceleration - drag)
    assert _evaluate(ca.norm_2(bridle_gap)) == pytest.approx(0.0, abs=1e-6)


def test_stall_terms_switch_on_past_the_stall_angle():
    import numpy as np
    from awetrim.system.kite import stall_blend

    a_s, w = np.deg2rad(17.0), np.deg2rad(1.0)
    sigma = stall_blend(np.deg2rad([5.0, 17.0, 30.0]), a_s, w, xp=np)
    assert sigma[0] < 1e-4 and sigma[1] == pytest.approx(0.5) and sigma[2] > 1 - 1e-4
    model = _model(_kite({"angle_of_attack_stall": a_s, "width_stall": w},
                         {"CL": [{"vars": {"stall": 1}, "coef": -0.3}]}))
    cl = model.kite.aerodynamic_force_coefficients_for(model)[0]
    assert ca.depends_on(cl, model.speed_tangential)  # through alpha


def test_stall_terms_without_their_parameters_are_rejected():
    model = _model(_kite({}, {"CL": [{"vars": {"stall": 1}, "coef": -0.3}]}))
    with pytest.raises(ValueError):
        model.kite.aerodynamic_force_coefficients_for(model)


def test_drag_terms_are_plain_unless_marked_abs():
    """An odd C_D term keeps its sign (no kink at zero); abs: true folds it."""
    for u_s, plain, folded in ((-0.5, 0.1, -0.1), (0.5, 0.1, 0.1)):
        for entry, expected in (({"var": "u_s", "coef": 0.1}, plain),
                                ({"var": "u_s", "coef": 0.1, "abs": True}, folded)):
            model = _model(_kite({}, {"CD": [entry]}))
            cd = model.kite.aerodynamic_force_coefficients_for(model)[1]
            jac = ca.jacobian(cd, model.input_steering)
            value = ca.evalf(ca.substitute(jac, model.input_steering, ca.MX(u_s)))
            assert float(value) == pytest.approx(expected, abs=1e-6)

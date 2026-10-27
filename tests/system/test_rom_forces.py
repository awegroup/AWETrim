"""ROM lift and drag at one state: explicit alpha and the gravity-free trim."""

import math

import pytest

from awetrim.system.kite import Kite
from awetrim.system.rom_forces import rom_aero_forces

RHO = 1.225
AREA = 5.0


def _kite(kcu=False) -> Kite:
    aero_input = {
        "model": "coeffs",
        "reference_area": AREA,
        "kcu_drag_in_coefficients": not kcu,
        "params": {
            "CL0": 0.3,
            "CD0": 0.05,
            "angle_pitch_tether_0": -0.2,
            "slope_angle_pitch_tether_depower": 0.1,
        },
        "coefficients": {
            "CL": [{"var": "alpha", "power": 1, "coef": 4.0}],
            "CD": [
                {"var": "alpha", "power": 2, "coef": 1.0},
                {"var": "v_a", "power": 1, "coef": 0.001},
            ],
        },
    }
    kwargs = dict(length_kcu=0.6, diameter_kcu=0.3) if kcu else {}
    return Kite(mass_wing=10.0, area_wing=AREA, aero_input=aero_input, **kwargs)


def _coefficients(alpha, v_a):
    return 0.3 + 4.0 * alpha, 0.05 + alpha**2 + 0.001 * v_a


def test_explicit_angle_of_attack():
    alpha, v_a = math.radians(5.0), 12.0
    result = rom_aero_forces(_kite(), 1.5, v_a, angle_of_attack=alpha)
    c_l, c_d = _coefficients(alpha, v_a)
    q_area = 0.5 * RHO * v_a**2 * AREA
    assert not result["trimmed"]
    assert result["lift_coefficient"] == pytest.approx(c_l)
    assert result["drag_coefficient"] == pytest.approx(c_d)
    assert result["lift"] == pytest.approx(q_area * c_l)
    assert result["drag"] == pytest.approx(q_area * c_d)
    assert result["drag_kcu"] == 0.0
    assert result["drag_total"] == result["drag"]


def test_trim_balances_the_bridle_pitch():
    u_p, v_a = 1.5, 12.0
    result = rom_aero_forces(_kite(), u_p, v_a)
    alpha = result["angle_of_attack"]
    c_l, c_d = _coefficients(alpha, v_a)
    theta_b = -0.2 + 0.1 * u_p
    assert result["trimmed"]
    assert result["angle_pitch_tether"] == pytest.approx(theta_b)
    assert alpha == pytest.approx(math.atan2(c_d, c_l) - theta_b, abs=1e-9)
    assert result["angle_pitch_bridle"] == pytest.approx(math.atan2(c_d, c_l))


def test_kcu_drag_is_added_outside_the_coefficients():
    result = rom_aero_forces(_kite(kcu=True), 1.5, 12.0, angle_of_attack=0.08)
    assert result["drag_kcu"] > 0.0
    assert result["drag_total"] == pytest.approx(result["drag"] + result["drag_kcu"])
    assert result["lift_total"] == pytest.approx(result["lift"] + result["lift_kcu"])


def test_rho_override_scales_the_forces():
    base = rom_aero_forces(_kite(), 1.5, 12.0, angle_of_attack=0.05)
    thin = rom_aero_forces(_kite(), 1.5, 12.0, angle_of_attack=0.05, rho=RHO / 2)
    assert thin["lift"] == pytest.approx(base["lift"] / 2)


def test_non_positive_apparent_wind_is_rejected():
    with pytest.raises(ValueError):
        rom_aero_forces(_kite(), 1.5, 0.0)

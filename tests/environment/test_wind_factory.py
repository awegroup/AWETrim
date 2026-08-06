"""Tests for awetrim.environment.wind_factory.create_wind_model."""

import casadi as ca
import numpy as np
import pytest

from awetrim.environment.wind_factory import create_wind_model


def _eval_speed(wind, height):
    return float(ca.evalf(wind.speed_wind(height)))


def test_logarithmic_wind_matches_reference_speed_at_z_ref():
    wind = create_wind_model("logarithmic", U_ref=8.0, z_ref=100.0, z0=0.03)
    assert _eval_speed(wind, 100.0) == pytest.approx(8.0, rel=1e-12)
    # Log profile: slower below the reference height, faster above.
    assert _eval_speed(wind, 10.0) < 8.0
    assert _eval_speed(wind, 300.0) > 8.0


def test_logarithmic_wind_honours_custom_z_ref():
    wind = create_wind_model("logarithmic", U_ref=6.0, z_ref=10.0, z0=0.01)
    assert _eval_speed(wind, 10.0) == pytest.approx(6.0, rel=1e-12)


def test_uniform_wind_is_constant_with_height():
    wind = create_wind_model("uniform", U_ref=7.5)
    assert _eval_speed(wind, 10.0) == pytest.approx(7.5)
    assert _eval_speed(wind, 500.0) == pytest.approx(7.5)


def test_tabulated_wind_interpolates_node_values():
    heights = [10.0, 100.0, 300.0]
    speeds = [5.0, 8.0, 9.0]
    wind = create_wind_model("tabulated", heights=heights, speeds=speeds)
    for h, v in zip(heights, speeds):
        assert _eval_speed(wind, h) == pytest.approx(v)
    # Linear interpolation between nodes
    assert _eval_speed(wind, 55.0) == pytest.approx(np.interp(55.0, heights, speeds))


def test_direction_wind_is_numeric_not_free_symbol():
    for wind in (
        create_wind_model("logarithmic", U_ref=8.0),
        create_wind_model("uniform", U_ref=8.0),
        create_wind_model(
            "tabulated", heights=[10.0, 100.0], speeds=[5.0, 8.0]
        ),
    ):
        assert not isinstance(wind.direction_wind, (ca.MX, ca.SX))
        # The speed profile must be fully numeric (evaluable without symbols).
        assert _eval_speed(wind, 100.0) > 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        create_wind_model("logarithmic")  # missing U_ref
    with pytest.raises(ValueError):
        create_wind_model("uniform", U_ref=-1.0)
    with pytest.raises(ValueError):
        create_wind_model("tabulated")  # missing tables
    with pytest.raises(ValueError):
        create_wind_model("tabulated", heights=[10.0, 20.0], speeds=[5.0])
    with pytest.raises(ValueError):
        create_wind_model("tabulated", heights=[20.0, 10.0], speeds=[5.0, 6.0])
    with pytest.raises(ValueError):
        create_wind_model("tabulated", heights=[10.0], speeds=[5.0])
    with pytest.raises(ValueError):
        create_wind_model("gaussian", U_ref=8.0)

"""Tests for the InflowConditions profile laws (awetrim.environment.wind_profiles)."""

import math

import numpy as np
import pytest

from awetrim.environment.wind_factory import create_wind_model
from awetrim.environment.wind_profiles import (
    DEFAULT_ALPHA,
    DEFAULT_Z0,
    REFERENCE_HEIGHT,
    ProfileLaw,
    direction_wind_from_compass,
    fit_log_law,
    fit_power_law,
    fit_jet_profile,
    wind_kwargs_from_inflow,
)

BASE = {"wind_speed": 8.0, "wind_direction": 0.0}


def _speeds(kwargs, heights):
    wind = create_wind_model(**kwargs)
    return np.array([float(wind.speed_wind(h)) for h in heights])


def test_analytic_laws_match_atmospheric_models_factors():
    """Laws 0-3 reproduce AtmosphericModels.jl's calc_wind_factor."""
    z = np.array([REFERENCE_HEIGHT, 50.0, 100.0, 200.0])
    u_ref, alpha, z0 = 8.0, DEFAULT_ALPHA, DEFAULT_Z0

    const = _speeds(wind_kwargs_from_inflow({**BASE, "profile_law": 0}), z)
    assert const == pytest.approx(np.full_like(z, u_ref))

    exp_factor = (z / REFERENCE_HEIGHT) ** alpha
    got = _speeds(wind_kwargs_from_inflow({**BASE, "profile_law": 1}), z)
    assert got == pytest.approx(u_ref * exp_factor)

    log_factor = np.log(z / z0) / math.log(REFERENCE_HEIGHT / z0)
    got = _speeds(wind_kwargs_from_inflow({**BASE, "profile_law": 2}), z)
    assert got == pytest.approx(u_ref * log_factor)

    # EXPLOG with K = 1: log + K*(log - exp)
    got = _speeds(wind_kwargs_from_inflow({**BASE, "profile_law": 3}), z)
    assert got == pytest.approx(u_ref * (2.0 * log_factor - exp_factor))


def test_every_law_is_smooth_enough_for_sx_expansion():
    """Only 'tabulated' forces the NLP out of its SX expansion."""
    samples = {
        "heights": [10.0, 50.0, 100.0, 200.0, 300.0],
        "speeds": [5.5, 7.4, 8.0, 9.3, 8.6],
    }
    for law in ProfileLaw:
        kwargs = wind_kwargs_from_inflow({**BASE, "profile_law": int(law), **samples})
        assert kwargs["model_type"] != "tabulated"


def test_fitted_laws_recover_their_generating_parameters():
    z = np.array([10.0, 30.0, 60.0, 100.0, 150.0, 250.0, 400.0])

    u_log = 9.0 * np.log(z / 0.05) / math.log(REFERENCE_HEIGHT / 0.05)
    u_ref, z0 = fit_log_law(z, u_log)
    assert (u_ref, z0) == pytest.approx((9.0, 0.05))

    u_exp = 7.5 * (z / REFERENCE_HEIGHT) ** 0.14
    u_ref, alpha = fit_power_law(z, u_exp)
    assert (u_ref, alpha) == pytest.approx((7.5, 0.14))

    u_jet = u_log + 3.0 * np.exp(-((z - 180.0) ** 2) / (2.0 * 50.0**2))
    fit = fit_jet_profile(z, u_jet)
    assert fit["jet_amplitude"] == pytest.approx(3.0, abs=1e-3)
    assert fit["jet_height"] == pytest.approx(180.0, abs=1.0)
    assert fit["jet_width"] == pytest.approx(50.0, abs=1.0)
    fitted = _speeds({"model_type": "jet", "z_ref": REFERENCE_HEIGHT, **fit}, z)
    assert fitted == pytest.approx(u_jet, abs=1e-6)


def test_custom_laws_need_enough_samples():
    for law, needed in ((4, 2), (5, 2), (6, 5)):
        with pytest.raises(ValueError, match="at least"):
            wind_kwargs_from_inflow(
                {**BASE, "profile_law": law, "heights": [10.0], "speeds": [6.0]}
            )
        assert needed >= 2


def test_custom_law_rejects_unfittable_samples():
    with pytest.raises(ValueError):  # decreasing with height
        fit_log_law([10.0, 100.0], [9.0, 5.0])
    with pytest.raises(ValueError):  # log of zero speed
        fit_power_law([10.0, 100.0], [0.0, 5.0])
    with pytest.raises(ValueError):  # negative height
        fit_log_law([-10.0, 100.0], [5.0, 9.0])


def test_wind_direction_conversion():
    # wind FROM the north blows towards the south; ENU +x = East, so the
    # direction the wind blows towards is -90 deg.
    assert direction_wind_from_compass(0.0) == pytest.approx(-math.pi / 2)
    assert direction_wind_from_compass(270.0) == pytest.approx(0.0, abs=1e-12)
    # from the east -> towards -x, i.e. +/-pi (both wrap to the same heading)
    assert abs(direction_wind_from_compass(90.0)) == pytest.approx(math.pi)
    assert wind_kwargs_from_inflow(
        {"wind_speed": 8.0, "wind_direction": 270.0, "profile_law": 0}
    )["direction_wind"] == pytest.approx(0.0, abs=1e-12)


def test_optional_fields_fall_back_to_contract_defaults():
    kwargs = wind_kwargs_from_inflow({"wind_speed": 8.0, "profile_law": 3})
    assert kwargs["alpha"] == DEFAULT_ALPHA
    assert kwargs["z0"] == DEFAULT_Z0
    assert kwargs["z_ref"] == REFERENCE_HEIGHT
    # explicit None (an omitted optional field of the client struct) too
    kwargs = wind_kwargs_from_inflow(
        {"wind_speed": 8.0, "profile_law": 3, "alpha": None, "z0": None,
         "heights": None, "speeds": None, "wind_direction": None}
    )
    assert kwargs["alpha"] == DEFAULT_ALPHA
    assert kwargs["direction_wind"] == pytest.approx(-math.pi / 2)


def test_unknown_profile_law_is_rejected():
    with pytest.raises(ValueError):
        wind_kwargs_from_inflow({**BASE, "profile_law": 7})

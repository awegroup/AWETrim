"""Tests for awetrim.aerostructural.case (shared case setup)."""

from awetrim.aerostructural import case


def test_flight_case_folder_names_every_input():
    name = case.build_flight_case_folder({
        "angle_elevation_deg": 30.0, "angle_azimuth_deg": -10.0, "angle_course_deg": 90.0,
        "distance_radial": 250.0, "speed_radial": 1.5, "wind_speed_wind_ref": 8.0,
        "is_with_gravity": True, "power_tape_final_extension": 0.1,
        "steering_tape_final_extension": -0.02,
    })
    assert name == "el30_az-10_chi90_r250_vr1.5_vw8_g1_depower_p0100mm_steer_m0020mm"


def test_flight_case_folder_falls_back_to_defaults():
    assert case.build_flight_case_folder({}).startswith("el0_az0_chi90_")


def test_common_shim_reexports_case(monkeypatch):
    import importlib
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "examples" / "2_aerostructural"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.delitem(sys.modules, "common", raising=False)
    common = importlib.import_module("common")
    assert common.CONFIG_DEFAULTS is case.CONFIG_DEFAULTS
    assert common.build_system_model is case.build_system_model


def test_air_density_at_case_constant_is_rho():
    assert case.air_density_at_case({"rho": 1.1}) == 1.1
    assert case.air_density_at_case({}) == case.CONFIG_DEFAULTS["rho"]


def test_air_density_at_case_exponential_uses_kite_height():
    import math

    from awetrim.environment.profile_laws import air_density

    config = {
        "rho": 1.225,
        "air_density_law": "exponential",
        "height_gnd": 1500.0,
        "angle_elevation_deg": 30.0,
        "distance_radial": 200.0,
    }
    z = 200.0 * math.sin(math.radians(30.0))
    assert case.air_density_at_case(config) == air_density(z, 1.225, height_gnd=1500.0)


def test_air_density_at_case_rejects_an_unknown_law():
    import pytest

    with pytest.raises(ValueError, match="air_density_law"):
        case.air_density_at_case({"air_density_law": "isa"})

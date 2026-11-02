"""Tests for awetrim.aerostructural.case (shared case setup)."""

import dataclasses

import pytest

from awetrim.aerostructural import case
from awetrim.aerostructural.case import Case
from awetrim.utils.config_paths import REPO_ROOT

AS_CONFIG = REPO_ROOT / "data" / "LEI-V3-KITE" / "as_config.yaml"


def _case(**changes):
    base = Case(
        elevation_deg=30.0, azimuth_deg=-10.0, course_deg=90.0, distance_radial=250.0,
        speed_radial=1.5, timeder_speed_tangential=0.0, timeder_speed_radial=0.0,
        wind={"model": "uniform", "speed_wind_ref": 8.0},
        power_tape_extension=0.1, steering_tape_extension=-0.02,
    )
    return dataclasses.replace(base, **changes)


def test_flight_case_folder_names_every_input():
    name = case.build_flight_case_folder({
        "angle_elevation_deg": 30.0, "angle_azimuth_deg": -10.0, "angle_course_deg": 90.0,
        "distance_radial": 250.0, "speed_radial": 1.5, "wind_speed_wind_ref": 8.0,
        "is_with_gravity": True, "power_tape_final_extension": 0.1,
        "steering_tape_final_extension": -0.02,
    })
    assert name == "el30_az-10_chi90_r250_vr1.5_vw8_g1_depower_p0100mm_steer_m0020mm"


def test_case_folder_name_keeps_the_uniform_format_and_names_another_law():
    assert _case().folder_name() == "el30_az-10_chi90_r250_vr1.5_vw8_g0_depower_p0100mm_steer_m0020mm"
    log = _case(wind={"model": "logarithmic", "speed_wind_ref": 8.0})
    assert "_vw8_logarithmic_g0_" in log.folder_name()


def test_a_case_has_no_default_operating_point():
    with pytest.raises(TypeError):
        Case(elevation_deg=0.0)  # every flight-state, wind and command field is required
    with pytest.raises(ValueError, match="speed_wind_ref"):
        _case(wind={"model": "uniform"})


def test_a_case_round_trips_through_the_runtime_keys():
    c = _case(wind={"model": "logarithmic", "speed_wind_ref": 8.0, "z0": 0.1})
    assert Case.from_config(c.to_config()) == c
    assert set(c.to_config()) == case.CASE_KEYS


def test_a_stored_config_without_a_wind_spec_reads_as_uniform():
    stored = {k: v for k, v in _case().to_config().items() if k != "wind"}
    assert Case.from_config(stored).wind == {"model": "uniform", "speed_wind_ref": 8.0}


def test_a_wind_speed_edited_without_the_case_is_refused():
    config = _case().to_config()
    config["wind_speed_wind_ref"] = 5.0  # the wind spec still says 8
    with pytest.raises(ValueError, match="through the Case"):
        Case.from_config(config)


def test_the_lei_v3_as_config_has_no_operating_point():
    config = case.load_as_config(AS_CONFIG)
    assert not case.CASE_KEYS & set(config)
    assert config["tether"]["model"] == "williams"  # model block
    assert config["tether"]["n_elements"] == 10  # numerics block


def test_as_config_must_be_split_and_disjoint():
    with pytest.raises(ValueError, match="top-level blocks"):
        case.flatten_as_config({"is_with_gravity": False})
    with pytest.raises(ValueError, match="both the model and the numerics"):
        case.flatten_as_config({"model": {"a": {"x": 1}}, "numerics": {"a": {"x": 2}}})
    with pytest.raises(ValueError, match="the Case's"):
        case.flatten_as_config({"model": {"rho": 1.2}, "numerics": {}})


def test_overrides_cannot_set_the_operating_point():
    with pytest.raises(ValueError, match="blocks"):
        case.apply_overrides({}, {"angle_elevation_deg": 30.0})
    with pytest.raises(ValueError, match="the Case's"):
        case.apply_overrides({}, {"numerics": {"wind_speed_wind_ref": 5.0}})


def test_runtime_config_is_as_config_then_overrides_then_the_case():
    config = case.runtime_config(
        _case(), AS_CONFIG,
        {"model": {"is_with_gravity": True}, "output": {"is_with_final_plot": True}},
        defaults={"numerics": {"aerodynamic": {"n_aero_panels_per_struc_section": 2}}},
    )
    assert config["is_with_gravity"] is True
    assert config["is_with_final_plot"] is True
    assert config["is_save_geometry_snapshots"] is True  # output default
    assert config["aerodynamic"]["n_aero_panels_per_struc_section"] == 2
    assert config["aerodynamic"]["gamma_loop_type"] == "anderson"  # as_config kept
    assert Case.from_config(config) == _case()


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
    assert common.Case is case.Case


def test_air_density_at_case_constant_is_rho():
    assert case.air_density_at_case({"rho": 1.1}) == 1.1


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
    with pytest.raises(ValueError, match="air_density_law"):
        case.air_density_at_case({"rho": 1.225, "air_density_law": "isa"})

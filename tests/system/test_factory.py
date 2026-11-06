import math

import casadi as ca
import pytest

from awetrim.environment.profile_laws import explog_law, jet_law

from awetrim.system import create_system_model_from_yaml


def _write_config(path, wind):
    """An awesIO system file at ``path`` and its sibling ``rom_config.yaml``."""
    (path.parent / "rom_config.yaml").write_text(
        """
aerodynamics:
  model: coeffs
  params:
    CD0: 0.1
    CL0: 0.0
    angle_pitch_tether_0: 0.1
    slope_angle_pitch_tether_depower: 0.2
    gain_roll_steering: -0.15
  coefficients:
    CL: []
    CD: []
""",
        encoding="utf-8",
    )
    path.write_text(
        f"""
metadata:
  name: test-kite
  awesIO_version: 0.1.0
  schema: system_schema.yml
components:
  kites:
    - wing:
        structure:
          mass: 15
          projected_surface_area: 19.75
      control_system:
        structure:
          mass: 10
  tethers:
    - structure:
        diameter: 0.01
wind:
{wind}
""",
        encoding="utf-8",
    )


def test_factory_builds_logarithmic_wind_from_yaml(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: logarithmic
  z0: 0.03
  direction_wind: 0.25
  speed_wind_at_100: 10
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert model.wind.wind_model == "logarithmic"
    assert model.wind.z0 == 0.03
    assert model.wind.direction_wind == 0.25
    assert math.isclose(
        float(model.wind.speed_friction),
        0.41 * 10 / math.log(100 / 0.03),
    )


def test_factory_builds_uniform_wind_from_yaml(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: uniform
  direction_wind: 0
  speed_wind_ref: 8
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert model.wind.wind_model == "uniform"
    assert model.wind.speed_wind_ref == 8


def test_factory_power_law_speed_wind_at_100_is_the_speed_at_100_m(tmp_path):
    """Regression: the old factory converted speed_wind_at_100 through the
    log law for every non-uniform model, mis-scaling power_law."""
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: power_law
  alpha: 0.2
  direction_wind: 0
  speed_wind_at_100: 10
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert model.wind.wind_model == "power_law"
    assert model.wind.alpha == 0.2
    assert float(model.wind.speed_wind(100.0)) == pytest.approx(10.0)
    assert float(model.wind.speed_wind(50.0)) == pytest.approx(10.0 * 0.5**0.2)


def test_factory_builds_jet_and_explog_from_yaml(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: jet
  z0: 0.01
  height_ref: 100
  speed_wind_ref: 8
  jet_amplitude: 3
  jet_height: 150
  jet_width: 40
  direction_wind: 0
""",
    )
    model = create_system_model_from_yaml(config_path)
    assert model.wind.wind_model == "jet"
    assert float(model.wind.speed_wind(100.0)) == pytest.approx(
        jet_law(100.0, 8.0, 100.0, 0.01, 3.0, 150.0, 40.0)
    )

    _write_config(
        config_path,
        """  model: explog
  z0: 0.0002
  alpha: 0.08163
  speed_wind_at_ref: 8
  height_ref: 6
  direction_wind: 0
""",
    )
    model = create_system_model_from_yaml(config_path)
    assert model.wind.wind_model == "explog"
    assert float(model.wind.speed_wind(6.0)) == pytest.approx(8.0)
    assert float(model.wind.speed_wind(120.0)) == pytest.approx(
        explog_law(120.0, 8.0, 6.0, 0.0002, 0.08163)
    )


def test_system_model_exposes_named_expressions(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: uniform
  direction_wind: 0
  speed_wind_ref: 8
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert "angle_of_attack" in model.available_expressions()
    func = model.extract_function("angle_of_attack")
    assert isinstance(func, ca.Function)


# --- hardware actuator limits (system.yaml -> optimizer) --------------------


def test_extract_hardware_limits_maps_kcu_and_tether():
    from awetrim.system.factory import _extract_hardware_limits

    cs_struct = {
        "steering": {"range": [-0.35, 0.35], "rate": [-0.29, 0.29]},
        "depower": {"range": [1.1, 2.3], "rate": [-0.2, 0.2]},
    }
    tether_struct = {"length": 350.0}

    hw = _extract_hardware_limits(cs_struct, tether_struct)

    assert hw["input_steering"] == (-0.35, 0.35)
    assert hw["steering_rate"] == (-0.29, 0.29)
    assert hw["input_depower"] == (1.1, 2.3)
    assert hw["depower_rate"] == (-0.2, 0.2)
    assert hw["_max_tether_length"] == 350.0


def test_extract_hardware_limits_omits_missing_fields():
    from awetrim.system.factory import _extract_hardware_limits

    assert _extract_hardware_limits({}, {}) == {}


def test_extract_hardware_limits_maps_winch_drive_envelope():
    # ground_station.drums[0] (awesIO drum_object) -> the radial limits the
    # trajectory optimizer applies to v_r.
    from awetrim.system.factory import _extract_hardware_limits

    drum = {
        "type": "electric_winch",
        "min_tether_speed": -8.0,
        "max_tether_speed": 3.5,
        "max_winch_acceleration": 4.0,
    }

    hw = _extract_hardware_limits({}, {}, drum)

    assert hw["speed_radial"] == (-8.0, 3.5)
    # the acceleration capability is symmetric about zero
    assert hw["winch_acceleration"] == (-4.0, 4.0)


def test_extract_hardware_limits_ignores_half_specified_winch_speed():
    # One-sided speed data must not silently half-apply a bound.
    from awetrim.system.factory import _extract_hardware_limits

    hw = _extract_hardware_limits({}, {}, {"max_tether_speed": 3.5})

    assert "speed_radial" not in hw


def test_extract_hardware_limits_maps_drum_friction_only_when_complete():
    # drums[0].friction_coulomb / friction_viscous -> limits["winch_friction"],
    # read by opti_phase only when sim_parameters.winch_friction is set.
    from awetrim.system.factory import _extract_hardware_limits

    hw = _extract_hardware_limits({}, {}, {"friction_coulomb": 122.0, "friction_viscous": 30.7})
    assert hw["winch_friction"] == (122.0, 30.7)
    assert "winch_friction" not in _extract_hardware_limits({}, {}, {"friction_coulomb": 122.0})
    hw = _extract_hardware_limits({}, {}, {"friction_load_fraction": 0.05})
    assert hw["drivetrain_load_fraction"] == 0.05
    assert "drivetrain_load_fraction" not in _extract_hardware_limits({}, {}, {})


def test_system_yaml_winch_envelope_reaches_the_optimizer_limits():
    from awetrim.timeseries.phase_parametrized import PhaseParameterized
    from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG

    model = create_system_model_from_yaml(LEI_V3_SYSTEM_CONFIG)

    class _Stub:
        kite_model = model
        _resolve_opti_limits = PhaseParameterized._resolve_opti_limits

    lim = _Stub()._resolve_opti_limits()

    assert lim["speed_radial"] == model.hardware_limits["speed_radial"]
    assert lim["winch_acceleration"] == model.hardware_limits["winch_acceleration"]
    assert lim["winch_friction"] == model.hardware_limits["winch_friction"]


def test_depower_box_is_clipped_to_the_rom_validity_band():
    # The actuator range alone let the optimizer run the polynomial ROM far
    # below the depower it was identified on.
    import yaml

    from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG
    from awetrim.system.factory import resolve_rom_config_path

    from awetrim.utils.system_config import get_kite

    model = create_system_model_from_yaml(LEI_V3_SYSTEM_CONFIG)
    with open(resolve_rom_config_path(LEI_V3_SYSTEM_CONFIG), encoding="utf-8") as f:
        band = yaml.safe_load(f)["validity"]["input_depower"]
    with open(LEI_V3_SYSTEM_CONFIG, encoding="utf-8") as f:
        box = get_kite(yaml.safe_load(f))["control_system"]["structure"]["depower"]["range"]
    lo, hi = model.hardware_limits["input_depower"]

    assert lo == pytest.approx(max(min(box), min(band)))
    assert hi == pytest.approx(min(max(box), max(band)))


def test_intersect_depower_validity_rules():
    from awetrim.system.factory import intersect_depower_validity

    assert intersect_depower_validity({"a": 1}, None) == {"a": 1}
    hw = intersect_depower_validity({"input_depower": (1.1, 2.1)}, (1.6, 2.2))
    assert hw["input_depower"] == (1.6, 2.1)
    with pytest.raises(ValueError):
        intersect_depower_validity({"input_depower": (1.1, 1.3)}, (1.6, 2.2))


def test_intersect_steering_validity_rules():
    from awetrim.system.factory import intersect_steering_validity

    assert intersect_steering_validity({"a": 1}, None) == {"a": 1}
    hw = intersect_steering_validity({"input_steering": (-0.175, 0.175)}, 0.1429)
    assert hw["input_steering"] == (-0.1429, 0.1429)
    hw = intersect_steering_validity({"input_steering": (-0.1, 0.1)}, 0.1429)
    assert hw["input_steering"] == (-0.1, 0.1)


def test_wing_point_mass_carries_the_bridle():
    import yaml

    from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG
    from awetrim.utils.system_config import get_kite

    kite = get_kite(yaml.safe_load(open(LEI_V3_SYSTEM_CONFIG, encoding="utf-8")))
    model = create_system_model_from_yaml(LEI_V3_SYSTEM_CONFIG)
    assert model.kite.mass_wing == pytest.approx(
        kite["wing"]["structure"]["mass"] + kite["bridle"]["structure"]["mass"]
    )


def test_apply_tension_floor_rules():
    from awetrim.system.factory import apply_tension_floor
    from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

    assert apply_tension_floor({"a": 1}, None) == {"a": 1}
    hw = apply_tension_floor({}, 602.0)
    assert hw["tension_tether_ground"] == (
        602.0, DEFAULT_OPTI_LIMITS["tension_tether_ground"][1]
    )
    # a floor below the existing bound leaves it alone
    assert apply_tension_floor({"tension_tether_ground": (800.0, 1e5)}, 602.0) == {
        "tension_tether_ground": (800.0, 1e5)
    }
    with pytest.raises(ValueError):
        apply_tension_floor({"tension_tether_ground": (300.0, 500.0)}, 602.0)


def test_a_flat_legacy_system_file_is_rejected(tmp_path):
    config_path = tmp_path / "lei_v3_system_config.yaml"
    config_path.write_text("wing:\n  mass: 15\n  area: 19.75\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an awesIO system file"):
        create_system_model_from_yaml(config_path)


def test_resolve_opti_limits_merges_hardware_over_defaults():
    from awetrim.timeseries.phase_parametrized import PhaseParameterized
    from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

    class _Stub:
        kite_model = type(
            "_KM",
            (),
            {
                "hardware_limits": {
                    "input_steering": (-0.4, 0.4),
                    "_max_tether_length": 350.0,
                }
            },
        )()
        _resolve_opti_limits = PhaseParameterized._resolve_opti_limits

    lim = _Stub()._resolve_opti_limits()

    # hardware value overrides the default
    assert lim["input_steering"] == (-0.4, 0.4)
    # distance_radial upper bound = tether length, lower bound kept from default
    assert lim["distance_radial"] == (DEFAULT_OPTI_LIMITS["distance_radial"][0], 350.0)
    # the sentinel is consumed, not exposed as a limit
    assert "_max_tether_length" not in lim
    # untouched keys fall through to the defaults
    assert lim["height"] == DEFAULT_OPTI_LIMITS["height"]


def test_resolve_opti_limits_empty_is_exactly_defaults():
    from awetrim.timeseries.phase_parametrized import PhaseParameterized
    from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

    class _Stub:
        kite_model = type("_KM", (), {"hardware_limits": {}})()
        _resolve_opti_limits = PhaseParameterized._resolve_opti_limits

    assert _Stub()._resolve_opti_limits() == dict(DEFAULT_OPTI_LIMITS)


def test_a_rom_run_flies_the_tether_its_rom_file_states(tmp_path):
    """The ROM file's ``tether`` block selects the model of a ROM run; an
    explicit ``tether_config`` (an aerostructural run's as_config) wins."""
    from awetrim.system.tether import RigidLumpedTether
    from awetrim.system.williams_tether import WilliamsTether

    config_path = tmp_path / "kite.yaml"
    _write_config(config_path, "  model: uniform\n  speed_wind_ref: 8.0")
    assert isinstance(create_system_model_from_yaml(config_path).tether, RigidLumpedTether)

    rom = tmp_path / "rom_config.yaml"
    rom.write_text(rom.read_text(encoding="utf-8") + "tether:\n  model: williams\n  n_elements: 4\n",
                   encoding="utf-8")
    assert isinstance(create_system_model_from_yaml(config_path).tether, WilliamsTether)
    explicit = create_system_model_from_yaml(config_path, tether_config={"model": "rigid_lumped"})
    assert isinstance(explicit.tether, RigidLumpedTether)


@pytest.mark.parametrize(
    "rom", ["rom_config_aerostructural.yaml", "rom_config_aerostructural_flight_corrected.yaml"]
)
def test_lei_v3_roms_fly_the_rigid_lumped_tether(rom):
    from awetrim.system.tether import RigidLumpedTether
    from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG

    model = create_system_model_from_yaml(LEI_V3_SYSTEM_CONFIG, aero_yaml_path=rom)
    assert isinstance(model.tether, RigidLumpedTether)


def test_extract_hardware_limits_maps_drum_inertia():
    # drums[0].inertia_equivalent_mass [kg at the tether] -> limits["winch_inertia_mass"]
    from awetrim.system.factory import _extract_hardware_limits

    hw = _extract_hardware_limits({}, {}, {"inertia_equivalent_mass": 300.7})
    assert hw["winch_inertia_mass"] == 300.7
    assert "winch_inertia_mass" not in _extract_hardware_limits({}, {}, {})


def test_lumped_tether_inertia_from_config():
    from awetrim.system.factory import create_tether_from_config

    assert not create_tether_from_config({"model": "rigid_lumped"}, diameter=0.008, density=724.0).inertia
    assert create_tether_from_config(
        {"model": "rigid_lumped", "inertia": True}, diameter=0.008, density=724.0
    ).inertia
    with pytest.raises(ValueError, match="lumped tether only"):
        create_tether_from_config(
            {"model": "williams", "inertia": True}, diameter=0.008, density=724.0
        )

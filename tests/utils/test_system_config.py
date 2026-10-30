"""KCU mass and onboard turbine accessors of the awesIO system layout."""

import pytest
import yaml

from awetrim.system.factory import create_system_model_from_yaml
from awetrim.utils.config_paths import LEI_V3_SYSTEM_FLOWN_2019_CONFIG
from awetrim.utils.system_config import get_kcu_mass, get_turbine


def _system(control_system: dict) -> dict:
    return {"components": {"kites": [{"control_system": control_system}]}}


def test_kcu_mass_without_turbine_is_the_body_mass():
    assert get_kcu_mass(_system({"structure": {"mass": 8.4}})) == pytest.approx(8.4)


def test_kcu_mass_adds_the_turbine_mass_once():
    system = _system(
        {
            "structure": {"mass": 19.2},
            "turbine": {"structure": {"rotor_diameter": 0.5, "mass": 2.8}},
        }
    )
    assert get_kcu_mass(system) == pytest.approx(22.0)


def test_kcu_mass_is_none_when_the_file_states_none():
    assert get_kcu_mass(_system({})) is None


def test_no_turbine_block_is_an_empty_structure():
    assert get_turbine(_system({"structure": {"mass": 8.4}})) == {}


def test_legacy_flat_turbine_block_keeps_its_drag_and_adds_no_mass():
    """Pre-awesIO ``turbine: {diameter, thrust_coefficient}``, mass in the KCU."""
    system = _system(
        {
            "structure": {"mass": 22.0},
            "turbine": {"diameter": 0.5, "thrust_coefficient": 0.85},
        }
    )
    with pytest.warns(DeprecationWarning):
        assert get_turbine(system) == {"rotor_diameter": 0.5}
    with pytest.warns(DeprecationWarning):
        assert get_kcu_mass(system) == pytest.approx(22.0)


def test_flown_2019_kcu_keeps_its_22_kg_with_the_turbine_split_out():
    """19.2 kg KCU + 2.8 kg turbine, stated separately, lumped back to 22.0."""
    with open(LEI_V3_SYSTEM_FLOWN_2019_CONFIG, encoding="utf-8") as f:
        system = yaml.safe_load(f)
    assert get_turbine(system)["mass"] == pytest.approx(2.8)
    assert get_kcu_mass(system) == pytest.approx(22.0)


def test_flown_2019_system_model_carries_the_turbine():
    kite = create_system_model_from_yaml(LEI_V3_SYSTEM_FLOWN_2019_CONFIG).kite
    assert kite.mass_kcu == pytest.approx(22.0)
    assert kite.diameter_turbine == pytest.approx(0.5)
    assert kite.thrust_coefficient_turbine == pytest.approx(0.85)

"""Tests for the full-model geometry reader (coupled/read_struc_geometry_yaml.py)."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerostructural.coupled import read_struc_geometry_yaml
from awetrim.utils.system_config import get_kite

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


@pytest.fixture(scope="module")
def fem_full_geometry():
    path = KITE_DIR / "struc_geometry_FEM_full.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def system_config():
    return yaml.safe_load((KITE_DIR / "system.yaml").read_text(encoding="utf-8"))


def _masses(geometry, system_config, kcu_mass):
    system = copy.deepcopy(system_config)
    get_kite(system)["control_system"]["structure"]["mass"] = kcu_mass
    # The reader mutates the geometry it is handed; give it its own copy.
    reader = read_struc_geometry_yaml.main(copy.deepcopy(geometry), system_config=system)
    return np.asarray(reader[1], dtype=float)


def test_the_kcu_mass_does_not_move_mass_off_the_wing(fem_full_geometry, system_config):
    light = _masses(fem_full_geometry, system_config, 8.4)
    heavy = _masses(fem_full_geometry, system_config, 22.0)

    assert light[0] == pytest.approx(heavy[0] - 13.6)
    np.testing.assert_allclose(light[1:], heavy[1:])


def test_no_node_carries_negative_mass(fem_full_geometry, system_config):
    masses = _masses(fem_full_geometry, system_config, 22.0)

    assert masses.min() >= 0.0
    # Wing (canopy + tubes) plus the bridle and its pulleys, on top of the KCU.
    assert masses[1:].sum() > fem_full_geometry["mass_without_bridles"]

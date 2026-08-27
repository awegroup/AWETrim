"""Tests for awetrim.aerostructural.pss.structural_geometry_io."""

import copy
from pathlib import Path

import pytest
import yaml

from awetrim.aerostructural.pss.structural_geometry_io import (
    _resolve_kcu_mass,
    main as pss_initialize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KITE_DIR = PROJECT_ROOT / "data" / "LEI-V3-KITE"

#: Index of ``power_tape_index`` in the ``main`` return tuple.
_POWER_TAPE_INDEX = 4


@pytest.fixture(scope="module")
def struc_geometry() -> dict:
    path = KITE_DIR / "struc_geometry.yaml"
    if not path.exists():  # pragma: no cover - shipped with the repo
        pytest.skip(f"{path} not available")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def system_config() -> dict:
    path = KITE_DIR / "system.yaml"
    if not path.exists():  # pragma: no cover - shipped with the repo
        pytest.skip(f"{path} not available")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_geometry_without_a_depower_tape_reports_no_power_tape(struc_geometry):
    """A missing ``depower_tape`` must give ``None``, not an UnboundLocalError.

    ``power_tape_index`` is typed ``int | None`` and every consumer guards on
    ``is not None``; leaving the local unbound crashed the whole initializer on
    any geometry that has no depower tape.
    """
    sg = copy.deepcopy(struc_geometry)
    rows = sg["bridle_connections"]["data"]
    kept = [row for row in rows if row[0] != "depower_tape"]
    assert len(kept) < len(rows), "fixture no longer has a depower_tape row"
    sg["bridle_connections"]["data"] = kept

    result = pss_initialize(sg)
    assert result[_POWER_TAPE_INDEX] is None


def test_depower_tape_is_indexed_when_present(struc_geometry):
    result = pss_initialize(copy.deepcopy(struc_geometry))
    power_tape_index = result[_POWER_TAPE_INDEX]
    assert isinstance(power_tape_index, int)
    connectivity = result[7]
    assert 0 <= power_tape_index < len(connectivity)


def test_kcu_mass_comes_from_the_system_config(struc_geometry, system_config):
    """system.yaml is the single source; struc_geometry.kcu_mass is deprecated."""
    sg = copy.deepcopy(struc_geometry)
    sg.pop("kcu_mass", None)

    from_system = _resolve_kcu_mass(sg, system_config=system_config)
    assert from_system > 0.0

    # Node 0 is the KCU/bridle point: it carries the KCU mass plus its share of
    # the bridle lines, so the whole difference must land there.
    with_system = pss_initialize(sg, system_config=system_config)[1]
    without = pss_initialize(sg)[1]
    assert with_system[0] - without[0] == pytest.approx(from_system)
    assert sum(with_system) - sum(without) == pytest.approx(from_system)


def test_element_stiffness_overrides_win_over_the_material_table(struc_geometry):
    """A snapshot's selected stiffnesses are re-applied on reload, per node pair."""
    import numpy as np

    parsed = pss_initialize(struc_geometry)
    connectivity = np.asarray(
        [[int(row[0]), int(row[1])] for row in parsed[7]], dtype=int
    )
    k_base = np.asarray(parsed[11], dtype=float)

    # One wing element and ONE arm of a pulley bridle line: the other arm, and
    # the same line on the other side of the kite, must stay untouched.
    wing_index, pulley_index = 0, len(k_base) - 1
    overridden = copy.deepcopy(struc_geometry)
    overridden["element_stiffness"] = {
        "headers": ["node_i", "node_j", "k"],
        "data": [
            [*connectivity[wing_index].tolist(), 3.2e5],
            [*connectivity[pulley_index].tolist(), 9.9e4],
        ],
    }
    k_over = np.asarray(pss_initialize(overridden)[11], dtype=float)

    changed = np.flatnonzero(k_base != k_over)
    assert set(changed.tolist()) == {wing_index, pulley_index}
    assert k_over[wing_index] == 3.2e5
    assert k_over[pulley_index] == 9.9e4

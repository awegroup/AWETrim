"""Tests for the awesIO structure document (coupled/awesio_structure.py)."""

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerostructural.coupled import awesio_structure

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"
LEGACY_DIR = Path(__file__).resolve().parents[1] / "data" / "legacy"
KCU_MASS = 8.4


def test_the_kites_file_validates_against_the_schema():
    validator = pytest.importorskip("awesio.validator")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validator.validate(KITE_DIR / "structure.yml")


def test_the_connectivity_preimage_is_the_schemas_example():
    three_in_a_row = {
        "points": {"data": [["1"], ["2"], ["3"]]},
        "segments": {"data": [["s1", ["1", "2"]], ["s2", ["2", "3"]]]},
        "bodies": {"data": [["a"], ["b"]]},
        "tubes": {"data": [["t", ["a", "b"]]]},
        "canopy_faces": {"data": [["f", "w", ["1", "2", "3"]]]},
    }
    assert awesio_structure.connectivity_preimage(three_in_a_row) == "3;1,2;2,3;2;1,2;1;1,2,3;"


# --- the Structure view the drivers use --------------------------------------


@pytest.fixture(scope="module")
def structure():
    return awesio_structure.load_structure(KITE_DIR / "structure.yml")


def test_the_stored_sha_hashes_the_connectivity(structure):
    document = structure.document
    assert awesio_structure.connectivity_sha(document) == document["metadata"]["connectivity_sha"]


def test_the_masses_follow_awesios_rule(structure):
    masses = structure.masses(kcu_mass=KCU_MASS)
    assert masses.min() >= 0.0
    assert masses.sum() == pytest.approx(
        awesio_structure.nodal_masses(structure.document).sum() + KCU_MASS
    )
    assert structure.resolved(kcu_mass=KCU_MASS)[1][structure.kcu] >= KCU_MASS


def test_the_structure_accessors(structure):
    assert structure.kcu == 0 and structure.fixed == [0]
    assert not structure.is_wireframe
    assert len(structure.stations) == len(structure.leading_edge)  # one per rib
    assert len(structure.trailing_edge) == len(structure.leading_edge)
    assert structure.leading_edge_chain == structure.leading_edge
    assert structure.pressure_bar > 0.0
    # a pulley is one bridle element of two segments
    bridle = structure.bridle
    assert len(structure.bridle_line_system()) == sum(len(e["nodes"]) - 1 for e in bridle)
    pulleys = [e for e in bridle if len(e["nodes"]) == 3]
    assert pulleys and all(len(e["segments"]) == 2 for e in pulleys)
    assert [e["name"] for e in bridle].count("steering_tape") == 2


def test_flat_tapes_carry_a_drag_width(structure):
    tapes = [s for s in structure.bridle_segments if s["line"] in ("depower_tape", "steering_tape")]
    assert tapes and all(s.get("tape_width") for s in tapes)


def test_with_positions_moves_points_and_bodies(structure):
    moved = structure.with_positions(structure.positions + [0.0, 0.0, 1.0])
    np.testing.assert_allclose(moved.positions, structure.positions + [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        moved.resolved()[0], structure.resolved()[0] + [0.0, 0.0, 1.0], atol=1e-12
    )
    assert structure.document is not moved.document  # a copy


def test_with_rest_lengths_changes_only_the_named_segments(structure):
    tape = next(e for e in structure.bridle if e["name"] == "depower_tape")
    name = tape["segments"][0]
    longer = structure.with_rest_lengths({name: tape["rest_length"] + 0.1})
    after = {s["name"]: s["rest_length"] for s in longer.segments}
    before = {s["name"]: s["rest_length"] for s in structure.segments}
    assert after[name] == pytest.approx(before[name] + 0.1)
    assert {k: v for k, v in after.items() if k != name} == {
        k: v for k, v in before.items() if k != name
    }


def test_written_structure_reads_back(structure, tmp_path):
    path = tmp_path / "structure.yml"
    structure.with_positions(structure.positions * 1.01).write(path)
    again = awesio_structure.load_structure(path)
    np.testing.assert_allclose(again.positions, structure.positions * 1.01, atol=1e-12)
    assert again.document["metadata"]["connectivity_sha"] == (
        structure.document["metadata"]["connectivity_sha"]
    )


def test_a_retired_format_file_is_rejected(tmp_path):
    path = tmp_path / "struc_geometry_FEM_full.yaml"
    path.write_text(yaml.safe_dump({"strut_tubes": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an awesIO"):
        awesio_structure.load_structure(path)


def test_an_in_memory_wireframe_dict_is_translated(legacy_wireframe):
    structure = awesio_structure.load_structure(copy.deepcopy(legacy_wireframe))
    assert structure.is_wireframe


# --- the wireframe -----------------------------------------------------------


@pytest.fixture(scope="module")
def legacy_wireframe():
    """A PSM-format wireframe, the wireframe's in-memory geometry."""
    path = LEGACY_DIR / "struc_geometry_wireframe_powered_2019.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_deformed_snapshot_is_the_converted_wireframe(legacy_wireframe):
    snapshot = awesio_structure.load_structure(
        KITE_DIR / "deformed_results" / "powered_2019" / "structure.yml"
    )
    converted = awesio_structure.Structure(awesio_structure.wireframe_to_awesio(legacy_wireframe))
    assert snapshot.is_wireframe
    np.testing.assert_allclose(snapshot.positions, converted.positions, atol=1e-12)
    assert [e["rest_length"] for e in snapshot.bridle] == pytest.approx(
        [e["rest_length"] for e in converted.bridle]
    )


def test_the_wireframe_round_trips(legacy_wireframe):
    document = awesio_structure.wireframe_to_awesio(legacy_wireframe)
    back = awesio_structure.wireframe_from_awesio(document)
    for key in ("wing_particles", "bridle_particles", "bridle_connections", "wing_connections"):
        assert back[key]["data"] == legacy_wireframe[key]["data"], key
    np.testing.assert_allclose(back["bridle_point_node"], legacy_wireframe["bridle_point_node"])
    rows = {r[0]: r for r in legacy_wireframe["bridle_lines"]["data"]}
    for row in back["bridle_lines"]["data"]:
        assert row[1] == pytest.approx(float(rows[row[0]][1]), abs=1e-12)


def test_an_actuated_tape_survives_the_wireframe_round_trip(legacy_wireframe):
    from awetrim.aerostructural.utils import (
        BRIDLE_CONNECTION_REST_LENGTHS,
        bridle_connection_rest_lengths,
    )

    psm = copy.deepcopy(legacy_wireframe)
    rows = [i for i, c in enumerate(psm["bridle_connections"]["data"]) if c[0] == "steering_tape"]
    l0 = {r[0]: float(r[1]) for r in psm["bridle_lines"]["data"]}["steering_tape"]
    psm[BRIDLE_CONNECTION_REST_LENGTHS] = {
        "headers": ["row", "l0"], "data": [[rows[0], l0 - 0.05], [rows[1], l0 + 0.05]]
    }
    back = awesio_structure.wireframe_from_awesio(awesio_structure.wireframe_to_awesio(psm))
    rest = {r[0]: float(r[1]) for r in back["bridle_lines"]["data"]}["steering_tape"]
    overrides = bridle_connection_rest_lengths(back)
    lengths = [overrides.get(row, rest) for row in rows]
    assert lengths == pytest.approx([l0 - 0.05, l0 + 0.05])

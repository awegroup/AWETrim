"""Tests for the awesIO structure document (coupled/awesio_structure.py)."""

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerostructural.coupled import awesio_structure, read_struc_geometry_yaml

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"
LEGACY_DIR = Path(__file__).resolve().parents[1] / "data" / "legacy"
KCU_MASS = 8.4


@pytest.fixture(scope="module")
def fem_full_geometry():
    path = LEGACY_DIR / "struc_geometry_FEM_full.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reader(fem_full_geometry):
    return read_struc_geometry_yaml.main(
        copy.deepcopy(fem_full_geometry), config={"kcu": {"mass": KCU_MASS}}
    )


@pytest.fixture(scope="module")
def written(fem_full_geometry, tmp_path_factory):
    """The document as a reader sees it: written to YAML and loaded back."""
    document = awesio_structure.to_awesio(
        fem_full_geometry, kcu_mass=KCU_MASS, height_kcu=200.0, name="LEI-V3"
    )
    path = tmp_path_factory.mktemp("awesio") / "structure.yml"
    awesio_structure.write_awesio(document, path)
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_document_validates_against_the_schema(written):
    validator = pytest.importorskip("awesio.validator")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validator.validate(written[0])


def test_the_input_geometry_is_not_modified(fem_full_geometry):
    before = copy.deepcopy(fem_full_geometry)
    awesio_structure.to_awesio(fem_full_geometry)
    assert fem_full_geometry == before


def test_the_connectivity_preimage_is_the_schemas_example():
    three_in_a_row = {
        "points": {"data": [["1"], ["2"], ["3"]]},
        "segments": {"data": [["s1", ["1", "2"]], ["s2", ["2", "3"]]]},
        "bodies": {"data": [["a"], ["b"]]},
        "tubes": {"data": [["t", ["a", "b"]]]},
        "canopy_faces": {"data": [["f", "w", ["1", "2", "3"]]]},
    }
    assert awesio_structure.connectivity_preimage(three_in_a_row) == "3;1,2;2,3;2;1,2;1;1,2,3;"


def test_the_written_sha_hashes_the_written_connectivity(written):
    document = written[1]
    assert awesio_structure.connectivity_sha(document) == document["metadata"]["connectivity_sha"]


def test_the_round_trip_rebuilds_the_readers_model(reader, written):
    rebuilt = awesio_structure.from_awesio(written[1])

    np.testing.assert_allclose(rebuilt[0], np.asarray(reader[0]), atol=1e-12)
    for k in (2, 3, 4, 5, 6, 7, 8, 10, 11, 17):
        np.testing.assert_array_equal(np.asarray(rebuilt[k]), np.asarray(reader[k]))
    np.testing.assert_allclose(rebuilt[12], reader[12])  # bridle diameters
    np.testing.assert_allclose(rebuilt[13], reader[13], atol=1e-12)  # rest lengths
    np.testing.assert_allclose(rebuilt[14], reader[14], rtol=1e-6)  # stiffness
    np.testing.assert_allclose(rebuilt[15], reader[15])  # damping / pressure
    assert list(rebuilt[16]) == list(reader[16])  # linktypes
    assert rebuilt[18].keys() == reader[18].keys()
    for key, pair in reader[18].items():
        np.testing.assert_allclose(rebuilt[18][key], pair, atol=1e-8)


def test_the_mass_total_is_the_readers(reader, written):
    masses = awesio_structure.nodal_masses(written[1])
    # The reader weighs a line at its rest length BEFORE the strut projection.
    assert masses.sum() == pytest.approx(np.sum(reader[1]), rel=1e-9)
    assert masses.min() >= 0.0
    # The KCU node also carries half of each tape ending on it.
    assert written[1]["points"]["data"][0][4] == pytest.approx(KCU_MASS)


def test_flat_tapes_keep_their_drag_width(fem_full_geometry, written):
    widths = {
        row[0]: row[5] for row in fem_full_geometry["bridle_lines"]["data"] if len(row) > 5
    }
    segments = awesio_structure._rows(written[1]["segments"])
    for segment in segments:
        assert segment["tape_width"] == widths.get(segment["line"])


# --- the Structure view the drivers use --------------------------------------


@pytest.fixture(scope="module")
def structure():
    return awesio_structure.load_structure(KITE_DIR / "structure.yml")


def test_the_kites_file_is_the_converted_legacy_model(structure, reader):
    resolved = structure.resolved(kcu_mass=KCU_MASS)
    np.testing.assert_allclose(resolved[0], np.asarray(reader[0]), atol=1e-9)
    for k in (2, 3, 4, 5, 6, 7, 8, 10, 11, 17):
        np.testing.assert_array_equal(np.asarray(resolved[k]), np.asarray(reader[k]))
    np.testing.assert_allclose(resolved[13], reader[13], atol=1e-9)
    # masses follow awesIO's rule: the same total, shared differently
    assert np.sum(resolved[1]) == pytest.approx(np.sum(reader[1]), rel=1e-9)
    assert resolved[1][structure.kcu] >= KCU_MASS


def test_the_structure_accessors(structure, reader):
    assert structure.kcu == 0 and structure.fixed == [0]
    assert not structure.is_wireframe
    assert len(structure.stations) == len(reader[2])  # one station per rib
    assert structure.leading_edge == list(reader[2])
    assert structure.trailing_edge == list(reader[3])
    assert structure.leading_edge_chain == list(reader[2])
    assert [s["nodes"] for s in structure.struts] == sorted(
        (list(s) for s in reader[8]), key=lambda s: s[0]
    )
    assert structure.pressure_bar == pytest.approx(float(reader[15][0]))
    # one bridle element per old bridle_connections row, a pulley as one
    bridle = structure.bridle
    assert sum(len(e["nodes"]) - 1 for e in bridle) == len(reader[11])
    assert len(structure.bridle_line_system()) == len(reader[11])
    pulleys = [e for e in bridle if len(e["nodes"]) == 3]
    assert all(len(e["segments"]) == 2 for e in pulleys)
    assert [e["name"] for e in bridle].count("steering_tape") == 2


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


def test_a_legacy_file_is_converted_with_a_warning(structure):
    with pytest.warns(DeprecationWarning, match="retired"):
        converted = awesio_structure.load_structure(LEGACY_DIR / "struc_geometry_FEM_full.yaml")
    np.testing.assert_allclose(converted.positions, structure.positions, atol=1e-9)
    assert converted.bridle_segments == structure.bridle_segments


# --- the wireframe -----------------------------------------------------------


@pytest.fixture(scope="module")
def legacy_wireframe():
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

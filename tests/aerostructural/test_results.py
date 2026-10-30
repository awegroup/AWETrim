"""Unit tests for the pure helpers in awetrim.aerostructural.results.

Covers the sweep-value generator, the restart directory-candidate logic, the
result-root path builders, and the deformed awesIO snapshot (built from the
full model or a wireframe, written by ``save_geometry_snapshot``). None of
these touch a solver.
"""

from pathlib import Path

import numpy as np
import pytest

from awetrim.aerostructural.results import (
    aerostructural_results_root,
    build_deformed_struc_geometry,
    candidate_case_dirs,
    legacy_results_root,
    steering_values_from_count_or_step,
)


# --- steering_values_from_count_or_step ------------------------------------


def test_steering_values_single_point():
    out = steering_values_from_count_or_step(0.1, 0.5, n_values=1)
    assert out == pytest.approx([0.1])


def test_steering_values_count_is_inclusive_linspace():
    out = steering_values_from_count_or_step(0.0, 2.0, n_values=3)
    assert out == pytest.approx([0.0, 1.0, 2.0])


def test_steering_values_step_is_inclusive_of_end():
    out = steering_values_from_count_or_step(0.0, 1.0, step_m=0.5)
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_steering_values_rejects_zero_count():
    with pytest.raises(ValueError):
        steering_values_from_count_or_step(0.0, 1.0, n_values=0)


def test_steering_values_requires_step_or_count():
    with pytest.raises(ValueError):
        steering_values_from_count_or_step(0.0, 1.0)


def test_steering_values_rejects_reversed_range_with_step():
    with pytest.raises(ValueError):
        steering_values_from_count_or_step(1.0, 0.0, step_m=0.5)


# --- result-root path builders ---------------------------------------------


def test_results_roots():
    root = Path("/proj")
    assert aerostructural_results_root(root, "V3") == root / "results" / "V3" / "aerostructural"
    assert legacy_results_root(root, "V3") == root / "results" / "aerostructural" / "V3"


# --- candidate_case_dirs ----------------------------------------------------


def test_candidate_case_dirs_adds_sign_variant():
    dirs = candidate_case_dirs(Path("/proj"), "V3", "depower_p0000mm_steer_p0150mm")
    names = {d.name for d in dirs}
    # The p0000mm token is mirrored to an m0000mm variant.
    assert "depower_p0000mm_steer_p0150mm" in names
    assert "depower_m0000mm_steer_p0150mm" in names
    # Two roots (canonical + legacy) times two variants.
    assert len(dirs) == 4
    assert all(isinstance(d, Path) for d in dirs)


def test_candidate_case_dirs_without_zero_token_has_single_variant():
    dirs = candidate_case_dirs(Path("/proj"), "V3", "depower_p0150mm_steer_p0150mm")
    # No p0000mm/m0000mm token to mirror -> one variant per root.
    assert len(dirs) == 2


# --- build_deformed_struc_geometry -----------------------------------------

import copy  # noqa: E402

import yaml  # noqa: E402

from awetrim.aerostructural.coupled.awesio_structure import (  # noqa: E402
    Structure,
    load_structure,
)
from awetrim.aerostructural.results import save_geometry_snapshot  # noqa: E402

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"
LEGACY_DIR = Path(__file__).resolve().parents[1] / "data" / "legacy"


@pytest.fixture(scope="module")
def full_structure():
    return load_structure(KITE_DIR / "structure.yml")


def _toy_geometry():
    """A PSM-format wireframe: two steering tapes on one line, a depower tape
    and one pulley rope."""
    return {
        "bridle_point_node": [0.0, 0.0, 0.0],
        "pulley_mass": 0.0,
        "dyneema": {"density": 970.0, "youngs_modulus": 5.5e10, "damping_per_stiffness": 0.0},
        "fixed_point_indices": [0],
        "wing_particles": {"headers": ["id", "x", "y", "z"],
                           "data": [[1, 0.0, 1.0, 5.0], [2, 0.0, -1.0, 5.0]]},
        "wing_connections": {"headers": ["name", "ci", "cj"], "data": [["span", 1, 2]]},
        "wing_elements": {"headers": ["name", "l0", "k", "c", "m", "linktype"],
                          "data": [["span", 2.0, 1.0e4, 0.0, 1.0, "noncompressive"]]},
        "bridle_particles": {"headers": ["id", "x", "y", "z"],
                             "data": [[3, 0.0, 0.0, 2.0], [4, 0.0, 0.0, 1.0]]},
        "bridle_lines": {
            "headers": ["name", "l0", "d", "material", "linktype"],
            "data": [
                ["steering_tape", 1.6, 0.005, "dyneema", "noncompressive"],
                ["M", 9.0, 0.004, "dyneema", "pulley"],
                ["depower_tape", 1.5, 0.005, "dyneema", "noncompressive"],
            ],
        },
        "bridle_connections": {
            "headers": ["name", "ci", "cj", "ck"],
            "data": [
                ["steering_tape", 0, 1],
                ["M", 1, 3, 2],
                ["steering_tape", 0, 2],
                ["depower_tape", 0, 4],
            ],
        },
    }


def _nodes():
    return np.array([[0, 0, 0], [0, 1, 5], [0, -1, 5], [0, 0, 2], [0, 0, 1]], dtype=float)


def _tapes(structure):
    """Rest length of every straight bridle connection, in row order."""
    return [(e["name"], e["rest_length"]) for e in structure.bridle if len(e["nodes"]) == 2]


def test_snapshot_moves_the_points_by_index(full_structure):
    nodes = full_structure.positions + [0.1, 0.0, -0.2]
    # a backend may append nodes of its own (Billow's canopy refinement)
    extra = np.vstack([nodes, np.zeros((5, 3))])
    out = build_deformed_struc_geometry(full_structure, extra)
    assert isinstance(out, Structure)
    np.testing.assert_allclose(out.positions, nodes, atol=1e-12)
    # the input is not modified
    assert np.abs(full_structure.positions - nodes).max() > 0.1


def test_snapshot_of_a_wireframe_dict_is_awesio():
    geometry = _toy_geometry()
    out = build_deformed_struc_geometry(geometry, _nodes() * 2.0)
    assert out.is_wireframe
    np.testing.assert_allclose(out.positions, _nodes() * 2.0)
    assert geometry["wing_particles"]["data"][0][1:] == [0.0, 1.0, 5.0]  # input untouched


def test_snapshot_writes_the_live_tape_lengths():
    # element order: steering, M arm, M arm, steering, depower
    live = np.array([1.58, 4.4, 4.6, 1.62, 1.58])
    out = build_deformed_struc_geometry(_toy_geometry(), _nodes(), bridle_rest_lengths=live)
    assert _tapes(out) == [
        ("steering_tape", pytest.approx(1.58)),
        ("steering_tape", pytest.approx(1.62)),
        ("depower_tape", pytest.approx(1.58)),
    ]
    rope = next(e for e in out.bridle if e["name"] == "M")
    assert rope["rest_length"] == pytest.approx(9.0)  # pulley rope: never actuated, kept


def test_snapshot_rest_lengths_skip_nan_and_check_the_count():
    live = np.array([np.nan, 4.4, 4.6, np.nan, 1.58])
    out = build_deformed_struc_geometry(_toy_geometry(), _nodes(), bridle_rest_lengths=live)
    assert [l0 for _, l0 in _tapes(out)] == pytest.approx([1.6, 1.6, 1.58])
    with pytest.raises(ValueError):
        build_deformed_struc_geometry(_toy_geometry(), _nodes(), bridle_rest_lengths=live[:4])


def _actuate(l0, power, left, right):
    l0 = np.asarray(l0, dtype=float).copy()
    l0[power] += 0.08
    l0[left] -= 0.03
    l0[right] += 0.03
    return l0


def test_full_model_snapshot_reads_the_tapes_back(full_structure):
    first = full_structure.resolved()
    power, (left, right) = first[4], first[5][:2]
    l0 = _actuate(first[13], power, left, right)
    snapshot = build_deformed_struc_geometry(
        full_structure, first[0], bridle_rest_lengths=l0[-len(first[11]):]
    )
    again = snapshot.resolved()[13]
    tapes = [power, left, right]
    np.testing.assert_allclose(again[tapes], l0[tapes], rtol=0, atol=1e-12)
    np.testing.assert_allclose(again, l0, rtol=0, atol=1e-12)


def test_wireframe_snapshot_reads_the_tapes_back():
    from awetrim.aerostructural.wireframe import structural_geometry_io

    geometry = yaml.safe_load(
        (LEGACY_DIR / "struc_geometry_wireframe_powered_2019.yaml").read_text(encoding="utf-8")
    )
    first = structural_geometry_io.main(copy.deepcopy(geometry))
    power, (left, right) = first[4], first[5][:2]
    l0 = _actuate(first[10], power, left, right)
    snapshot = build_deformed_struc_geometry(
        geometry, np.asarray(first[0]), bridle_rest_lengths=l0[-len(first[8]):]
    )
    again = np.asarray(structural_geometry_io.main(snapshot.wireframe_dict())[10], dtype=float)
    tapes = [power, left, right]
    np.testing.assert_allclose(again[tapes], l0[tapes], rtol=0, atol=1e-12)
    np.testing.assert_allclose(again, l0, rtol=0, atol=1e-7)


def test_save_geometry_snapshot_writes_awesio(full_structure, tmp_path):
    from awetrim.aerostructural.case import SNAPSHOT_STRUC_GEOMETRY_FILENAME

    aero = {"wing_sections": {"data": []}}
    save_geometry_snapshot({"is_save_geometry_snapshots": True}, full_structure, aero, tmp_path)
    written = load_structure(tmp_path / SNAPSHOT_STRUC_GEOMETRY_FILENAME)
    np.testing.assert_allclose(written.positions, full_structure.positions, atol=1e-12)
    assert (tmp_path / "aero_geometry.yaml").exists()
    save_geometry_snapshot({}, full_structure, aero, tmp_path / "off")
    assert not (tmp_path / "off").exists()

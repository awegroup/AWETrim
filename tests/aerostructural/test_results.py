"""Unit tests for the pure helpers in awetrim.aerostructural.results.

Covers the sweep-value generator, the restart directory-candidate logic, the
result-root path builders, and the deformed-geometry rewriter. None of these
touch a solver or the filesystem (path objects are built, not created).
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


def test_build_deformed_struc_geometry_replaces_nodes_by_index():
    struc_geometry = {
        "bridle_point_node": [0.0, 0.0, 0.0],
        "wing_particles": {
            "data": [
                [0, 1.0, 1.0, 1.0],
                [1, 2.0, 2.0, 2.0],
            ]
        },
    }
    struc_nodes = np.array(
        [
            [10.0, 11.0, 12.0],  # node 0
            [20.0, 21.0, 22.0],  # node 1
        ]
    )
    out = build_deformed_struc_geometry(struc_geometry, struc_nodes)

    # Original is not mutated (deepcopy contract).
    assert struc_geometry["wing_particles"]["data"][0][1] == 1.0

    assert out["bridle_point_node"] == [10.0, 11.0, 12.0]
    assert out["wing_particles"]["data"][0][1:] == [10.0, 11.0, 12.0]
    assert out["wing_particles"]["data"][1][1:] == [20.0, 21.0, 22.0]
    # The node-id column is preserved.
    assert out["wing_particles"]["data"][1][0] == 1

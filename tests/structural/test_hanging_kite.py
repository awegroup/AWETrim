# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""The structural facts the hanging-kite validation rests on.

None of these are about the *answer*; they pin the things that would silently
rot and quietly invalidate it: the canopy grid recovered from a flat spring
list, the split between canopy and bridle, the length extractor, the load
vector, and the fact that the distributed reference model has no load path to
its anchor.

Skipped when the vendored data is absent, so a checkout without it still runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "structural"
sys.path.insert(0, str(SCRIPTS))

hanging_kite = pytest.importorskip("hanging_kite")

DATA = hanging_kite.DATA
pytestmark = pytest.mark.skipif(
    not (DATA / "hanging_test_initial.npz").exists(),
    reason="vendored hanging-test data not present",
)


@pytest.fixture(scope="module")
def reference():
    return hanging_kite.load_reference()


@pytest.fixture(scope="module")
def grid(reference):
    return hanging_kite.canopy_grid(reference.nodes)


# --------------------------------------------------------------------------
# Canopy grid recovery
# --------------------------------------------------------------------------


def test_grid_is_29_by_8(grid):
    assert grid.shape == (29, 8)


def test_grid_sections_run_leading_edge_to_trailing_edge(grid):
    assert grid[:, 0].tolist() == list(range(1, 58, 2))
    assert grid[:, -1].tolist() == list(range(2, 59, 2))


def test_grid_nodes_are_unique(grid):
    flat = grid.ravel()
    assert len(set(flat.tolist())) == flat.size


def test_every_canopy_spring_is_a_grid_edge(reference, grid):
    """The split is unambiguous, which is what makes the comparison clean.

    721 of the 767 springs are canopy, and *every* spring joining two grid
    nodes is a grid edge -- no spring is left straddling the classification.
    """
    is_canopy = hanging_kite.classify_springs(reference, grid)
    assert int(is_canopy.sum()) == 721
    assert int((~is_canopy).sum()) == 46


def test_classify_springs_rejects_a_broken_grid(reference, grid):
    """Corrupting the grid must raise, not silently mis-split the model."""
    broken = grid.copy()
    # Reorder one section's interior: the node set is unchanged, so every
    # canopy spring still joins two grid nodes, but the chordwise chain now
    # skips and those springs are no longer grid edges.
    broken[0, 2], broken[0, 5] = broken[0, 5], broken[0, 2]
    with pytest.raises(RuntimeError, match="not grid edges"):
        hanging_kite.classify_springs(reference, broken)


def test_triangulation_covers_every_quad(grid):
    triangles = hanging_kite.canopy_triangles(grid)
    rows, columns = grid.shape
    assert len(triangles) == 2 * (rows - 1) * (columns - 1) == 392
    assert set(triangles.ravel().tolist()) <= set(grid.ravel().tolist())


# --------------------------------------------------------------------------
# The measured quantities
# --------------------------------------------------------------------------


def test_extractor_reproduces_the_reference_table(reference):
    """Our twelve lengths must reproduce kite_fem's published table exactly.

    Run its own solved states through our extractor and compare against the
    numbers it published from them. This is what makes the scoring in the
    report a like-for-like comparison rather than two different measurements.
    """
    published = hanging_kite.kite_fem_lengths()
    for case in range(1, 11):
        positions = hanging_kite.kite_fem_state(case)
        lengths = hanging_kite.extract_lengths(positions)
        assert lengths == pytest.approx(published[case - 1], abs=1e-9), case


def test_extractor_on_the_as_built_shape(reference):
    lengths = hanging_kite.extract_lengths(reference.nodes)
    assert lengths.shape == (12,)
    assert lengths[9] == pytest.approx(8.3052, abs=1e-3)     # as-built span
    # Left/right symmetry of the as-built kite, in the extractor's own order.
    assert lengths[0] == pytest.approx(lengths[8], abs=1e-9)
    assert lengths[1] == pytest.approx(lengths[7], abs=1e-9)
    assert lengths[10] == pytest.approx(lengths[11], abs=1e-9)


def test_measured_and_reference_tables_line_up():
    measured = hanging_kite.measured_lengths()
    published = hanging_kite.kite_fem_lengths()
    assert measured.shape == published.shape == (10, 12)
    # The measurements are mirrored/averaged, so they are exactly symmetric.
    assert measured[:, 0] == pytest.approx(measured[:, 8])
    assert measured[:, 10] == pytest.approx(measured[:, 11])


def test_measured_span_closes_under_gravity():
    """The deformation this dataset is really about."""
    measured = hanging_kite.measured_lengths()
    assert measured[0][9] == pytest.approx(4.867, abs=1e-3)   # bare gravity
    assert measured[4][9] > 8.0                               # tips spread open


# --------------------------------------------------------------------------
# Loads
# --------------------------------------------------------------------------


def test_load_vector_matches_the_reference_loading(reference):
    """Verbatim reimplementation of Hanging_test.py::loading, 6-DOF strided."""
    masses = reference.masses
    for tip, point in ((0.0, 0.0), (0.0, 9.7), (2.0, 0.0), (5.0, 25.2)):
        expected = np.zeros(6 * reference.n_nodes)
        expected[2::6] = masses * 9.81
        expected[2 * 6 + 1] += tip * 9.81
        expected[58 * 6 + 1] += -tip * 9.81
        expected[29 * 6 + 2] += point * 9.81
        expected = expected.reshape(-1, 6)

        forces = hanging_kite.load_vector(reference, tip, point)
        assert forces == pytest.approx(expected[:, :3], abs=0.0)
        assert np.abs(expected[:, 3:]).max() == 0.0      # no applied moments


def test_total_weight(reference):
    assert reference.masses.sum() == pytest.approx(12.8592, abs=1e-4)
    forces = hanging_kite.load_vector(reference, 0.0, 0.0)
    assert forces[:, 2].sum() == pytest.approx(126.15, abs=1e-2)


# --------------------------------------------------------------------------
# The reference model's missing load path
# --------------------------------------------------------------------------


def test_anchor_node_is_unconnected_in_the_stored_model(reference):
    """Node 0 is flagged fixed but belongs to no element.

    This is the defect that makes the distributed model ill posed, and the
    reason a suspension has to be supplied. If upstream ever fixes it, this
    test should fail and the suspension handling be revisited.
    """
    assert 0 not in reference.springs
    assert 0 not in reference.pulleys
    assert 0 not in reference.beams


def test_orphan_nodes_carry_no_mass(reference):
    """Pinning them is therefore free of physical consequence."""
    for node in hanging_kite.UNWIRED_NODES:
        assert reference.masses[node] == 0.0


def test_bar_suspension_fixes_the_convergence_points(reference, grid):
    model, _, _ = hanging_kite.build_model(
        reference, grid, pressure=0.15, canopy_stiffness=5000.0,
    )
    for node in hanging_kite.BAR_NODES:
        assert node in model.fixed_translation_nodes
    half = 0.5 * hanging_kite.BAR_LENGTH
    for node in hanging_kite.BAR_NODES:
        assert abs(model.nodes[node, 1]) == pytest.approx(half)


def test_bar_nodes_are_not_canopy_or_beam_nodes(reference, grid):
    """Moving them to the bar half-span must not disturb the reference mesh."""
    on_grid = set(grid.ravel().tolist())
    on_beams = set(reference.beams.ravel().tolist())
    for node in hanging_kite.BAR_NODES:
        assert node not in on_grid
        assert node not in on_beams


# --------------------------------------------------------------------------
# Thesis bridle corrections
# --------------------------------------------------------------------------


def test_thesis_bridle_edits_only_what_it_claims(reference):
    corrected = hanging_kite.apply_thesis_bridle(reference)
    assert int((corrected.spring_rest != reference.spring_rest).sum()) == 6
    assert int((corrected.pulley_rest != reference.pulley_rest).sum()) == 4
    # Topology untouched.
    assert corrected.springs is reference.springs
    assert corrected.beams is reference.beams


def test_ai_is_kept_unless_asked_for(reference):
    """Table B.1 marks A_I as N/A; Figure B.1 draws it. Keeping it is default.

    Cutting it removes a front load path and drops the four centre-load cases
    into a folded local minimum, so the default follows the figure and the
    physics rather than the table.
    """
    kept = hanging_kite.apply_thesis_bridle(reference)
    cut = hanging_kite.apply_thesis_bridle(reference, remove_ai=True)
    assert int((kept.spring_stiffness == 0.0).sum()) == 0
    assert int((cut.spring_stiffness == 0.0).sum()) == 2


def test_thesis_bridle_swaps_bri_and_brii(reference):
    corrected = hanging_kite.apply_thesis_bridle(reference)
    was_low = np.isclose(reference.spring_rest, 2.0450)
    was_high = np.isclose(reference.spring_rest, 2.3330)
    assert corrected.spring_rest[was_low] == pytest.approx(2.3330)
    assert corrected.spring_rest[was_high] == pytest.approx(2.0450)


def test_thesis_bridle_leaves_br5_alone_by_default(reference):
    stored, corrected_value = hanging_kite.THESIS_BRIDLE["br5"]
    default = hanging_kite.apply_thesis_bridle(reference)
    opted_in = hanging_kite.apply_thesis_bridle(reference, include_br5=True)
    assert np.isclose(default.spring_rest, stored).sum() == 2
    assert np.isclose(opted_in.spring_rest, corrected_value).sum() == 2

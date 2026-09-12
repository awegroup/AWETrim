# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PSS/QSM coupled driver's bridle-line bookkeeping.

The bridle drag is carried by BOTH sides of the coupling -- the trim balances
it inside the VSM solve, the structure receives it as nodal loads -- so the two
have to agree on which structural nodes each segment hangs between. The driver
used to recover that by matching the VSM's built coordinates to the nearest
node, which only holds while the structure is still at its built shape.
"""

import numpy as np
import pytest

driver = pytest.importorskip(
    "awetrim.aerostructural.pss.aerostructural_coupled_solver_qsm",
    reason="needs the PSS and VSM solvers",
)

#: two segments, as ``parse_bridle_line_specs`` returns them
SPECS = [(0, 7, 0.004), (7, 12, 0.012)]


def test_pairs_come_from_the_specs_not_from_the_geometry():
    pairs = driver._bridle_node_pairs(SPECS, struc_nodes=None, body_aero=None)
    np.testing.assert_array_equal(pairs, [[0, 7], [7, 12]])


def test_pairs_do_not_move_when_the_structure_deforms():
    """The geometric match does, and that is the bug.

    On the LEI-V3 PSM geometry a converged shape moves nodes by up to 0.571 m;
    matching the BUILT segment coordinates against it snapped both ends of 2 of
    45 segments onto one node -- zero length, NaN force -- and re-paired 8 more.
    A continuation run (``starting_from_sim_subdir``) starts on such a shape.
    """
    built = driver._bridle_node_pairs(SPECS, struc_nodes=np.zeros((13, 3)), body_aero=None)
    deformed = driver._bridle_node_pairs(
        SPECS, struc_nodes=np.full((13, 3), 5.0), body_aero=None
    )
    np.testing.assert_array_equal(built, deformed)
    assert not any(i == j for i, j in deformed), "a segment collapsed onto one node"


def test_the_geometric_match_is_still_the_fallback_without_specs():
    class _Body:
        _bridle_line_system = [
            [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.004],
        ]

    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    pairs = driver._bridle_node_pairs(None, struc_nodes=nodes, body_aero=_Body())
    np.testing.assert_array_equal(pairs, [[0, 1]])

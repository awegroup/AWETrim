# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Mirror symmetry: pairing, the symmetry equalities, and mirror-consistent frames.

The frame test is the regression test for the unsteered LEI-V3's left-right
asymmetry: a beam network whose frames are transported from one tip, through
junctions where a member crossing the plane meets members beside it, is NOT
mirror-symmetric even though every node position is -- and ``mirror_frames``
makes it so, to roundoff.
"""

import numpy as np
import pytest

from awetrim.structural import (
    DofLayout,
    MinimumEnergySolver,
    PotentialEnergy,
    StructuralModel,
)
from awetrim.structural.elements import build_cable_elements
from awetrim.structural.elements.beam import BeamSection, build_beam_elements
from awetrim.structural.rotations import minimal_rotation, orthonormalize
from awetrim.structural.symmetry import (
    REFLECTION_Y,
    frame_mirror_mismatch,
    mirror_equalities,
    mirror_frames,
    mirror_partners,
)

M = REFLECTION_Y


def symmetric_vectors(vectors, partner, pseudo=False):
    """The mirror-symmetric part of a nodal vector field."""
    sign = -1.0 if pseudo else 1.0
    return 0.5 * (vectors + sign * vectors[partner] @ M.T)


# --------------------------------------------------------------------------
# Pairing and the equalities
# --------------------------------------------------------------------------


def test_mirror_partners_pair_every_node_and_reject_an_asymmetric_set():
    nodes = np.array([[0.0, 1.0, 0.0], [0.3, 0.0, 1.0], [0.0, -1.0, 0.0]])
    np.testing.assert_array_equal(mirror_partners(nodes), [2, 1, 0])
    nodes[2, 1] += 1e-6
    with pytest.raises(ValueError, match="not mirror-symmetric"):
        mirror_partners(nodes)


def test_equalities_hold_exactly_on_symmetric_configurations_and_only_there():
    # Two mirror pairs plus two nodes on the plane, one of each kind rotational.
    nodes = np.array([
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],      # pair, rotational
        [1.0, 2.0, 0.5], [1.0, -2.0, 0.5],      # pair
        [0.5, 0.0, 1.0],                        # on the plane, rotational
        [0.5, 0.0, -1.0],                       # on the plane
    ])
    partner = mirror_partners(nodes)
    layout = DofLayout(n_nodes=len(nodes), rotational_nodes=np.array([0, 1, 4]))
    equalities = mirror_equalities(layout, partner)

    # 3 per translation pair, 1 per plane node; 3 per rotation pair, 2 on the plane
    assert equalities.n_rows == 3 + 3 + 1 + 1 + 3 + 2
    matrix = equalities.dense(layout.n_dof)
    assert np.linalg.matrix_rank(matrix) == equalities.n_rows

    rng = np.random.default_rng(3)
    slots = np.array([layout.rotation_slot(int(partner[n])) for n in layout.rotational_nodes])
    positions = symmetric_vectors(rng.normal(size=nodes.shape), partner)
    increments = rng.normal(size=(3, 3))
    increments = 0.5 * (increments - increments[slots] @ M.T)
    symmetric = np.concatenate([positions.ravel(), increments.ravel()])
    assert np.abs(equalities.residual(symmetric)).max() < 1e-14
    np.testing.assert_allclose(matrix @ symmetric, equalities.residual(symmetric), atol=1e-14)

    assert np.abs(equalities.residual(rng.normal(size=layout.n_dof))).max() > 1e-3


def test_rows_on_pinned_dof_only_are_dropped():
    nodes = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    layout = DofLayout(n_nodes=3, rotational_nodes=np.array([], dtype=int))
    partner = mirror_partners(nodes)
    assert mirror_equalities(layout, partner).n_rows == 4
    assert mirror_equalities(layout, partner, pinned_nodes=[0, 1, 2]).n_rows == 0
    assert mirror_equalities(layout, partner, pinned_nodes=[0]).n_rows == 4


# --------------------------------------------------------------------------
# Frames: the LEI defect and its fix
# --------------------------------------------------------------------------


def lei_like_beam_tree(n_leading_edge=8, strut_at=(1, 2)):
    """An arched leading edge crossing y = 0, with a strut pair per ``strut_at``.

    Exactly mirror-symmetric in position. Frames are built the way the Billow
    adapter builds them: the leading edge's tangent runs along the chain (so
    the reflection REVERSES it), each strut owns its junction node with a
    tangent running leading edge to trailing edge (which the reflection
    KEEPS), and the roll is transported by minimal rotation over the tree from
    one tip.

    The struts stay off the two leading-edge nodes either side of the plane,
    as on the LEI-V3: an element crossing the plane between two strut-owned
    nodes would join frames whose ``d1`` the mirror convention reverses, a
    ~180 degree relative rotation where the Cayley map is singular.
    """
    angle = np.radians(np.linspace(70.0, -70.0, n_leading_edge))
    leading_edge = np.column_stack([0.3 * np.cos(angle) ** 2, 4.0 * np.sin(angle),
                                    4.0 * np.cos(angle)])
    nodes = [*leading_edge]
    elements = [[k, k + 1] for k in range(n_leading_edge - 1)]
    tangents = np.gradient(leading_edge, axis=0)
    tangents = list(tangents / np.linalg.norm(tangents, axis=1, keepdims=True))

    for k in [*strut_at, *(n_leading_edge - 1 - np.asarray(strut_at))]:
        previous = int(k)
        for j in (1, 2):
            nodes.append(leading_edge[k] + [0.6 * j, 0.0, -0.1 * j**2])
            tangents.append(np.zeros(3))
            elements.append([previous, len(nodes) - 1])
            previous = len(nodes) - 1
        chord = np.asarray(nodes[-1]) - leading_edge[k]
        chord /= np.linalg.norm(chord)
        tangents[k] = chord                     # the strut owns the junction
        tangents[-1] = tangents[-2] = chord
    nodes, tangents = np.asarray(nodes), np.asarray(tangents)

    neighbours = {}
    for a, b in elements:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    frames = np.tile(np.eye(3), (len(nodes), 1, 1))
    seed = np.cross(tangents[0], [0.0, 0.0, 1.0])
    seed /= np.linalg.norm(seed)
    frames[0] = np.column_stack([tangents[0], seed, np.cross(tangents[0], seed)])
    visited, queue = {0}, [0]
    while queue:
        node = queue.pop()
        for other in neighbours[node]:
            if other not in visited:
                frames[other] = orthonormalize(
                    minimal_rotation(tangents[node], tangents[other]) @ frames[node]
                )
                visited.add(other)
                queue.append(other)
    return nodes, np.asarray(elements), frames


def mirror_errors(nodes, elements, frames, seed=0):
    """Relative mirror errors of the internal forces and moments at a symmetric state."""
    section = BeamSection(ea=1e5, ga_2=4e4, ga_3=4e4, gj=30.0, ei_2=50.0, ei_3=50.0)
    beams = build_beam_elements(nodes, elements, section, frames)
    model = StructuralModel(nodes, [beams], node_frames=frames)
    layout = model.layout
    partner = mirror_partners(nodes)
    slots = np.array([layout.rotation_slot(int(partner[n])) for n in layout.rotational_nodes])

    rng = np.random.default_rng(seed)
    positions = nodes + symmetric_vectors(0.05 * rng.normal(size=nodes.shape), partner)
    increments = 0.1 * rng.normal(size=(layout.n_rotational_nodes, 3))
    increments = 0.5 * (increments - increments[slots] @ M.T)

    energy = PotentialEnergy(model)
    state = model.initial_state()
    parameters = energy.pack_parameters(state, model.element_sets, np.zeros_like(nodes),
                                        None, nodes)
    load = np.asarray(energy.internal_load(energy.pack_unknowns(positions, increments),
                                           parameters)).ravel()
    forces = load[: layout.n_translation_dof].reshape(-1, 3)
    moments = load[layout.n_translation_dof:].reshape(-1, 3)
    force_error = np.abs(forces[partner] - forces @ M.T).max() / np.abs(forces).max()
    moment_error = np.abs(moments[slots] + moments @ M.T).max() / np.abs(moments).max()
    mismatch = frame_mirror_mismatch(state.frames, layout, partner)
    return force_error, moment_error, float(mismatch.max())


def test_frames_transported_through_strut_junctions_break_mirror_symmetry():
    nodes, elements, frames = lei_like_beam_tree()
    force_error, moment_error, mismatch = mirror_errors(nodes, elements, frames)
    # Positions are exact, yet the far half arrives rolled...
    assert np.abs(nodes[mirror_partners(nodes)] - nodes @ M.T).max() < 1e-12
    assert np.degrees(mismatch) > 1.0
    # ... and the element energy sees it.
    assert moment_error > 1e-3


def test_mirror_frames_make_the_beam_energy_exactly_mirror_symmetric():
    nodes, elements, frames = lei_like_beam_tree()
    partner = mirror_partners(nodes)
    beam_nodes = np.unique(elements)
    mirrored = mirror_frames(frames, nodes, partner, beam_nodes)

    near = nodes[:, 1] > 0
    np.testing.assert_array_equal(mirrored[near], frames[near])   # near half untouched
    for frame in mirrored[beam_nodes]:
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.0)

    force_error, moment_error, mismatch = mirror_errors(nodes, elements, mirrored)
    assert mismatch < 1e-6
    assert force_error < 1e-11
    assert moment_error < 1e-11


def test_a_member_in_the_plane_cannot_share_the_leading_edge_signs():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])       # a strut ON the plane
    frames = np.tile(np.eye(3), (2, 1, 1))                     # d1 = x, in the plane
    with pytest.raises(ValueError, match="mirror plane"):
        mirror_frames(frames, nodes, mirror_partners(nodes), [0, 1])


# --------------------------------------------------------------------------
# Constrained solves
# --------------------------------------------------------------------------


def hanging_truss():
    """Two free knots hung from four pins by tension-only cables, mirror-symmetric."""
    nodes = np.array([
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],        # pins
        [1.0, 1.0, 0.0], [1.0, -1.0, 0.0],        # pins
        [0.5, 0.5, -1.0], [0.5, -0.5, -1.0],      # free knots
    ])
    connectivity = np.array([[0, 4], [2, 4], [1, 5], [3, 5], [4, 5]])
    lengths = np.linalg.norm(nodes[connectivity[:, 1]] - nodes[connectivity[:, 0]], axis=1)
    cables = build_cable_elements(connectivity, 0.99 * lengths, np.full(5, 1e4))
    return StructuralModel(nodes, [cables], fixed_translation_nodes=[0, 1, 2, 3])


def test_constrained_solve_matches_the_free_one_under_a_symmetric_load():
    model = hanging_truss()
    partner = mirror_partners(model.nodes)
    equalities = mirror_equalities(model.layout, partner, pinned_nodes=[0, 1, 2, 3])
    assert equalities.n_rows == 3

    forces = np.zeros((6, 3))
    forces[[4, 5], 2] = -20.0
    free = MinimumEnergySolver(model).solve(forces)
    constrained = MinimumEnergySolver(model, equalities=equalities).solve(forces)
    assert free.converged and constrained.converged
    assert constrained.residual_norm < 1e-6
    np.testing.assert_allclose(constrained.state.positions, free.state.positions, atol=1e-8)


def test_constraint_reaction_shows_up_as_residual_under_an_asymmetric_load():
    model = hanging_truss()
    partner = mirror_partners(model.nodes)
    equalities = mirror_equalities(model.layout, partner, pinned_nodes=[0, 1, 2, 3])

    forces = np.zeros((6, 3))
    forces[4, 2] = -20.0                           # only one side loaded
    free = MinimumEnergySolver(model).solve(forces)
    constrained = MinimumEnergySolver(model, equalities=equalities).solve(forces)
    assert free.residual_norm < 1e-6
    unknowns = np.asarray(constrained.state.positions).ravel()
    assert np.abs(equalities.residual(unknowns)).max() < 1e-9   # held symmetric
    assert constrained.residual_norm > 1.0                     # ... by a reaction
    assert not constrained.converged

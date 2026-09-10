# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""The assembler itself: DOF bookkeeping, gather maps, and the mapped kernels.

The whole scalability argument rests on evaluating each element kernel through
``casadi.Function.map`` over a gathered index matrix instead of looping in
Python. That machinery is easy to get subtly wrong -- a column-major reshape,
a rotational slot off by one -- and wrong in a way that still converges to a
plausible-looking shape. So the central test here rebuilds the same energy the
slow, obvious way and demands the two agree.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from awetrim.structural import DofLayout, MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import (
    BeamSection,
    build_beam_elements,
    build_cable_elements,
    build_membrane_elements,
    build_pulley_elements,
    initial_frames_from_polyline,
)
from awetrim.structural.energy import PotentialEnergy


def mixed_model(seed: int = 0) -> StructuralModel:
    """One model containing every element type at once, deliberately irregular.

    The beam runs over a scattered subset of the node numbering rather than a
    contiguous block, so a rotational-slot mistake cannot hide behind a tidy
    layout.
    """
    rng = np.random.default_rng(seed)
    spar_nodes = np.array([7, 2, 9, 4])  # deliberately out of order
    nodes = rng.normal(scale=0.4, size=(12, 3))
    nodes[spar_nodes] = np.column_stack(
        [np.linspace(0.0, 1.5, 4), np.zeros(4), np.linspace(0.0, 0.2, 4)]
    )

    frames = np.tile(np.eye(3), (len(nodes), 1, 1))
    frames[spar_nodes] = initial_frames_from_polyline(nodes[spar_nodes])
    spar = build_beam_elements(
        nodes,
        np.column_stack([spar_nodes[:-1], spar_nodes[1:]]),
        BeamSection.from_tube(0.05, 5e-4, 3.0e9, 1.1e9),
        frames,
        name="spar",
    )

    triangles = np.array([[0, 1, 3], [1, 5, 3], [5, 6, 8], [0, 10, 11]])
    canopy = build_membrane_elements(nodes, triangles, 3e-4, 5e8, 0.3, name="canopy")

    cables = build_cable_elements(
        [[0, 5], [3, 8], [11, 6]],
        np.linalg.norm(nodes[[0, 3, 11]] - nodes[[5, 8, 6]], axis=1) * 0.9,
        [2.0e4, 3.0e4, 1.5e4],
        name="cables",
    )
    pulleys = build_pulley_elements([[1, 10, 6]], [1.4], [2.5e4], name="pulleys")

    return StructuralModel(
        nodes,
        [spar, canopy, cables, pulleys],
        node_frames=frames,
        fixed_translation_nodes=[7],
        fixed_rotation_nodes=[7],
    )


def naive_energy(model: StructuralModel, unknowns, frames_flat) -> float:
    """Same total energy, assembled one element at a time in plain Python.

    Intentionally the slow implementation the assembler exists to avoid: it
    slices the DOF vector by hand for every element and calls the kernel
    directly, so it shares no gather, reshape or map code with the thing it is
    checking.
    """
    layout = model.layout
    total = 0.0
    for element_set in model.element_sets:
        kernel = element_set.kernel
        for element, connectivity in enumerate(element_set.connectivity):
            dof = []
            for node in connectivity:
                dof.extend(unknowns[3 * node: 3 * node + 3])
            for local in kernel.rotational_nodes:
                slot = layout.rotation_slot(connectivity[local])
                start = layout.n_translation_dof + 3 * slot
                dof.extend(unknowns[start: start + 3])

            frames = []
            for local in kernel.rotational_nodes:
                slot = layout.rotation_slot(connectivity[local])
                frames.extend(frames_flat[9 * slot: 9 * slot + 9])

            total += float(
                ca.DM(
                    kernel.energy(
                        ca.DM(dof),
                        ca.DM(frames) if frames else ca.DM(0, 1),
                        ca.DM(element_set.params[element]),
                    )
                )
            )
    return total


# --------------------------------------------------------------------------
# The assembler against the obvious implementation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mapped_assembly_matches_an_element_by_element_sum(seed):
    """The headline correctness test for the gather-and-map assembler."""
    model = mixed_model(seed)
    energy = PotentialEnergy(model, anchor_stiffness=0.0)

    rng = np.random.default_rng(seed + 100)
    positions = model.nodes + rng.normal(scale=0.15, size=model.nodes.shape)
    increments = rng.normal(scale=0.2, size=(model.layout.n_rotational_nodes, 3))
    unknowns = energy.pack_unknowns(positions, increments)
    frames_flat = np.asarray(model.node_frames)[model.layout.rotational_nodes].reshape(-1)

    parameters = energy.pack_parameters(
        model.initial_state(), model.element_sets,
        np.zeros_like(positions), None, positions,
    )
    mapped = float(energy.total_energy(unknowns, parameters))
    assert mapped == pytest.approx(naive_energy(model, unknowns, frames_flat), rel=1e-9)
    assert mapped > 0.0, "the perturbed configuration should store energy"


@pytest.mark.parametrize("seed", [0, 1])
def test_internal_load_is_minus_the_energy_gradient(seed):
    """``internal_load`` must be exactly ``-dU/dX``, checked by finite differences."""
    model = mixed_model(seed)
    energy = PotentialEnergy(model, anchor_stiffness=0.0)

    rng = np.random.default_rng(seed + 200)
    positions = model.nodes + rng.normal(scale=0.1, size=model.nodes.shape)
    increments = rng.normal(scale=0.1, size=(model.layout.n_rotational_nodes, 3))
    unknowns = energy.pack_unknowns(positions, increments)
    parameters = energy.pack_parameters(
        model.initial_state(), model.element_sets,
        np.zeros_like(positions), None, positions,
    )

    analytic = np.asarray(energy.internal_load(unknowns, parameters)).reshape(-1)
    step = 1e-7
    for dof in rng.choice(model.layout.n_dof, size=12, replace=False):
        forward, backward = unknowns.copy(), unknowns.copy()
        forward[dof] += step
        backward[dof] -= step
        gradient = (
            float(energy.total_energy(forward, parameters))
            - float(energy.total_energy(backward, parameters))
        ) / (2.0 * step)
        assert analytic[dof] == pytest.approx(-gradient, abs=1e-3 * max(1.0, abs(gradient)))


# --------------------------------------------------------------------------
# DOF layout
# --------------------------------------------------------------------------


def test_layout_allocates_rotations_only_where_a_kernel_needs_them():
    model = mixed_model()
    assert model.n_nodes == 12
    assert sorted(model.layout.rotational_nodes) == [2, 4, 7, 9]
    assert model.layout.n_dof == 3 * 12 + 3 * 4
    for node in (0, 1, 3, 5):
        assert model.layout.rotation_slot(node) == -1


def test_gather_maps_have_the_right_shape():
    model = mixed_model()
    spar = model.element_set("spar")
    canopy = model.element_set("canopy")

    assert model.layout.element_dof_indices(spar).shape == (12, spar.n_elements)
    assert model.layout.element_frame_indices(spar).shape == (18, spar.n_elements)
    assert model.layout.element_dof_indices(canopy).shape == (9, canopy.n_elements)
    assert model.layout.element_frame_indices(canopy).shape == (0, canopy.n_elements)


def test_layout_rejects_a_rotation_request_for_a_translation_only_node():
    layout = DofLayout(n_nodes=5, rotational_nodes=np.array([1, 3]))
    with pytest.raises(ValueError, match="no rotational DOF"):
        layout.rotation_dof(np.array([0]))


# --------------------------------------------------------------------------
# Parameters stay live
# --------------------------------------------------------------------------


def test_element_parameter_shape_is_validated():
    with pytest.raises(ValueError, match="parameter rows"):
        build_cable_elements([[0, 1]], [1.0, 2.0], [1.0, 2.0])


def test_solving_with_a_different_topology_is_refused():
    model = mixed_model()
    solver = MinimumEnergySolver(model)
    other = mixed_model()
    other = StructuralModel(
        np.vstack([other.nodes, [[0.0, 0.0, 5.0]]]),
        other.element_sets,
        np.vstack([other.node_frames, np.eye(3)[None]]),
    )
    with pytest.raises(ValueError, match="topology"):
        solver.solve(np.zeros((model.n_nodes, 3)), model=other)


def restrained_mixed_model() -> StructuralModel:
    """The mixed model with enough nodes pinned that an equilibrium exists."""
    loose = mixed_model()
    return StructuralModel(
        loose.nodes,
        loose.element_sets,
        loose.node_frames,
        fixed_translation_nodes=[7, 0, 5, 6, 11],
        fixed_rotation_nodes=[7],
    )


def test_whole_mixed_model_reaches_equilibrium():
    """Everything at once: beams, fabric, cables and a pulley in one solve."""
    model = restrained_mixed_model()
    rng = np.random.default_rng(7)
    forces = rng.normal(scale=2.0, size=(model.n_nodes, 3))

    solution = MinimumEnergySolver(model, tolerance=1e-9).solve(forces)
    assert solution.converged, solution.status
    assert solution.residual_norm < 1e-5


def test_an_unrestrained_mechanism_is_reported_as_a_failure():
    """A structure with free nodes no taut element reaches has no equilibrium.

    Tension-only cables and wrinkling fabric can both stop carrying load, so it
    is easy to build a model that is quietly a mechanism. The solver must say
    so through ``converged`` rather than return a plausible-looking shape: here
    the free nodes simply accelerate away under the applied load.
    """
    model = mixed_model()
    rng = np.random.default_rng(7)
    forces = rng.normal(scale=8.0, size=(model.n_nodes, 3))

    solution = MinimumEnergySolver(model, tolerance=1e-9).solve(forces)
    assert not solution.converged
    assert solution.residual_norm > 1.0
    assert np.abs(solution.state.positions).max() > 1e3  # it ran away

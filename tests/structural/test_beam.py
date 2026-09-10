# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Geometrically exact Timoshenko beam, on its own.

Checks the three things a finite-rotation beam has to get right before it is
worth putting inside an optimiser: it must be blind to rigid-body motion, it
must converge to the analytic Timoshenko solution, and it must not lock in
shear when the member is slender.
"""

from __future__ import annotations

import numpy as np
import pytest

from awetrim.structural import MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import (
    BeamSection,
    build_beam_elements,
    initial_frames_from_polyline,
)
from awetrim.structural.rotations import cayley, cayley_vector

MODULUS = 70.0e9  # Pa
SHEAR_MODULUS = 26.0e9  # Pa
DIAMETER = 0.02  # m
WALL = 1.0e-3  # m


def tube_section() -> BeamSection:
    return BeamSection.from_tube(DIAMETER, WALL, MODULUS, SHEAR_MODULUS)


def tube_properties() -> tuple[float, float]:
    """``(EI, kappa G A)`` of the tube used throughout this file."""
    radius = 0.5 * DIAMETER
    second_moment = np.pi * radius**3 * WALL
    area = 2.0 * np.pi * radius * WALL
    return MODULUS * second_moment, 0.5 * SHEAR_MODULUS * area


def straight_beam(n_elements: int, length: float = 1.0, clamped: bool = True):
    nodes = np.column_stack(
        [
            np.linspace(0.0, length, n_elements + 1),
            np.zeros(n_elements + 1),
            np.zeros(n_elements + 1),
        ]
    )
    return beam_model(nodes, clamped=clamped)


def beam_model(nodes: np.ndarray, clamped: bool = True) -> StructuralModel:
    frames = initial_frames_from_polyline(nodes)
    connectivity = np.column_stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))])
    beams = build_beam_elements(nodes, connectivity, tube_section(), frames, name="spar")
    return StructuralModel(
        nodes,
        [beams],
        node_frames=frames,
        fixed_translation_nodes=[0] if clamped else [],
        fixed_rotation_nodes=[0] if clamped else [],
    )


def strain_energy(model: StructuralModel, positions, increments=None) -> float:
    energy = MinimumEnergySolver(model).energy
    state = model.initial_state()
    parameters = energy.pack_parameters(
        state, model.element_sets, np.zeros_like(positions), None, positions
    )
    unknowns = energy.pack_unknowns(positions, increments)
    return float(energy.total_energy(unknowns, parameters))


# --------------------------------------------------------------------------
# Invariance
# --------------------------------------------------------------------------


def test_reference_configuration_is_stress_free():
    model = straight_beam(6)
    assert strain_energy(model, model.nodes) == pytest.approx(0.0, abs=1e-12)


def test_curved_beam_is_stress_free_as_built():
    """A pre-curved member must not be pre-stressed by its own curvature.

    The reference strains are taken from the built geometry, so an arched
    leading-edge tube starts at zero energy rather than trying to straighten
    itself out.
    """
    angle = np.linspace(0.0, 0.8, 9)
    nodes = np.column_stack([np.sin(angle), np.zeros_like(angle), 1.0 - np.cos(angle)])
    model = beam_model(nodes)
    assert strain_energy(model, model.nodes) == pytest.approx(0.0, abs=1e-12)


def test_rigid_body_motion_costs_no_energy():
    """Translate and rotate the whole beam, rotating its frames to match.

    This is the objectivity property. It is what stops a slack member from
    being spuriously stiffened just because it swung somewhere else.
    """
    model = straight_beam(6, clamped=False)
    increment = np.array([0.3, -0.45, 0.2])
    rotation = cayley(increment, np)

    moved = model.nodes @ rotation.T + np.array([5.0, -3.0, 2.0])
    increments = np.tile(increment, (model.layout.n_rotational_nodes, 1))
    assert strain_energy(model, moved, increments) == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------
# Against analytic beam theory
# --------------------------------------------------------------------------


def tip_loaded_cantilever(n_elements: int, load: float = 1.0, length: float = 1.0):
    model = straight_beam(n_elements, length=length)
    forces = np.zeros((model.n_nodes, 3))
    forces[-1, 2] = load
    solution = MinimumEnergySolver(model).solve(forces)
    assert solution.converged, solution.status
    return solution.state.positions[-1, 2]


def test_cantilever_converges_to_timoshenko_at_second_order():
    """Tip deflection must approach ``PL^3/3EI + PL/kGA`` like ``O(h^2)``."""
    load, length = 1.0, 1.0
    bending, shear = tube_properties()
    exact = load * length**3 / (3.0 * bending) + load * length / shear

    errors = []
    for n_elements in (5, 10, 20, 40):
        tip = tip_loaded_cantilever(n_elements, load=load, length=length)
        errors.append(abs(tip / exact - 1.0))

    assert errors[-1] < 1e-3, f"not converged: {errors}"
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(ratio > 3.4 for ratio in ratios), f"not second order: {ratios}"


def test_slender_beam_does_not_shear_lock():
    """Locking would show up as a beam that gets stiffer as it gets thinner.

    One-point integration is the cure; without it the same refinement would
    drive the tip deflection towards zero instead of towards beam theory.
    """
    load = 0.01
    bending, shear = tube_properties()
    for length in (1.0, 4.0, 16.0):
        exact = load * length**3 / (3.0 * bending) + load * length / shear
        tip = tip_loaded_cantilever(24, load=load, length=length)
        assert tip / exact == pytest.approx(1.0, abs=2e-3), f"{length=} locked"


def test_large_deflection_is_stiffer_than_linear_theory():
    """Geometric nonlinearity: the load arm shortens as the beam bends over."""
    bending, _ = tube_properties()
    load = 800.0
    tip = tip_loaded_cantilever(24, load=load)
    linear = load / (3.0 * bending)
    assert tip < 0.85 * linear
    assert tip > 0.3  # it really did bend a long way


def test_applied_torque_twists_by_gj():
    """Torsion about the beam axis: twist rate must be ``T / GJ``."""
    n_elements, length, torque = 12, 1.0, 5.0
    model = straight_beam(n_elements, length=length)

    moments = np.zeros((model.layout.n_rotational_nodes, 3))
    moments[-1] = [torque, 0.0, 0.0]  # about d1, the beam axis
    solution = MinimumEnergySolver(model).solve(
        np.zeros((model.n_nodes, 3)), moments=moments
    )
    assert solution.converged, solution.status

    # Recover the total twist from the tip frame relative to the root frame,
    # via the library Rodrigues vector rather than by rederiving it here.
    relative = solution.state.frames[-1] @ solution.state.frames[0].T
    twist = 2.0 * np.arctan(0.5 * np.linalg.norm(cayley_vector(relative, np)))

    section = tube_section()
    assert twist == pytest.approx(torque * length / section.gj, rel=2e-3)


# --------------------------------------------------------------------------
# Degrees of freedom
# --------------------------------------------------------------------------


def test_only_beam_nodes_carry_rotational_dof():
    model = straight_beam(4)
    assert model.layout.n_rotational_nodes == 5
    assert model.layout.n_dof == 3 * 5 + 3 * 5


def test_a_model_with_beams_demands_frames():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    frames = initial_frames_from_polyline(nodes)
    beams = build_beam_elements(nodes, [[0, 1]], tube_section(), frames)
    with pytest.raises(ValueError, match="node_frames"):
        StructuralModel(nodes, [beams])


def test_parallel_transported_frames_are_orthonormal_and_tangent():
    angle = np.linspace(0.0, 1.2, 7)
    points = np.column_stack([np.sin(angle), 0.3 * angle, 1.0 - np.cos(angle)])
    frames = initial_frames_from_polyline(points)

    for index, frame in enumerate(frames):
        assert frame @ frame.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.0)
        if index == 0:
            tangent = points[1] - points[0]
            director = frame[:, 0]
            assert director @ tangent / np.linalg.norm(tangent) == pytest.approx(1.0)

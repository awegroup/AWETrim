# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Classical large-rotation benchmarks, kept small enough to run in CI.

References here come from outside this repository -- an exact circle, published
tip displacements -- so they cannot drift with the implementation. The roll-up
tests in particular guard a real bug: an applied moment is work-conjugate to the
exponential-map rotation vector, not to the Rodrigues vector the DOF carry, and
using the latter made the answer depend on how many load steps were taken.
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

MODULUS = 1.0e7
SECOND_MOMENT = 1.0e-6
AREA = 1.0e-2  # slender: axial and shear flexibility stay below 0.1%
BENDING = MODULUS * SECOND_MOMENT


def slender_section() -> BeamSection:
    shear_modulus = MODULUS / 2.6
    return BeamSection(
        ea=MODULUS * AREA,
        ga_2=shear_modulus * AREA,
        ga_3=shear_modulus * AREA,
        gj=shear_modulus * 2.0 * SECOND_MOMENT,
        ei_2=BENDING,
        ei_3=BENDING,
    )


def straight_cantilever(n_elements: int, length: float = 1.0) -> StructuralModel:
    nodes = np.column_stack(
        [
            np.linspace(0.0, length, n_elements + 1),
            np.zeros(n_elements + 1),
            np.zeros(n_elements + 1),
        ]
    )
    frames = initial_frames_from_polyline(nodes)
    connectivity = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    beams = build_beam_elements(nodes, connectivity, slender_section(), frames, name="beam")
    return StructuralModel(
        nodes,
        [beams],
        node_frames=frames,
        fixed_translation_nodes=[0],
        fixed_rotation_nodes=[0],
    )


def roll_up(model: StructuralModel, fraction: float = 1.0, load_steps: int = 4):
    """Apply ``fraction * 2 pi EI / L`` at the tip, walking the load up."""
    length = model.nodes[-1, 0]
    target = fraction * 2.0 * np.pi * BENDING / length
    solver = MinimumEnergySolver(model, tolerance=1e-9, max_iterations=2000)

    state, solution = None, None
    for step in range(1, load_steps + 1):
        moments = np.zeros((model.layout.n_rotational_nodes, 3))
        moments[-1] = [0.0, target * step / load_steps, 0.0]
        solution = solver.solve(
            np.zeros((model.n_nodes, 3)), moments=moments, state=state
        )
        state = solution.state
    return solution


def circle_error(model: StructuralModel, solution, fraction: float = 1.0) -> float:
    """Tip distance from the exact circular arc, normalised by beam length."""
    length = model.nodes[-1, 0]
    radius = length / (2.0 * np.pi * fraction)
    angle = 2.0 * np.pi * fraction
    # A moment about +y curves a beam lying along +x towards -z.
    exact = np.array([radius * np.sin(angle), 0.0, -radius * (1.0 - np.cos(angle))])
    return float(np.linalg.norm(solution.state.positions[-1] - exact)) / length


# --------------------------------------------------------------------------
# Roll-up: the exact answer is a closed circle
# --------------------------------------------------------------------------


def test_rollup_converges_to_the_exact_circle_at_second_order():
    errors = [circle_error(model := straight_cantilever(n), roll_up(model))
              for n in (10, 20, 40)]
    assert errors[-1] < 1e-2, f"not converged: {errors}"
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(ratio > 3.0 for ratio in ratios), f"not second order: {ratios}"


@pytest.mark.parametrize("load_steps", [4, 8, 16])
def test_rollup_is_independent_of_the_load_stepping(load_steps):
    """Statics is path independent -- the answer must not depend on the ramp.

    It did: ``-M . psi`` is only first-order work-conjugate, so every absorbed
    frame update left a bias and more load steps meant more accumulated error
    (4 steps 8.1e-3 L, 8 steps 2.1e-1 L). Using the rotation vector in the work
    term removed it.
    """
    model = straight_cantilever(40)
    reference = circle_error(model, roll_up(model, load_steps=4))
    assert circle_error(model, roll_up(model, load_steps=load_steps)) == pytest.approx(
        reference, rel=1e-3
    )


def test_half_rollup_matches_its_arc():
    model = straight_cantilever(40)
    assert circle_error(model, roll_up(model, fraction=0.5), fraction=0.5) < 5e-3


# --------------------------------------------------------------------------
# Bathe and Bolourchi 45-degree bend
# --------------------------------------------------------------------------


def test_bend45_matches_the_published_tip_displacement():
    """Out-of-plane load on a 45-degree curved cantilever, F = 600.

    The arc here starts at the origin with its tangent along +x and sweeps
    towards +y, which swaps the two in-plane components relative to the usual
    tabulation. Reference: Simo & Vu-Quoc (1986), whose values sit within ~1% of
    Bathe & Bolourchi (1979) and Crisfield (1990).
    """
    radius, n_elements = 100.0, 32
    angle = np.linspace(0.0, np.pi / 4.0, n_elements + 1)
    nodes = np.column_stack(
        [radius * np.sin(angle), radius * (1.0 - np.cos(angle)), np.zeros_like(angle)]
    )
    frames = initial_frames_from_polyline(nodes)
    connectivity = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    section = BeamSection(
        ea=1.0e7 * 1.0,
        ga_2=5.0e6 * 1.0,
        ga_3=5.0e6 * 1.0,
        gj=5.0e6 * 2.0 / 12.0,
        ei_2=1.0e7 / 12.0,
        ei_3=1.0e7 / 12.0,
    )
    beams = build_beam_elements(nodes, connectivity, section, frames, name="bend")
    model = StructuralModel(
        nodes,
        [beams],
        node_frames=frames,
        fixed_translation_nodes=[0],
        fixed_rotation_nodes=[0],
    )

    forces = np.zeros((model.n_nodes, 3))
    forces[-1, 2] = 600.0
    solution = MinimumEnergySolver(model, tolerance=1e-9).solve(forces)
    assert solution.converged, solution.status

    displacement = solution.state.positions[-1] - nodes[-1]
    expected = np.array([-23.48, -13.50, 53.37])
    assert displacement == pytest.approx(expected, rel=0.02)
    assert np.linalg.norm(displacement) == pytest.approx(
        np.linalg.norm(expected), rel=0.01
    )

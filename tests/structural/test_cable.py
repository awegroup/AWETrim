# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Cables and frictionless pulleys.

These reproduce the spring law the PSS particle system and the kite_fem spring
element both use, so this file is the parity check: same force law, same
tension-only cut, same shared-stretch pulley.
"""

from __future__ import annotations

import numpy as np
import pytest

from awetrim.structural import MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import build_cable_elements, build_pulley_elements

STIFFNESS = 1.0e5  # N/m


def two_node_model(rest_length=1.0, tension_only=True) -> StructuralModel:
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cables = build_cable_elements(
        [[0, 1]], [rest_length], [STIFFNESS], tension_only=tension_only
    )
    return StructuralModel(nodes, [cables], fixed_translation_nodes=[0])


def test_stretched_cable_matches_hookes_law():
    """``f = k (l - l0)``: the same law the PSS spring uses."""
    model = two_node_model()
    load = 250.0
    forces = np.array([[0.0, 0.0, 0.0], [load, 0.0, 0.0]])
    solution = MinimumEnergySolver(model).solve(forces)

    assert solution.converged, solution.status
    extension = solution.state.positions[1, 0] - 1.0
    assert extension == pytest.approx(load / STIFFNESS, rel=1e-4)
    assert solution.strain_energy == pytest.approx(
        0.5 * STIFFNESS * extension**2, rel=1e-4
    )


def test_reaction_at_the_fixed_node_carries_the_load():
    model = two_node_model()
    load = 250.0
    forces = np.array([[0.0, 0.0, 0.0], [load, 0.0, 0.0]])
    solution = MinimumEnergySolver(model).solve(forces)
    assert solution.internal_forces[0, 0] == pytest.approx(load, rel=1e-4)


def test_tension_only_cable_goes_slack_under_compression():
    """Pushing the end towards the anchor must cost nothing."""
    model = two_node_model()
    energy = MinimumEnergySolver(model).energy
    state = model.initial_state()

    squashed = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    parameters = energy.pack_parameters(
        state, model.element_sets, np.zeros_like(squashed), None, squashed
    )
    assert float(
        energy.total_energy(energy.pack_unknowns(squashed), parameters)
    ) == pytest.approx(0.0, abs=1e-12)


def test_two_way_cable_resists_compression():
    model = two_node_model(tension_only=False)
    energy = MinimumEnergySolver(model).energy
    state = model.initial_state()

    squashed = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    parameters = energy.pack_parameters(
        state, model.element_sets, np.zeros_like(squashed), None, squashed
    )
    stored = float(energy.total_energy(energy.pack_unknowns(squashed), parameters))
    assert stored == pytest.approx(0.5 * STIFFNESS * 0.3**2, rel=1e-9)


def test_hanging_node_finds_the_catenary_corner():
    """Two anchored cables holding one loaded node: a symmetric V."""
    nodes = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cables = build_cable_elements([[0, 1], [1, 2]], [1.0, 1.0], [STIFFNESS] * 2)
    model = StructuralModel(nodes, [cables], fixed_translation_nodes=[0, 2])

    load = 400.0
    forces = np.zeros((3, 3))
    forces[1, 2] = -load
    solution = MinimumEnergySolver(model).solve(forces)
    assert solution.converged, solution.status

    sag = solution.state.positions[1]
    assert sag[2] < 0.0
    assert sag[0] == pytest.approx(0.0, abs=1e-9)  # symmetric

    # Vertical equilibrium: 2 T sin(theta) = load, with T = k (l - l0).
    length = np.linalg.norm(sag - solution.state.positions[0])
    tension = STIFFNESS * (length - 1.0)
    sine = abs(sag[2]) / length
    assert 2.0 * tension * sine == pytest.approx(load, rel=1e-4)


def test_pulley_equalises_tension_either_side():
    """A frictionless sheave: the pulley node slides until both arms pull equally.

    The single shared stretch is what enforces it -- exactly the construction
    the PSS and kite_fem pulley elements use.
    """
    nodes = np.array([[-1.0, 0.0, 0.0], [0.1, -0.5, 0.0], [1.0, 0.0, 0.0]])
    total_rest_length = 2.4
    pulleys = build_pulley_elements([[0, 1, 2]], [total_rest_length], [STIFFNESS])
    model = StructuralModel(nodes, [pulleys], fixed_translation_nodes=[0, 2])

    forces = np.zeros((3, 3))
    forces[1, 1] = -300.0
    solution = MinimumEnergySolver(model).solve(forces)
    assert solution.converged, solution.status

    positions = solution.state.positions
    arm_a = np.linalg.norm(positions[1] - positions[0])
    arm_b = np.linalg.norm(positions[2] - positions[1])
    assert arm_a == pytest.approx(arm_b, rel=1e-4), "pulley did not centre itself"

    # Both anchors must feel the same rope tension.
    assert np.linalg.norm(solution.internal_forces[0]) == pytest.approx(
        np.linalg.norm(solution.internal_forces[2]), rel=1e-4
    )


def test_pulley_rest_length_spans_both_arms():
    """The stored rest length is the whole rope, not one arm.

    Worth pinning down: the PSS reader splits ``l0`` across the two arms while
    kite_fem stores the total on each, so the convention has to be explicit.
    """
    nodes = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pulleys = build_pulley_elements([[0, 1, 2]], [2.0], [STIFFNESS])
    model = StructuralModel(nodes, [pulleys])

    energy = MinimumEnergySolver(model).energy
    state = model.initial_state()
    parameters = energy.pack_parameters(
        state, model.element_sets, np.zeros_like(nodes), None, nodes
    )
    # Both arms are 1.0 long, so the rope is exactly at its rest length.
    assert float(
        energy.total_energy(energy.pack_unknowns(nodes), parameters)
    ) == pytest.approx(0.0, abs=1e-12)


def test_actuating_a_rest_length_reuses_the_compiled_solver():
    """Shortening a tape must not need a rebuild -- rest length is a parameter."""
    model = two_node_model()
    solver = MinimumEnergySolver(model)
    forces = np.array([[0.0, 0.0, 0.0], [250.0, 0.0, 0.0]])

    slack = solver.solve(forces)
    shortened = model.replaced(
        model.element_set("cables").with_param_column("rest_length", [0.9])
    )
    pulled = solver.solve(forces, model=shortened)

    assert pulled.converged
    assert pulled.state.positions[1, 0] == pytest.approx(
        slack.state.positions[1, 0] - 0.1, rel=1e-4
    )

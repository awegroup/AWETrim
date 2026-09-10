# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Fabric on its own: the CST membrane element and the canopy it makes.

Deliberately isolated from the cable and beam tests. The membrane is the part
of the model with no analogue in the current PSS spring network, it is the part
that will carry the most elements, and its wrinkling branches are the only
place the energy stops being smooth -- so it gets its own file, its own
analytic checks, and its own scaling check.
"""

from __future__ import annotations

import numpy as np
import pytest

from awetrim.structural import MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import build_membrane_elements, membrane_reference

MODULUS = 5.0e8  # Pa, a stiff technical fabric
POISSON = 0.3
THICKNESS = 3.0e-4  # m
REGULARIZATION = 1e-12  # MembraneKernel default


def wrinkling_floor(area: float) -> float:
    """Residual energy the principal-strain regularisation leaves at zero strain.

    The relaxed law floors the principal strain at ``sqrt(regularization)``, so
    a perfectly unstressed panel keeps ``1/2 E t A * regularization`` joules.
    Negligible against a loaded panel, but not exactly zero, so the
    stress-free tests compare against it rather than against 0.
    """
    return 0.5 * MODULUS * REGULARIZATION * area * THICKNESS


def unit_triangle() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def single_triangle_model(nodes: np.ndarray, **kwargs) -> StructuralModel:
    """One triangle with the pure relaxed law.

    ``slack_stiffness_ratio`` defaults to zero here so these tests exercise the
    tension-field branches exactly; the numerical blend the solver ships with
    is checked separately in :func:`test_slack_stiffness_blend_leaves_taut_alone`.
    """
    kwargs.setdefault("slack_stiffness_ratio", 0.0)
    canopy = build_membrane_elements(
        nodes, [[0, 1, 2]], THICKNESS, MODULUS, POISSON, **kwargs
    )
    return StructuralModel(nodes, [canopy])


def strain_energy(model: StructuralModel, positions: np.ndarray) -> float:
    """Strain energy of an arbitrary configuration, with no solve involved."""
    energy = MinimumEnergySolver(model).energy
    state = model.initial_state()
    state.positions = np.asarray(positions, dtype=float)
    parameters = energy.pack_parameters(
        state, model.element_sets, np.zeros_like(positions), None, positions
    )
    return float(energy.total_energy(energy.pack_unknowns(positions), parameters))


# --------------------------------------------------------------------------
# Reference geometry
# --------------------------------------------------------------------------


def test_reference_area_and_inverse_edge_matrix():
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    area, d0_inverse = membrane_reference(nodes, [0, 1, 2])
    assert area == pytest.approx(3.0)
    # D0 is [[2, 0], [0, 3]] in the local in-plane basis, so its inverse is diagonal.
    assert d0_inverse == pytest.approx(np.diag([0.5, 1.0 / 3.0]), abs=1e-12)


def test_degenerate_triangle_is_rejected():
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="degenerate"):
        membrane_reference(collinear, [0, 1, 2])


# --------------------------------------------------------------------------
# Frame invariance -- the property that lets fabric billow
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wrinkling", [False, True])
def test_reference_configuration_is_stress_free(wrinkling):
    nodes = unit_triangle()
    model = single_triangle_model(nodes, wrinkling=wrinkling)
    floor = wrinkling_floor(0.5) if wrinkling else 0.0
    assert strain_energy(model, nodes) == pytest.approx(floor, abs=1e-12)


@pytest.mark.parametrize("wrinkling", [False, True])
def test_rigid_body_motion_costs_no_energy(wrinkling):
    """A panel that translates and rotates out of plane must stay unstressed.

    This is what a finite-strain (Green-Lagrange) measure buys over a
    small-strain one: the canopy can rotate to any attitude and billow far out
    of its reference plane without the element inventing energy.
    """
    nodes = unit_triangle()
    model = single_triangle_model(nodes, wrinkling=wrinkling)

    angle = 0.9
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle) * np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    orthonormal, _ = np.linalg.qr(rotation)
    moved = nodes @ orthonormal.T + np.array([3.0, -2.0, 7.0])
    floor = wrinkling_floor(0.5) if wrinkling else 0.0
    assert strain_energy(model, moved) == pytest.approx(floor, abs=1e-12)


# --------------------------------------------------------------------------
# Constitutive law against closed form
# --------------------------------------------------------------------------


def test_equibiaxial_stretch_matches_analytic_svk():
    """Uniform biaxial stretch: both principal strains equal, membrane taut."""
    nodes = unit_triangle()
    model = single_triangle_model(nodes, wrinkling=False)

    stretch = 1.02
    strain = 0.5 * (stretch**2 - 1.0)  # Green-Lagrange, equal in both directions
    expected_density = (MODULUS / (2.0 * (1.0 - POISSON**2))) * (
        2.0 * strain**2 + 2.0 * POISSON * strain**2
    )
    expected = expected_density * 0.5 * THICKNESS  # area of the unit triangle is 1/2

    assert strain_energy(model, nodes * stretch) == pytest.approx(expected, rel=1e-9)


def test_wrinkling_and_taut_laws_agree_while_taut():
    """The relaxed energy must reduce to plain SVK wherever nothing wrinkles."""
    nodes = unit_triangle()
    stretched = nodes * 1.02
    taut = strain_energy(single_triangle_model(nodes, wrinkling=False), stretched)
    relaxed = strain_energy(single_triangle_model(nodes, wrinkling=True), stretched)
    assert relaxed == pytest.approx(taut, rel=1e-6)


def test_wrinkled_branch_is_uniaxial():
    """Stretch along x, squash along y: the fabric wrinkles instead of resisting.

    Past the wrinkling threshold the relaxed energy is the uniaxial
    ``1/2 E t A e1^2`` and carries no trace of the lateral compression, which
    is exactly the statement that fabric cannot take compression.
    """
    nodes = unit_triangle()
    model = single_triangle_model(nodes, wrinkling=True)

    stretch_x, squash_y = 1.05, 0.90
    deformed = nodes * np.array([stretch_x, squash_y, 1.0])
    principal_1 = 0.5 * (stretch_x**2 - 1.0)
    expected = 0.5 * MODULUS * principal_1**2 * 0.5 * THICKNESS

    assert strain_energy(model, deformed) == pytest.approx(expected, rel=1e-6)


def test_wrinkled_energy_is_blind_to_further_lateral_slack():
    """Once wrinkled, squashing the panel further must not change the energy."""
    nodes = unit_triangle()
    model = single_triangle_model(nodes, wrinkling=True)
    mild = strain_energy(model, nodes * np.array([1.05, 0.90, 1.0]))
    severe = strain_energy(model, nodes * np.array([1.05, 0.70, 1.0]))
    assert severe == pytest.approx(mild, rel=1e-6)


def test_slack_panel_stores_no_energy():
    """Compressed in both directions the panel is limp, not buckled."""
    nodes = unit_triangle()
    model = single_triangle_model(nodes, wrinkling=True)
    assert strain_energy(model, nodes * 0.8) == 0.0


def test_slack_stiffness_blend_leaves_taut_alone():
    """The numerical slack stiffness must not touch a taut panel.

    ``U = (1-r) U_relaxed + r U_taut`` coincides with the exact relaxed law
    wherever the fabric is taut, and adds exactly ``r U_taut`` where it is not.
    """
    nodes = unit_triangle()
    ratio = 1e-4
    stretched = nodes * 1.02
    squashed = nodes * 0.8

    exact = single_triangle_model(nodes, slack_stiffness_ratio=0.0)
    blended = single_triangle_model(nodes, slack_stiffness_ratio=ratio)
    unrelaxed = single_triangle_model(nodes, wrinkling=False)

    assert strain_energy(blended, stretched) == pytest.approx(
        strain_energy(exact, stretched), rel=1e-9
    )
    assert strain_energy(blended, squashed) == pytest.approx(
        ratio * strain_energy(unrelaxed, squashed), rel=1e-9
    )


def test_unrelaxed_law_would_fight_the_slack_panel():
    """Contrast: without wrinkling the same limp panel carries large energy.

    This is the failure the relaxed energy exists to remove -- an unrelaxed SVK
    canopy resists compression and buckles into whatever mode the mesh offers.
    """
    nodes = unit_triangle()
    unrelaxed = single_triangle_model(nodes, wrinkling=False)
    assert strain_energy(unrelaxed, nodes * 0.8) > 1.0


# --------------------------------------------------------------------------
# Equilibrium of a real fabric patch
# --------------------------------------------------------------------------


def square_canopy(n_side: int, size: float = 1.0):
    """Flat square membrane on an ``n_side x n_side`` grid, edges clamped."""
    axis = np.linspace(0.0, size, n_side)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    nodes = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]
    )

    def index(i, j):
        return i * n_side + j

    triangles = []
    for i in range(n_side - 1):
        for j in range(n_side - 1):
            triangles.append([index(i, j), index(i + 1, j), index(i + 1, j + 1)])
            triangles.append([index(i, j), index(i + 1, j + 1), index(i, j + 1)])

    on_edge = (
        (grid_x.ravel() == 0.0)
        | (grid_y.ravel() == 0.0)
        | (grid_x.ravel() == size)
        | (grid_y.ravel() == size)
    )
    return nodes, np.array(triangles), np.flatnonzero(on_edge)


def inflate(n_side: int, total_load: float = 200.0, slack: float = 0.0, **kwargs):
    """Push a clamped flat square out of plane with a uniform normal load.

    ``slack`` oversizes the *reference* fabric relative to the frame it is
    clamped into, so the panel starts loose and has to wrinkle -- the state a
    real canopy spends most of its time in.
    """
    nodes, triangles, edge_nodes = square_canopy(n_side)
    canopy = build_membrane_elements(
        nodes * (1.0 + slack), triangles, THICKNESS, MODULUS, POISSON, **kwargs
    )
    model = StructuralModel(nodes, [canopy], fixed_translation_nodes=edge_nodes)

    forces = np.zeros_like(nodes)
    interior = np.setdiff1d(np.arange(len(nodes)), edge_nodes)
    forces[interior, 2] = total_load / len(interior)

    solver = MinimumEnergySolver(model, tolerance=1e-10)
    return solver.solve(forces), model


def test_flat_canopy_inflates_from_a_singular_start():
    """A flat membrane has zero out-of-plane stiffness in its reference state.

    Newton on the residual has nothing to invert here; energy minimisation with
    IPOPT walks out of the singular point on its own. This is the load case
    that decides whether fabric is usable at all.
    """
    solution, _ = inflate(9)
    assert solution.converged, solution.status

    deflection = solution.state.positions[:, 2]
    assert deflection.max() > 0.02, "the canopy did not lift out of plane"
    assert solution.residual_norm < 1e-6
    assert solution.strain_energy > 0.0


def test_inflated_canopy_is_in_force_balance():
    """Internal and applied loads must cancel node by node at the free nodes."""
    solution, _ = inflate(7)
    free = solution.residual_forces[solution.free_node_mask]
    assert np.abs(free).max() < 1e-6


def test_inflated_shape_converges_under_mesh_refinement():
    """Centre deflection must settle as the fabric mesh is refined."""
    deflections = []
    for n_side in (7, 11, 15):
        solution, _ = inflate(n_side)
        assert solution.converged, f"{n_side=} {solution.status}"
        centre = (n_side * n_side - 1) // 2
        deflections.append(solution.state.positions[centre, 2])

    coarse_step = abs(deflections[1] - deflections[0])
    fine_step = abs(deflections[2] - deflections[1])
    assert fine_step < coarse_step, f"not converging: {deflections}"
    assert fine_step / abs(deflections[2]) < 0.1


def test_taut_canopy_does_not_care_whether_wrinkling_is_on():
    """Under biaxial tension the relaxed law must reproduce the unrelaxed one."""
    relaxed, _ = inflate(9, wrinkling=True)
    unrelaxed, _ = inflate(9, wrinkling=False)
    assert relaxed.converged and unrelaxed.converged
    assert relaxed.state.positions[:, 2].max() == pytest.approx(
        unrelaxed.state.positions[:, 2].max(), rel=1e-3
    )


def test_loose_canopy_wrinkles_and_is_softer():
    """A panel clamped into a frame smaller than itself must go slack, not buckle.

    With the relaxed energy the surplus fabric costs nothing and the panel
    billows freely. Without it the same surplus is resisted as compression, so
    the unrelaxed canopy comes out artificially stiff -- the mesh-dependent
    buckling that tension-field theory exists to remove.
    """
    relaxed, _ = inflate(9, slack=0.06, wrinkling=True)
    unrelaxed, _ = inflate(9, slack=0.06, wrinkling=False)
    assert relaxed.converged, relaxed.status
    assert unrelaxed.converged, unrelaxed.status

    loose = relaxed.state.positions[:, 2].max()
    stiff = unrelaxed.state.positions[:, 2].max()
    assert loose > stiff, f"relaxed {loose:.4f} should billow past unrelaxed {stiff:.4f}"
    assert relaxed.strain_energy < unrelaxed.strain_energy


# --------------------------------------------------------------------------
# Scale -- the reason the assembler maps instead of looping
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_few_thousand_triangles_still_build_and_solve():
    """Canopy-sized fabric: build the graph and solve it, not just assemble it."""
    n_side = 33  # 2048 triangles, 1089 nodes, 3267 DOF
    solution, model = inflate(n_side)
    assert model.element_set("canopy").n_elements == 2 * (n_side - 1) ** 2
    assert solution.converged, solution.status
    assert solution.residual_norm < 1e-5

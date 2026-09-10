# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""The ASKITE inflatable-tube fits, ported from secant stiffnesses to an energy.

Two levels of check:

* the constitutive level -- the energy really is the integral of the fitted
  moment, and the conversion from the tip-load fit back to a tip-load curve is
  self-consistent;
* the element level -- a tube under a pure end moment takes a constant-curvature
  arc, and the curvature it settles at must be the one the fit prescribes. That
  test is exact regardless of how far the tube bends, which is what makes it the
  right instrument here.
"""

from __future__ import annotations

import numpy as np
import pytest

from awetrim.structural import MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import (
    InflatableTubeLaw,
    build_inflatable_beam_elements,
    inflatable_beam_state,
    initial_frames_from_polyline,
)
from awetrim.structural.elements.inflatable import (
    BENDING_COEFFICIENTS,
    CURVATURE_PER_DEFLECTION,
)

DIAMETER = 0.16  # m, the tube used in kite_fem/examples/FEM_beam_verification.py
PRESSURE = 0.3  # bar
LENGTH = 1.0  # m, the calibration length of the bending fit

# Axial and shear stiffness are not covered by the fits; a stiff thin wall keeps
# them from contaminating the bending and torsion checks.
AXIAL_STIFFNESS = 1.0e8
SHEAR_STIFFNESS = 1.0e8


def law() -> InflatableTubeLaw:
    return InflatableTubeLaw.from_fit(DIAMETER, PRESSURE)


def tube(n_elements: int = 20, tube_law: InflatableTubeLaw | None = None):
    tube_law = tube_law or law()
    nodes = np.column_stack(
        [
            np.linspace(0.0, LENGTH, n_elements + 1),
            np.zeros(n_elements + 1),
            np.zeros(n_elements + 1),
        ]
    )
    frames = initial_frames_from_polyline(nodes)
    connectivity = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    tubes = build_inflatable_beam_elements(
        nodes,
        connectivity,
        tube_law,
        frames,
        axial_stiffness=AXIAL_STIFFNESS,
        shear_stiffness=SHEAR_STIFFNESS,
        name="tubes",
    )
    model = StructuralModel(
        nodes,
        [tubes],
        node_frames=frames,
        fixed_translation_nodes=[0],
        fixed_rotation_nodes=[0],
    )
    return model, tubes, tube_law


def apply_end_moment(model, axis: int, magnitude: float, load_steps: int = 4):
    """Ramp a tip moment about a global axis, warm-starting each step."""
    solver = MinimumEnergySolver(model, tolerance=1e-10, max_iterations=3000)
    state, solution = None, None
    for step in range(1, load_steps + 1):
        moments = np.zeros((model.layout.n_rotational_nodes, 3))
        moments[-1, axis] = magnitude * step / load_steps
        solution = solver.solve(
            np.zeros((model.n_nodes, 3)), moments=moments, state=state
        )
        state = solution.state
    return solution


# --------------------------------------------------------------------------
# The constitutive law itself
# --------------------------------------------------------------------------


def test_bending_energy_integrates_the_fitted_moment():
    """``dW/dkappa`` must reproduce ``M(kappa)`` -- the whole point of the port."""
    tube_law = law()
    step = 1e-7
    for curvature in (0.01, 0.05, 0.1, 0.2, 0.4, 0.8):
        gradient = (
            tube_law.bending_energy_density(curvature + step)
            - tube_law.bending_energy_density(curvature - step)
        ) / (2.0 * step)
        assert gradient == pytest.approx(tube_law.moment(curvature), rel=1e-6)


def test_torsion_energy_integrates_the_fitted_torque():
    tube_law = law()
    step = 1e-7
    for twist in (0.01, 0.1, 0.5, 1.0, 2.0):
        gradient = (
            tube_law.torsion_energy_density(twist + step)
            - tube_law.torsion_energy_density(twist - step)
        ) / (2.0 * step)
        assert gradient == pytest.approx(tube_law.torque(twist), rel=1e-6)


def test_energies_vanish_and_are_even_at_zero_strain():
    tube_law = law()
    assert tube_law.bending_energy_density(0.0) == pytest.approx(0.0, abs=1e-15)
    assert tube_law.torsion_energy_density(0.0) == pytest.approx(0.0, abs=1e-15)
    for value in (0.05, 0.3):
        assert tube_law.bending_energy_density(-value) == pytest.approx(
            tube_law.bending_energy_density(value), rel=1e-12
        )
        assert tube_law.torsion_energy_density(-value) == pytest.approx(
            tube_law.torsion_energy_density(value), rel=1e-12
        )


def test_small_strain_limits_are_the_initial_stiffnesses():
    tube_law = law()
    # The next term in M(kappa) is -kappa/(2 k0) relative, so the strain has to
    # sit well below the saturation scale for the linear limit to show cleanly.
    tiny = 1e-9
    assert tube_law.moment(tiny) == pytest.approx(
        tube_law.bending_stiffness * tiny, rel=1e-6
    )
    assert tube_law.bending_energy_density(tiny) == pytest.approx(
        0.5 * tube_law.bending_stiffness * tiny**2, rel=1e-5
    )
    assert tube_law.torque(tiny) == pytest.approx(
        tube_law.torsion_stiffness * tiny, rel=1e-6
    )


def test_moment_saturates_at_the_limit_moment():
    tube_law = law()
    assert tube_law.moment(50.0 * tube_law.curvature_scale) == pytest.approx(
        tube_law.moment_max, rel=1e-9
    )
    assert tube_law.moment(tube_law.curvature_scale) < tube_law.moment_max


def test_conversion_round_trips_the_original_tip_load_fit():
    """``M(kappa)`` mapped back through ``kappa = 3v``, ``M = P`` must be the fit.

    The published form is ``P(v) = D (1 - exp(-(N/D) v))``. Recovering it from
    ``EI_0`` and ``M_max`` confirms the tip-load-to-curvature conversion, which
    is the one step of the port that could silently rescale everything.
    """
    tube_law = law()
    radius = 0.5 * DIAMETER
    b = BENDING_COEFFICIENTS
    limit_load = (b["C1"] * radius + b["C2"]) * PRESSURE**2 + (
        b["C3"] * radius**3 + b["C4"]
    )
    initial_slope = (b["C5"] * radius**5 + b["C6"]) * PRESSURE + (
        b["C7"] * radius + b["C8"]
    )

    for deflection in (0.005, 0.02, 0.05, 0.09):
        published = limit_load * (
            1.0 - np.exp(-(initial_slope / limit_load) * deflection)
        )
        ported = tube_law.moment(CURVATURE_PER_DEFLECTION * deflection / LENGTH)
        assert ported == pytest.approx(published, rel=1e-12)


def test_pressure_stiffens_and_strengthens_the_tube():
    soft = InflatableTubeLaw.from_fit(DIAMETER, 0.3)
    hard = InflatableTubeLaw.from_fit(DIAMETER, 0.5)
    assert hard.bending_stiffness > soft.bending_stiffness
    assert hard.moment_max > soft.moment_max
    assert hard.curvature_collapse > soft.curvature_collapse


def test_non_positive_pressure_is_rejected():
    with pytest.raises(ValueError, match="pressure must be positive"):
        InflatableTubeLaw.from_fit(DIAMETER, 0.0)


# --------------------------------------------------------------------------
# The element
# --------------------------------------------------------------------------


def test_reference_configuration_is_stress_free():
    model, tubes, _ = tube()
    energy = MinimumEnergySolver(model).energy
    state = model.initial_state()
    parameters = energy.pack_parameters(
        state, model.element_sets, np.zeros((model.n_nodes, 3)), None, model.nodes
    )
    assert float(
        energy.total_energy(energy.pack_unknowns(model.nodes), parameters)
    ) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 0.9])
def test_pure_bending_settles_at_the_curvature_the_fit_prescribes(fraction):
    """A tip moment gives a constant-curvature arc; the fit sets which one.

    Exact at any deflection, because the constant-curvature arc is the exact
    solution of pure end-moment bending -- no linear-cantilever assumption is
    involved, unlike in the tip-load form the fit was published in.
    """
    model, tubes, tube_law = tube(n_elements=20)
    applied = fraction * tube_law.moment_max
    solution = apply_end_moment(model, axis=1, magnitude=applied)
    assert solution.converged, solution.status

    diagnostic = inflatable_beam_state(model, solution.state, tubes, tube_law)
    curvature = diagnostic["curvature"]
    assert curvature.std() / curvature.mean() < 5e-3, "arc is not of constant curvature"
    assert tube_law.moment(curvature.mean()) == pytest.approx(applied, rel=5e-3)


def test_pure_torsion_settles_at_the_twist_the_fit_prescribes():
    model, tubes, tube_law = tube(n_elements=12)
    applied = 0.6 * tube_law.torque_max
    solution = apply_end_moment(model, axis=0, magnitude=applied)
    assert solution.converged, solution.status

    diagnostic = inflatable_beam_state(model, solution.state, tubes, tube_law)
    twist = diagnostic["twist_rate"]
    assert twist.std() / abs(twist.mean()) < 5e-3
    assert tube_law.torque(twist.mean()) == pytest.approx(applied, rel=5e-3)


def test_tube_is_softer_than_a_linear_beam_of_the_same_initial_stiffness():
    """Saturation is the physics being ported: the tube must give way.

    A linear ``EI_0`` beam would reach ``kappa = M / EI_0``; the fitted tube has
    to bend further than that once the moment approaches its limit.
    """
    model, tubes, tube_law = tube(n_elements=20)
    applied = 0.8 * tube_law.moment_max
    solution = apply_end_moment(model, axis=1, magnitude=applied)
    assert solution.converged, solution.status

    curvature = inflatable_beam_state(
        model, solution.state, tubes, tube_law
    )["curvature"].mean()
    assert curvature > 1.5 * applied / tube_law.bending_stiffness


def test_collapse_is_reported_not_enforced():
    """Past the collapse curvature the solve still works; the report flags it.

    A dropping post-collapse moment would make the energy decrease with
    curvature, i.e. an unbounded mechanism. The law stops at saturation and the
    diagnostic says which elements have left the range it was calibrated over.
    """
    model, tubes, tube_law = tube(n_elements=20)
    solution = apply_end_moment(
        model, axis=1, magnitude=0.97 * tube_law.moment_max, load_steps=8
    )
    assert solution.converged, solution.status

    diagnostic = inflatable_beam_state(model, solution.state, tubes, tube_law)
    assert diagnostic["utilisation"].max() > 1.0
    assert diagnostic["collapsed"].any()

    gentle = apply_end_moment(model, axis=1, magnitude=0.3 * tube_law.moment_max)
    assert not inflatable_beam_state(model, gentle.state, tubes, tube_law)[
        "collapsed"
    ].any()


def test_mismatched_law_count_is_rejected():
    model, tubes, tube_law = tube(n_elements=4)
    with pytest.raises(ValueError, match="tube elements but"):
        inflatable_beam_state(model, model.initial_state(), tubes, [tube_law] * 3)

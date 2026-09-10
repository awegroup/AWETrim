# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Standard large-rotation beam benchmarks, run in both structural models.

Three canonical cases with references that do not depend on either code:

* ``elastica``  -- tip-loaded cantilever. Reference is the exact Euler elastica,
  obtained here by solving ``EI theta'' = -P cos theta`` as a boundary-value
  problem to 1e-12, so it is independent of any beam element.
* ``rollup``    -- cantilever under an end moment ``M = 2 pi EI / L``. The exact
  answer is a closed circle: the tip lands exactly on the root. Error is
  measured against the exact arc, so there is nothing to look up. ``structural``
  walks the moment up in four steps because its rotation chart cannot turn a
  node most of the way round in one solve; ``kite_fem`` is run single-shot
  because its solver restarts from the initial configuration on every call and
  applies its own internal step limiting instead.
* ``bend45``    -- Bathe & Bolourchi 45-degree curved cantilever with an
  out-of-plane tip load. The classic 3-D finite-rotation benchmark; published
  tip displacements differ by ~1% between authors, so the reference is quoted
  as a band.

Both models solve the same discretisation with matched section properties:

* ``structural`` -- this repository's minimum-energy NLP (IPOPT).
* ``kite_fem``   -- ``pyfe3d`` BeamC (Timoshenko, Luo 2008) driven through
  ``kite_fem.FEMStructure``, with the inflatable constitutive update disabled
  and linear properties set instead, so the two share one material model.

Sections are deliberately slender: the classical references are inextensible
and shear-rigid, so axial and shear flexibility are kept below ~0.1% and cannot
be mistaken for discretisation error.

Run from the project root::

    python scripts/structural/run_validation_benchmarks.py
    python scripts/structural/run_validation_benchmarks.py --case rollup --show
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

from awetrim.plotting import PALETTE
from awetrim.structural import MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import (
    BeamSection,
    InflatableTubeLaw,
    build_beam_elements,
    build_inflatable_beam_elements,
    inflatable_beam_state,
    initial_frames_from_polyline,
)

OUTPUT_DIR = Path("results/structural/validation")

COLOURS = {"structural": PALETTE["Vermillion"], "kite_fem": PALETTE["Blue"]}
MARKERS = {"structural": "o", "kite_fem": "s"}


# ---------------------------------------------------------------------------
# Model drivers
# ---------------------------------------------------------------------------


def solve_structural(nodes, section, load_node, force=None, moment=None,
                     tolerance=1e-9, load_steps=1):
    """Solve with this repository's minimum-energy NLP.

    ``load_steps`` walks the load up, each step warm-started from the previous
    configuration. Forces do not need it. Applied *moments* do: the rotation
    DOF are Rodrigues vectors, whose chart is singular at a rotation of pi, so a
    node cannot be asked to turn most of the way round inside a single solve.
    """
    frames = initial_frames_from_polyline(nodes)
    connectivity = np.column_stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))])
    beams = build_beam_elements(nodes, connectivity, section, frames, name="beam")
    model = StructuralModel(
        nodes,
        [beams],
        node_frames=frames,
        fixed_translation_nodes=[0],
        fixed_rotation_nodes=[0],
    )

    forces = np.zeros((len(nodes), 3))
    if force is not None:
        forces[load_node] = force
    moments = None
    if moment is not None:
        moments = np.zeros((model.layout.n_rotational_nodes, 3))
        moments[model.layout.rotation_slot(load_node)] = moment

    solver = MinimumEnergySolver(model, tolerance=tolerance, max_iterations=3000)
    started = time.perf_counter()
    state, iterations, solution = None, 0, None
    for step in range(1, load_steps + 1):
        fraction = step / load_steps
        solution = solver.solve(
            fraction * forces,
            moments=None if moments is None else fraction * moments,
            state=state,
        )
        state = solution.state
        iterations += solution.iterations
    elapsed = time.perf_counter() - started
    return solution.state.positions, elapsed, solution.converged, iterations


def solve_kite_fem(nodes, section, load_node, force=None, moment=None,
                   tolerance=1e-6, max_iterations=3000, i_stiffness=0.0):
    """Solve the same problem with kite_fem / pyfe3d BeamC.

    ``FEM_structure`` always installs the inflatable constitutive law, so the
    per-element update is disabled and linear properties are written in its
    place. ``BeamProp.Ay``/``Az`` are first moments of area (zero on a
    centroidal axis), and pyfe3d applies no shear correction internally, so
    ``kappa`` is folded into ``G`` exactly as its documentation prescribes.

    ``i_stiffness`` is set to zero rather than the ``FEM_structure.solve``
    default of 25. That default adds ``25 I`` [N/m] to the tangent matrix on
    every DOF, which is tuned for kite-scale bridle stiffnesses; on the slender
    benchmark beams here ``EI/L^3`` is 10 N/m, so the default regularisation is
    larger than the structure and the solve stalls (2000 iterations, residual
    stuck at 20 N). With it off the same problem converges in 38 iterations.
    Reported results use the setting that works, not the shipped default.
    """
    from kite_fem.FEMStructure import FEM_structure

    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    initial_conditions = [
        [np.asarray(point, dtype=float), np.zeros(3), 1.0, index == 0]
        for index, point in enumerate(nodes)
    ]
    beam_matrix = [
        [index, index + 1, 0.1, 0.3, float(length)]
        for index, length in enumerate(lengths)
    ]
    structure = FEM_structure(
        initial_conditions=initial_conditions, beam_matrix=beam_matrix
    )

    area = section.ea / section.youngs_modulus
    second_moment = section.ei_2 / section.youngs_modulus
    for element, length in zip(structure.beam_elements, lengths):
        element.set_beam_properties(
            section.youngs_modulus, area, second_moment, float(length)
        )
        element.prop.G = section.shear_modulus
        element.prop.J = 2.0 * second_moment
        element.update_inflatable_beam_properties = lambda: None

    external = np.zeros(structure.N)
    if force is not None:
        external[6 * load_node: 6 * load_node + 3] = force
    if moment is not None:
        external[6 * load_node + 3: 6 * load_node + 6] = moment

    started = time.perf_counter()
    converged, _ = structure.solve(
        external,
        max_iterations=max_iterations,
        tolerance=tolerance,
        convergence_criteria="residual",
        I_stiffness=i_stiffness,
        print_info=False,
    )
    elapsed = time.perf_counter() - started
    positions = np.asarray(structure.coords_current, dtype=float).reshape(-1, 3)
    return positions, elapsed, bool(converged), len(structure.iteration_history)


class MatchedSection(BeamSection):
    """A :class:`BeamSection` that remembers the moduli it was built from."""


def matched_section(youngs_modulus, shear_modulus, area, second_moment,
                    shear_correction=1.0):
    """Section with matching properties for both codes.

    ``shear_correction`` multiplies ``G`` in the shear terms of both models.
    The classical references are shear-rigid, so the benchmarks below keep the
    section slender rather than relying on this.
    """
    section = MatchedSection(
        ea=youngs_modulus * area,
        ga_2=shear_correction * shear_modulus * area,
        ga_3=shear_correction * shear_modulus * area,
        gj=shear_modulus * 2.0 * second_moment,
        ei_2=youngs_modulus * second_moment,
        ei_3=youngs_modulus * second_moment,
    )
    section.youngs_modulus = youngs_modulus
    section.shear_modulus = shear_correction * shear_modulus
    return section


SOLVERS = {"structural": solve_structural, "kite_fem": solve_kite_fem}


# ---------------------------------------------------------------------------
# Reference: the exact Euler elastica
# ---------------------------------------------------------------------------


def elastica_reference(alphas, samples: int = 400, continuation_step: float = 0.25):
    """Exact tip positions of a tip-loaded cantilever, normalised by length.

    Solves ``theta''(s) = -alpha cos theta(s)`` on ``s in [0, 1]`` with
    ``theta(0) = 0`` and ``theta'(1) = 0``, where ``alpha = P L^2 / EI``. The
    result is the inextensible, shear-rigid elastica -- no beam element
    involved.

    The BVP is walked up in ``alpha`` by continuation, each solve seeded from
    the previous one: a cold small-deflection guess stops converging around
    ``alpha = 4``, where the tip has already turned through more than a radian.
    ``tol=1e-10`` is the tightest the collocation reaches inside the node budget,
    which is five orders below the discretisation error being measured.
    """
    mesh = np.linspace(0.0, 1.0, samples)
    fine = np.linspace(0.0, 1.0, 20001)
    guess = np.vstack([np.zeros_like(mesh), np.zeros_like(mesh)])

    targets = np.asarray(alphas, dtype=float)
    ladder = np.unique(
        np.concatenate(
            [np.arange(continuation_step, targets.max() + continuation_step,
                       continuation_step), targets]
        )
    )

    tips = {}
    solution = None
    for alpha in ladder:
        solution = solve_bvp(
            lambda s, y, a=alpha: np.vstack([y[1], -a * np.cos(y[0])]),
            lambda ya, yb: np.array([ya[0], yb[1]]),
            mesh if solution is None else solution.x,
            guess if solution is None else solution.y,
            tol=1e-10,
            max_nodes=200000,
        )
        if not solution.success:
            raise RuntimeError(f"elastica reference failed at alpha={alpha}")
        angle = solution.sol(fine)[0]
        tips[float(alpha)] = (
            float(np.trapezoid(np.cos(angle), fine)),
            float(np.trapezoid(np.sin(angle), fine)),
        )

    return np.array([tips[float(alpha)] for alpha in targets])


# ---------------------------------------------------------------------------
# Case 1 -- elastica
# ---------------------------------------------------------------------------


def elastica_case(n_elements=40, length=1.0):
    """Slender cantilever: axial and shear flexibility below 0.1%."""
    modulus, second_moment = 1.0e7, 1.0e-6
    area = 1.0e-2  # slenderness ~ 1000, so EA and GA are effectively rigid
    section = matched_section(modulus, modulus / 2.6, area, second_moment)
    nodes = np.column_stack(
        [np.linspace(0.0, length, n_elements + 1), np.zeros(n_elements + 1),
         np.zeros(n_elements + 1)]
    )
    return nodes, section, modulus * second_moment


def case_elastica():
    alphas = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    nodes, section, bending = elastica_case()
    length = nodes[-1, 0]

    reference = elastica_reference(alphas)
    results = {name: {"tip": [], "time": [], "converged": []} for name in SOLVERS}

    for name, solver in SOLVERS.items():
        for alpha in alphas:
            load = alpha * bending / length**2
            try:
                positions, elapsed, converged, _ = solver(
                    nodes, section, len(nodes) - 1, force=[0.0, 0.0, load]
                )
            except Exception as error:  # noqa: BLE001 - report, do not hide
                print(f"  {name} alpha={alpha}: {type(error).__name__}: {error}")
                results[name]["tip"].append([np.nan, np.nan])
                results[name]["time"].append(np.nan)
                results[name]["converged"].append(False)
                continue
            results[name]["tip"].append(
                [positions[-1, 0] / length, positions[-1, 2] / length]
            )
            results[name]["time"].append(elapsed)
            results[name]["converged"].append(converged)
            print(
                f"  {name:11s} alpha={alpha:5.1f}  tip=({positions[-1, 0]/length:.4f}, "
                f"{positions[-1, 2]/length:.4f})  exact=({reference[len(results[name]['tip'])-1][0]:.4f}, "
                f"{reference[len(results[name]['tip'])-1][1]:.4f})  "
                f"{elapsed:6.3f} s  converged={converged}"
            )

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    left.plot(reference[:, 0], reference[:, 1], color=PALETTE["Black"],
              linestyle="-", linewidth=2, label="exact elastica", zorder=1)
    for name in SOLVERS:
        tip = np.array(results[name]["tip"])
        left.plot(tip[:, 0], tip[:, 1], color=COLOURS[name], marker=MARKERS[name],
                  linestyle="none", markersize=7, label=name, zorder=2)
    left.set_xlabel("tip x / L (-)")
    left.set_ylabel("tip z / L (-)")
    left.set_aspect("equal")
    left.legend(frameon=False)
    left.grid(alpha=0.3)
    left.set_title("Tip locus, 40 elements, alpha up to 10", fontsize=10)

    for name in SOLVERS:
        tip = np.array(results[name]["tip"])
        error = np.linalg.norm(tip - reference, axis=1)
        right.semilogy(alphas, np.maximum(error, 1e-16), color=COLOURS[name],
                       marker=MARKERS[name], label=name)
    right.set_xlabel(r"$\alpha = PL^2/EI$ (-)")
    right.set_ylabel("tip position error / L (-)")
    right.legend(frameon=False)
    right.grid(alpha=0.3, which="both")
    right.set_title("Error against the exact elastica", fontsize=10)

    figure.suptitle("Euler elastica: large-deflection tip-loaded cantilever", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


# ---------------------------------------------------------------------------
# Case 2 -- roll-up into a closed circle
# ---------------------------------------------------------------------------


def case_rollup():
    """``M = 2 pi EI / L`` must roll the cantilever into an exact closed circle."""
    fractions = np.linspace(0.1, 1.0, 10)
    element_counts = (10, 20, 40)
    nodes_by_count = {}
    results = {name: {} for name in SOLVERS}

    for n_elements in element_counts:
        nodes, section, bending = elastica_case(n_elements=n_elements)
        nodes_by_count[n_elements] = (nodes, section, bending)
        for name, solver in SOLVERS.items():
            gaps, times, shapes = [], [], {}
            for fraction in fractions:
                moment = fraction * 2.0 * np.pi * bending / nodes[-1, 0]
                try:
                    extra = {"load_steps": 4} if name == "structural" else {}
                    positions, elapsed, converged, _ = solver(
                        nodes, section, len(nodes) - 1, moment=[0.0, moment, 0.0],
                        **extra,
                    )
                except Exception as error:  # noqa: BLE001
                    print(f"  {name} n={n_elements} f={fraction:.1f}: "
                          f"{type(error).__name__}: {error}")
                    gaps.append(np.nan)
                    times.append(np.nan)
                    continue
                # Exact solution: a circular arc of radius L/(2 pi fraction).
                radius = nodes[-1, 0] / (2.0 * np.pi * fraction)
                angle = 2.0 * np.pi * fraction
                # A moment about +y curves a beam lying along +x towards -z.
                exact_tip = np.array(
                    [radius * np.sin(angle), 0.0, -radius * (1.0 - np.cos(angle))]
                )
                gaps.append(
                    float(np.linalg.norm(positions[-1] - exact_tip)) / nodes[-1, 0]
                )
                times.append(elapsed)
                shapes[round(float(fraction), 2)] = positions
            results[name][n_elements] = {"gap": gaps, "time": times, "shape": shapes}
            print(f"  {name:11s} n={n_elements:3d}  full-circle gap = "
                  f"{gaps[-1]:.3e} L   mean {np.nanmean(times):.3f} s/solve")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    finest = element_counts[-1]
    for name in SOLVERS:
        shapes = results[name][finest]["shape"]
        for fraction in (0.3, 0.6, 1.0):
            shape = shapes.get(fraction)
            if shape is None:
                continue
            left.plot(shape[:, 0], shape[:, 2], color=COLOURS[name],
                      linestyle="-" if name == "structural" else "--", linewidth=1.6,
                      label=name if fraction == 1.0 else None)
    circle = np.linspace(0.0, 2.0 * np.pi, 200)
    radius = nodes_by_count[finest][0][-1, 0] / (2.0 * np.pi)
    left.plot(radius * np.sin(circle), -radius * (1.0 - np.cos(circle)),
              color=PALETTE["Black"], linestyle=":", linewidth=2, label="exact circle")
    left.set_aspect("equal")
    left.set_xlabel("x (m)")
    left.set_ylabel("z (m)")
    left.legend(frameon=False, fontsize=9)
    left.grid(alpha=0.3)
    left.set_title(f"Roll-up at 30, 60 and 100% of $2\\pi EI/L$ ({finest} elements)",
                   fontsize=10)

    for name in SOLVERS:
        for n_elements, style in zip(element_counts, ["-", "--", ":"]):
            right.semilogy(
                fractions,
                np.maximum(results[name][n_elements]["gap"], 1e-16),
                color=COLOURS[name], linestyle=style,
                marker=MARKERS[name] if n_elements == finest else None,
                markersize=5,
                label=f"{name}, {n_elements} el" if n_elements == finest else None,
            )
    right.set_xlabel(r"applied moment / $(2\pi EI/L)$ (-)")
    right.set_ylabel("tip error / L (-)")
    right.legend(frameon=False, fontsize=9)
    right.grid(alpha=0.3, which="both")
    right.set_title("Distance from the exact circular arc\n(solid/dashed/dotted = "
                    f"{element_counts[0]}/{element_counts[1]}/{element_counts[2]} elements)",
                    fontsize=10)

    figure.suptitle("Roll-up under an end moment: the exact answer is a closed circle",
                    fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


# ---------------------------------------------------------------------------
# Case 3 -- Bathe and Bolourchi 45-degree bend
# ---------------------------------------------------------------------------

#: Published tip DISPLACEMENTS for the 45-degree bend at F = 600, radius 100.
#:
#: The literature reports displacement, not position, and orders the in-plane
#: pair according to how the arc is laid out. The arc here starts at the origin
#: with its tangent along +x and sweeps towards +y, which swaps the two in-plane
#: components relative to the usual tabulation, so they are stored swapped to
#: match this geometry. The out-of-plane component (53.4) and the magnitude are
#: convention independent and are what the comparison keys on.
BEND45_REFERENCES = {
    "Bathe & Bolourchi (1979)": (-23.5, -13.4, 53.4),
    "Simo & Vu-Quoc (1986)": (-23.48, -13.50, 53.37),
    "Crisfield (1990)": (-23.5, -13.6, 53.4),
}


def bend45_geometry(n_elements: int, radius: float = 100.0):
    angle = np.linspace(0.0, np.pi / 4.0, n_elements + 1)
    return np.column_stack(
        [radius * np.sin(angle), radius * (1.0 - np.cos(angle)), np.zeros_like(angle)]
    )


def case_bend45():
    modulus, shear_modulus = 1.0e7, 5.0e6
    area, second_moment = 1.0, 1.0 / 12.0
    section = matched_section(modulus, shear_modulus, area, second_moment)

    element_counts = (8, 16, 32)
    results = {name: {"tip": [], "time": [], "converged": []} for name in SOLVERS}

    for n_elements in element_counts:
        nodes = bend45_geometry(n_elements)
        for name, solver in SOLVERS.items():
            try:
                positions, elapsed, converged, iterations = solver(
                    nodes, section, len(nodes) - 1, force=[0.0, 0.0, 600.0]
                )
                tip = positions[-1]
            except Exception as error:  # noqa: BLE001
                print(f"  {name} n={n_elements}: {type(error).__name__}: {error}")
                tip, elapsed, converged, iterations = (
                    np.full(3, np.nan), np.nan, False, 0
                )
            displacement = np.asarray(tip, dtype=float) - nodes[-1]
            results[name]["tip"].append(displacement)
            results[name]["time"].append(elapsed)
            results[name]["converged"].append(converged)
            reference = np.array(BEND45_REFERENCES["Simo & Vu-Quoc (1986)"])
            print(f"  {name:11s} n={n_elements:3d}  disp=({displacement[0]:7.3f},"
                  f"{displacement[1]:7.3f},{displacement[2]:7.3f})  "
                  f"|d|={np.linalg.norm(displacement):6.2f} "
                  f"(ref {np.linalg.norm(reference):.2f})  {elapsed:6.3f} s  "
                  f"{iterations:4d} it  converged={converged}")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    labels = ["disp x", "disp y", "disp z"]
    offsets = np.arange(3)
    width = 0.22

    for index, (name, marker) in enumerate(MARKERS.items()):
        tip = np.array(results[name]["tip"][-1], dtype=float)
        left.bar(offsets + (index - 0.5) * width, tip, width,
                 color=COLOURS[name], label=f"{name} ({element_counts[-1]} el)")
    for style, (source, value) in zip(["--", ":", "-."], BEND45_REFERENCES.items()):
        for axis, component in enumerate(value):
            left.hlines(component, axis - 0.45, axis + 0.45, color=PALETTE["Black"],
                        linestyles=style, linewidth=1.4,
                        label=source if axis == 0 else None)
    left.set_xticks(offsets, labels)
    left.set_ylabel("tip displacement (-)")
    left.legend(frameon=False, fontsize=8)
    left.set_title("Tip displacement against published references", fontsize=10)

    for name in SOLVERS:
        spread = []
        for tip in results[name]["tip"]:
            errors = [
                np.linalg.norm(np.asarray(tip, dtype=float) - np.array(value))
                for value in BEND45_REFERENCES.values()
            ]
            spread.append(min(errors))
        right.semilogy(element_counts, np.maximum(spread, 1e-16), color=COLOURS[name],
                       marker=MARKERS[name], label=name)
    band = max(
        np.linalg.norm(np.array(a) - np.array(b))
        for a in BEND45_REFERENCES.values()
        for b in BEND45_REFERENCES.values()
    )
    right.axhline(band, color=PALETTE["Black"], linestyle="--",
                  label="spread between published references")
    right.set_xlabel("elements (-)")
    right.set_ylabel("distance to nearest reference (-)")
    right.legend(frameon=False, fontsize=8)
    right.grid(alpha=0.3, which="both")
    right.set_title("Convergence", fontsize=10)

    figure.suptitle("Bathe & Bolourchi 45-degree curved cantilever, out-of-plane "
                    "load F = 600 (tip displacement)", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


# ---------------------------------------------------------------------------
# Case 4 -- the inflatable tube law, ported from secant stiffness to energy
# ---------------------------------------------------------------------------


def inflatable_cantilever(tube_law, n_elements=20, length=1.0):
    nodes = np.column_stack(
        [np.linspace(0.0, length, n_elements + 1), np.zeros(n_elements + 1),
         np.zeros(n_elements + 1)]
    )
    frames = initial_frames_from_polyline(nodes)
    connectivity = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    tubes = build_inflatable_beam_elements(
        nodes, connectivity, tube_law, frames,
        axial_stiffness=1.0e8, shear_stiffness=1.0e8, name="tubes",
    )
    model = StructuralModel(
        nodes, [tubes], node_frames=frames,
        fixed_translation_nodes=[0], fixed_rotation_nodes=[0],
    )
    return model, tubes


def sweep_end_moment(model, tubes, tube_law, axis, magnitudes, load_steps=4):
    """Ramp a tip moment and read back the curvature or twist it produces."""
    solver = MinimumEnergySolver(model, tolerance=1e-10, max_iterations=3000)
    curvature, twist = [], []
    for magnitude in magnitudes:
        state = None
        for step in range(1, load_steps + 1):
            moments = np.zeros((model.layout.n_rotational_nodes, 3))
            moments[-1, axis] = magnitude * step / load_steps
            solution = solver.solve(
                np.zeros((model.n_nodes, 3)), moments=moments, state=state
            )
            state = solution.state
        diagnostic = inflatable_beam_state(model, state, tubes, tube_law)
        curvature.append(diagnostic["curvature"].mean())
        twist.append(diagnostic["twist_rate"].mean())
        solver.reset_warm_start()
    return np.array(curvature), np.array(twist)


def case_inflatable():
    """Completes kite_fem/examples/FEM_beam_verification.py.

    That example plots the fitted bending and torsion curves but leaves the
    simulation loop as TODO comments. Here the fitted curves are drawn and the
    ported energy element is solved on top of them: a pure end moment gives a
    constant-curvature arc, so the curvature the element settles at must be the
    one the fit prescribes -- exactly, at any deflection.
    """
    pressures = (0.3, 0.5)
    diameter = 0.16
    styles = {0.3: "-", 0.5: "--"}

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for pressure in pressures:
        tube_law = InflatableTubeLaw.from_fit(diameter, pressure)
        model, tubes = inflatable_cantilever(tube_law)

        fractions = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.97])
        applied = fractions * tube_law.moment_max
        measured, _ = sweep_end_moment(model, tubes, tube_law, 1, applied)

        curvatures = np.linspace(
            0.0, max(1.6 * tube_law.curvature_collapse, 1.1 * measured.max()), 400
        )
        left.plot(curvatures, tube_law.moment(curvatures), color=PALETTE["Black"],
                  linestyle=styles[pressure], linewidth=1.5,
                  label=f"ASKITE fit, p={pressure} bar")
        left.axvline(tube_law.curvature_collapse, color=PALETTE["Sky Blue"],
                     linestyle=styles[pressure], linewidth=1.2,
                     label=f"collapse, p={pressure} bar")

        # Past the collapse curvature the fit is extrapolation, so those points
        # are drawn hollow: the solve is fine, the calibration is not.
        beyond = measured >= tube_law.curvature_collapse
        left.plot(measured[~beyond], applied[~beyond], color=PALETTE["Vermillion"],
                  marker="o", linestyle="none", markersize=6,
                  label="ported energy element" if pressure == pressures[0] else None)
        left.plot(measured[beyond], applied[beyond], color=PALETTE["Vermillion"],
                  marker="o", linestyle="none", markersize=6,
                  markerfacecolor="none", markeredgewidth=1.4,
                  label="beyond collapse (extrapolated)"
                  if pressure == pressures[0] else None)
        worst = np.abs(tube_law.moment(measured) / applied - 1.0).max()
        print(f"  bending p={pressure} bar: worst moment error {worst * 100:.3f}%")

        torques = np.linspace(0.0, 0.95 * tube_law.torque_max, 200)
        twist_axis = np.tan(torques / tube_law.torsion_c1) / tube_law.torsion_c2
        right.plot(twist_axis, torques, color=PALETTE["Black"],
                   linestyle=styles[pressure], linewidth=1.5,
                   label=f"ASKITE fit, p={pressure} bar")

        applied_torque = np.array([0.2, 0.4, 0.6, 0.75, 0.85]) * tube_law.torque_max
        _, measured_twist = sweep_end_moment(model, tubes, tube_law, 0, applied_torque)
        right.plot(measured_twist, applied_torque, color=PALETTE["Vermillion"],
                   marker="o", linestyle="none", markersize=6,
                   label="ported energy element" if pressure == pressures[0] else None)
        worst = np.abs(tube_law.torque(measured_twist) / applied_torque - 1.0).max()
        print(f"  torsion p={pressure} bar: worst torque error {worst * 100:.3f}%")

    left.set_xlabel(r"curvature $\kappa$ (m$^{-1}$)")
    left.set_ylabel("bending moment (N m)")
    left.legend(frameon=False, fontsize=8)
    left.grid(alpha=0.3)
    left.set_title(
        f"Bending, d={diameter} m\n"
        r"$M = M_{max}(1 - e^{-EI_0 \kappa / M_{max}})$",
        fontsize=10,
    )

    right.set_xlabel(r"twist rate $\omega_1$ (rad m$^{-1}$)")
    right.set_ylabel("torque (N m)")
    right.legend(frameon=False, fontsize=8)
    right.grid(alpha=0.3)
    right.set_title(
        f"Torsion, d={diameter} m\n" r"$T = c_1 \arctan(c_2 \omega_1)$",
        fontsize=10,
    )

    figure.suptitle(
        "Inflatable tube: ASKITE fits recast as a strain energy, verified by pure "
        "end-moment solves",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


CASES = {
    "elastica": case_elastica,
    "rollup": case_rollup,
    "bend45": case_bend45,
    "inflatable": case_inflatable,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name in arguments.case or sorted(CASES):
        print(f"[{name}]")
        started = time.perf_counter()
        figure = CASES[name]()
        target = arguments.output_dir / f"{name}.png"
        figure.savefig(target, dpi=160)
        print(f"  -> {target}  ({time.perf_counter() - started:.1f} s)")
        if not arguments.show:
            plt.close(figure)
    if arguments.show:
        plt.show()


if __name__ == "__main__":
    main()

# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Controlled cost and accuracy comparison: this model against kite_fem.

Identical geometry, identical mesh, matched linear section properties, and the
same reference (the exact Euler elastica). Everything is measured on the same
machine in one run so the numbers are comparable, which the scattered timings in
the benchmark scripts are not.

Separately reported, because they behave very differently:

* **setup**  -- building the model and the solver. For ``structural`` that is
  the CasADi graph plus the IPOPT problem, paid once and reused for every
  subsequent solve. For ``kite_fem`` it is the pyfe3d element and sparsity
  allocation.
* **cold solve** -- first solve from the undeformed configuration.
* **warm solve** -- a second solve at a 5% different load. ``structural`` warm
  starts from the previous state and duals; ``kite_fem.FEM_structure.solve``
  resets the displacement to zero on every call, so it cannot, and the number
  is reported as-is rather than pretending otherwise.

``kite_fem`` runs with ``I_stiffness=0``: its default of 25 N/m is an absolute
regularisation larger than these beams (``EI/L^3`` = 10 N/m) and stalls them.

Run from the project root::

    python scripts/structural/run_cost_comparison.py
    python scripts/structural/run_cost_comparison.py --alpha 6 --output table.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from awetrim.structural import MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import build_beam_elements, initial_frames_from_polyline

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_validation_benchmarks import elastica_reference, matched_section  # noqa: E402

LENGTH = 1.0
MODULUS = 1.0e7
SECOND_MOMENT = 1.0e-6
AREA = 1.0e-2
BENDING = MODULUS * SECOND_MOMENT


def geometry(n_elements: int):
    return np.column_stack(
        [
            np.linspace(0.0, LENGTH, n_elements + 1),
            np.zeros(n_elements + 1),
            np.zeros(n_elements + 1),
        ]
    )


def measure_structural(n_elements: int, load: float) -> dict:
    nodes = geometry(n_elements)
    section = matched_section(MODULUS, MODULUS / 2.6, AREA, SECOND_MOMENT)

    started = time.perf_counter()
    frames = initial_frames_from_polyline(nodes)
    connectivity = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    beams = build_beam_elements(nodes, connectivity, section, frames, name="beam")
    model = StructuralModel(
        nodes, [beams], node_frames=frames,
        fixed_translation_nodes=[0], fixed_rotation_nodes=[0],
    )
    solver = MinimumEnergySolver(model, tolerance=1e-9, max_iterations=3000)
    setup = time.perf_counter() - started

    forces = np.zeros((model.n_nodes, 3))
    forces[-1, 2] = load

    started = time.perf_counter()
    cold = solver.solve(forces)
    cold_time = time.perf_counter() - started

    started = time.perf_counter()
    warm = solver.solve(1.05 * forces, state=cold.state)
    warm_time = time.perf_counter() - started

    return {
        "setup": setup,
        "cold": cold_time,
        "warm": warm_time,
        "iterations": cold.iterations,
        "residual": cold.residual_norm,
        "converged": cold.converged,
        "tip": cold.state.positions[-1, [0, 2]] / LENGTH,
        "dof": model.layout.n_dof,
    }


def measure_kite_fem(n_elements: int, load: float) -> dict:
    from kite_fem.FEMStructure import FEM_structure

    nodes = geometry(n_elements)
    section = matched_section(MODULUS, MODULUS / 2.6, AREA, SECOND_MOMENT)
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)

    started = time.perf_counter()
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
    for element, length in zip(structure.beam_elements, lengths):
        element.set_beam_properties(MODULUS, AREA, SECOND_MOMENT, float(length))
        element.prop.G = section.shear_modulus
        element.prop.J = 2.0 * SECOND_MOMENT
        element.update_inflatable_beam_properties = lambda: None
    setup = time.perf_counter() - started

    external = np.zeros(structure.N)
    external[6 * n_elements + 2] = load

    started = time.perf_counter()
    converged, _ = structure.solve(
        external, max_iterations=3000, tolerance=1e-6,
        convergence_criteria="residual", I_stiffness=0.0, print_info=False,
    )
    cold_time = time.perf_counter() - started
    iterations = len(structure.iteration_history)
    residual = structure.residual_norm_history[-1]
    positions = np.asarray(structure.coords_current, dtype=float).reshape(-1, 3)

    started = time.perf_counter()
    structure.solve(
        1.05 * external, max_iterations=3000, tolerance=1e-6,
        convergence_criteria="residual", I_stiffness=0.0, print_info=False,
    )
    warm_time = time.perf_counter() - started

    return {
        "setup": setup,
        "cold": cold_time,
        "warm": warm_time,
        "iterations": iterations,
        "residual": residual,
        "converged": bool(converged),
        "tip": positions[-1, [0, 2]] / LENGTH,
        "dof": 6 * (n_elements + 1),
    }


MEASURE = {"structural": measure_structural, "kite_fem": measure_kite_fem}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, action="append",
                        help="load levels P L^2 / EI (default: 1 and 6)")
    parser.add_argument("--elements", type=int, action="append",
                        help="element counts (default: 10 20 40 80)")
    parser.add_argument("--output", type=Path,
                        default=Path("results/structural/validation/cost_comparison.md"))
    arguments = parser.parse_args()

    alphas = arguments.alpha or [1.0, 6.0]
    counts = arguments.elements or [10, 20, 40, 80]
    exact = {alpha: tip for alpha, tip in zip(alphas, elastica_reference(alphas))}

    rows = []
    for alpha in alphas:
        load = alpha * BENDING / LENGTH**2
        for n_elements in counts:
            for name, measure in MEASURE.items():
                try:
                    result = measure(n_elements, load)
                except Exception as error:  # noqa: BLE001 - report, do not hide
                    print(f"  {name} n={n_elements} alpha={alpha}: "
                          f"{type(error).__name__}: {error}")
                    continue
                result["error"] = float(np.linalg.norm(result["tip"] - exact[alpha]))
                result.update(model=name, alpha=alpha, elements=n_elements)
                rows.append(result)
                print(
                    f"  alpha={alpha:4.1f} n={n_elements:3d} {name:11s} "
                    f"dof={result['dof']:4d} setup={result['setup']:6.3f}s "
                    f"cold={result['cold']:7.3f}s warm={result['warm']:7.3f}s "
                    f"it={result['iterations']:5d} res={result['residual']:.2e} "
                    f"err={result['error']:.3e} conv={result['converged']}"
                )

    header = (
        "| alpha | elements | model | DOF | setup (s) | cold solve (s) | "
        "warm solve (s) | iterations | residual (N) | tip error / L | converged |"
    )
    divider = "|" + "---|" * 11
    lines = [
        "# Cost and accuracy: structural vs kite_fem",
        "",
        "Tip-loaded slender cantilever, matched linear section properties,",
        "reference = the exact Euler elastica. `kite_fem` runs with `I_stiffness=0`.",
        "`kite_fem.FEM_structure.solve` restarts from the undeformed configuration",
        "on every call, so its warm-solve column is a second cold solve.",
        "",
        header,
        divider,
    ]
    for row in rows:
        lines.append(
            f"| {row['alpha']:.1f} | {row['elements']} | {row['model']} | {row['dof']} | "
            f"{row['setup']:.3f} | {row['cold']:.3f} | {row['warm']:.3f} | "
            f"{row['iterations']} | {row['residual']:.2e} | {row['error']:.3e} | "
            f"{row['converged']} |"
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  -> {arguments.output}")


if __name__ == "__main__":
    main()

# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""One canopy bay, three ways -- separating the canopy model from the solver.

The full-kite comparison is currently uninformative: ``kite_fem`` does not
converge on a whole kite, which is a known open issue on its side (the ASKITE
integration commit says as much). A single bay is small, fast, and isolates the
two effects that get conflated:

1. ``structural`` + **membrane**   -- CST with tension-field wrinkling
2. ``structural`` + **spring net** -- the ``kite_fem`` canopy recipe, minimised
3. ``kite_fem``  + **spring net**  -- the same net, its own damped Newton

Row 2 versus row 3 is purely the **solver**: identical elements, identical rest
lengths, identical loads. Row 1 versus row 2 is purely the **canopy model**.

Where both solvers converge they must agree, and they do -- that agreement is
what makes the ``kite_fem`` driver here trustworthy.

Run from the project root::

    python scripts/structural/run_panel_cost.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from awetrim.structural import MinimumEnergySolver, StructuralModel  # noqa: E402
from awetrim.structural.elements import build_cable_elements  # noqa: E402

import run_mesh_requirement as bay  # noqa: E402

NET_STIFFNESS = bay.FABRIC_MODULUS * bay.FABRIC_THICKNESS  # k = E t on a square grid


def net_links(n_chord: int, n_span: int):
    """Chordwise and spanwise links only -- no diagonals, as kite_fem does it."""

    def index(i, j):
        return i * (n_span + 1) + j

    links = []
    for i in range(n_chord + 1):
        for j in range(n_span + 1):
            if i < n_chord:
                links.append([index(i, j), index(i + 1, j)])
            if j < n_span:
                links.append([index(i, j), index(i, j + 1)])
    return np.array(links)


def structural_net(n_chord: int, n_span: int, slack: float) -> dict:
    model_ref, nodes, _, edge_nodes, forces, _ = bay.panel(n_chord, n_span, slack)
    reference = nodes * (1.0 + slack)
    links = net_links(n_chord, n_span)
    rest = np.linalg.norm(reference[links[:, 1]] - reference[links[:, 0]], axis=1)

    net = build_cable_elements(
        links, rest, np.full(len(links), NET_STIFFNESS), name="net"
    )
    model = StructuralModel(nodes, [net], fixed_translation_nodes=edge_nodes)

    started = time.perf_counter()
    solver = MinimumEnergySolver(model, tolerance=1e-9)
    setup = time.perf_counter() - started
    started = time.perf_counter()
    solution = solver.solve(forces)
    solve = time.perf_counter() - started

    return {
        "model": "structural + net",
        "elements": len(links),
        "dof": model.layout.n_dof,
        "setup": setup,
        "solve": solve,
        "iterations": solution.iterations,
        "residual": solution.residual_norm,
        "converged": solution.converged,
        "camber": float(solution.state.positions[:, 2].max() / bay.CHORD),
    }


def kite_fem_net(n_chord: int, n_span: int, slack: float,
                 tolerance: float = 1e-2, i_stiffness: float = 25.0) -> dict:
    from kite_fem.FEMStructure import FEM_structure

    _, nodes, _, edge_nodes, forces, _ = bay.panel(n_chord, n_span, slack)
    reference = nodes * (1.0 + slack)
    links = net_links(n_chord, n_span)

    spring_matrix = [
        [int(a), int(b), NET_STIFFNESS, 0.0,
         float(np.linalg.norm(reference[b] - reference[a])), "noncompressive"]
        for a, b in links
    ]
    fixed = set(edge_nodes.tolist())
    initial_conditions = [
        [np.asarray(point, dtype=float), np.zeros(3), 1.0, index in fixed]
        for index, point in enumerate(nodes)
    ]

    started = time.perf_counter()
    structure = FEM_structure(
        initial_conditions=initial_conditions, spring_matrix=spring_matrix
    )
    setup = time.perf_counter() - started

    external = np.zeros(structure.N)
    for node in range(len(nodes)):
        external[6 * node: 6 * node + 3] = forces[node]

    started = time.perf_counter()
    converged, _ = structure.solve(
        external, max_iterations=3000, tolerance=tolerance,
        convergence_criteria="residual", I_stiffness=i_stiffness, print_info=False,
    )
    solve = time.perf_counter() - started
    positions = np.asarray(structure.coords_current, dtype=float).reshape(-1, 3)

    return {
        "model": "kite_fem + net",
        "elements": len(links),
        "dof": structure.N,
        "setup": setup,
        "solve": solve,
        "iterations": len(structure.iteration_history),
        "residual": structure.residual_norm_history[-1],
        "converged": bool(converged),
        "camber": float(positions[:, 2].max() / bay.CHORD),
    }


def structural_membrane(n_chord: int, n_span: int, slack: float) -> dict:
    result = bay.solve_panel(n_chord, n_span, slack)
    return {
        "model": "structural + membrane",
        "elements": result["n_triangles"],
        "dof": result["dof"],
        "setup": result["build"],
        "solve": result["solve"],
        "iterations": result["iterations"],
        "residual": 0.0,
        "converged": result["converged"],
        "camber": result["camber"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/structural/validation/panel_cost.md"))
    arguments = parser.parse_args()

    meshes = [(4, 3), (8, 6), (12, 9), (16, 12)]
    rows = []
    for slack, label in [(0.0, "taut"), (0.03, "slack 3%")]:
        print(f"[{label}]  one bay {bay.CHORD} x {bay.BAY_WIDTH} m at "
              f"{bay.PRESSURE:.0f} Pa, all edges fixed")
        for n_chord, n_span in meshes:
            for build in (structural_membrane, structural_net, kite_fem_net):
                try:
                    result = build(n_chord, n_span, slack)
                except Exception as error:  # noqa: BLE001 - report, do not hide
                    print(f"    {n_chord}x{n_span} {build.__name__}: "
                          f"{type(error).__name__}: {error}")
                    continue
                result.update(mesh=f"{n_chord}x{n_span}", case=label)
                rows.append(result)
                print(
                    f"    {result['mesh']:6s} {result['model']:22s} "
                    f"{result['elements']:4d} el {result['dof']:5d} dof  "
                    f"setup={result['setup']:5.2f}s solve={result['solve']:7.3f}s  "
                    f"it={result['iterations']:5d}  camber={result['camber'] * 100:6.3f}%c "
                    f"conv={result['converged']}"
                )

    lines = [
        "# One canopy bay: canopy model vs solver",
        "",
        f"{bay.CHORD} x {bay.BAY_WIDTH} m bay, {bay.PRESSURE:.0f} Pa, all edges fixed.",
        "Rows 2 and 3 solve the *identical* spring net, so their difference is the",
        "solver alone. Row 1 changes the canopy model instead.",
        "",
        "| case | mesh | model | elements | DOF | setup (s) | solve (s) | iterations | "
        "camber (% c) | converged |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['mesh']} | {row['model']} | {row['elements']} | "
            f"{row['dof']} | {row['setup']:.2f} | {row['solve']:.3f} | "
            f"{row['iterations']} | {row['camber'] * 100:.3f} | {row['converged']} |"
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  -> {arguments.output}")


if __name__ == "__main__":
    main()

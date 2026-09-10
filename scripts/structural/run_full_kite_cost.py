# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""One full-kite structural solve, timed in both models.

A V3-scale structure: an inflated leading-edge tube, inflated struts at every
bay boundary, a canopy over nine bays, a trailing-edge hem and a bridle down to
a fixed attachment, under uniform differential pressure.

Both models get the **same nodes, the same tubes and the same cables**. The one
thing they cannot share is the canopy, because ``kite_fem`` has no membrane
element:

* ``structural`` -- CST membrane with tension-field wrinkling;
* ``kite_fem``   -- the noncompressive spring net of its own
  ``examples/FEM_canopy_section.py``, chordwise and spanwise links on the same
  grid.

So this is not a like-for-like accuracy comparison -- it is "what does one
full-kite solve cost in each code's native canopy model". The tubes *are*
like-for-like: both run the ASKITE inflatable law, one as a secant stiffness and
one as the ported strain energy.

``kite_fem`` is run at its shipped ``I_stiffness=25`` and at 0, and the better of
the two is reported, since that constant is kite-scale tuned and this is a kite.

Run from the project root::

    python scripts/structural/run_full_kite_cost.py
    python scripts/structural/run_full_kite_cost.py --chord-elements 12 --span-per-bay 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from awetrim.structural import MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import (
    InflatableTubeLaw,
    build_cable_elements,
    build_inflatable_beam_elements,
    build_membrane_elements,
    initial_frames_from_polyline,
)

SPAN = 8.28  # m, LEI-V3
CHORD = 2.0  # m
ARC_HEIGHT = 1.6  # m, canopy arc rise from centre to tip
BAYS = 9
PRESSURE = 250.0  # Pa

LE_DIAMETER, LE_PRESSURE = 0.16, 0.3  # m, bar
STRUT_DIAMETER, STRUT_PRESSURE = 0.10, 0.3

TUBE_AXIAL = 2.0e6  # N,   not covered by the ASKITE fits
TUBE_SHEAR = 5.0e5  # N

FABRIC_MODULUS, FABRIC_POISSON, FABRIC_THICKNESS = 5.0e8, 0.3, 3.0e-4
NET_STIFFNESS = FABRIC_MODULUS * FABRIC_THICKNESS  # k = E t for a square grid
HEM_STIFFNESS = 2.0e5
BRIDLE_STIFFNESS = 4.0e5


def kite_grid(n_chord: int, span_per_bay: int):
    """Nodes, triangles and the index maps of an arched V3-like canopy."""
    n_span = BAYS * span_per_bay
    chord_axis = np.linspace(0.0, CHORD, n_chord + 1)
    span_axis = np.linspace(-0.5 * SPAN, 0.5 * SPAN, n_span + 1)
    grid_x, grid_y = np.meshgrid(chord_axis, span_axis, indexing="ij")
    arc = -ARC_HEIGHT * (grid_y / (0.5 * SPAN)) ** 2
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel(), arc.ravel()])

    def index(i, j):
        return i * (n_span + 1) + j

    triangles = []
    for i in range(n_chord):
        for j in range(n_span):
            triangles.append([index(i, j), index(i + 1, j), index(i + 1, j + 1)])
            triangles.append([index(i, j), index(i + 1, j + 1), index(i, j + 1)])

    leading_edge = np.array([index(0, j) for j in range(n_span + 1)])
    trailing_edge = np.array([index(n_chord, j) for j in range(n_span + 1)])
    struts = [
        np.array([index(i, bay * span_per_bay) for i in range(n_chord + 1)])
        for bay in range(BAYS + 1)
    ]
    return nodes, np.array(triangles), leading_edge, trailing_edge, struts, index


def pressure_load(nodes, triangles):
    """Uniform differential pressure, lumped a third to each triangle corner."""
    forces = np.zeros_like(nodes)
    for triangle in triangles:
        corners = nodes[triangle]
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        area = 0.5 * np.linalg.norm(normal)
        direction = normal / max(np.linalg.norm(normal), 1e-12)
        if direction[2] < 0.0:
            direction = -direction
        forces[triangle] += PRESSURE * area * direction / 3.0
    return forces


def spring_net_links(n_chord: int, n_span: int, index):
    """Chordwise and spanwise links -- the kite_fem canopy recipe, no diagonals."""
    links = []
    for i in range(n_chord + 1):
        for j in range(n_span + 1):
            if i < n_chord:
                links.append([index(i, j), index(i + 1, j)])
            if j < n_span:
                links.append([index(i, j), index(i, j + 1)])
    return np.array(links)


def bridle_attachments(n_chord: int, span_per_bay: int, index):
    """A few trailing-edge and mid-chord stations feeding one attachment point."""
    n_span = BAYS * span_per_bay
    stations = np.linspace(0, n_span, 7, dtype=int)
    return [index(n_chord, j) for j in stations] + [
        index(n_chord // 2, j) for j in stations[1:-1]
    ]


def build_structural(n_chord: int, span_per_bay: int):
    nodes, triangles, leading_edge, trailing_edge, struts, index = kite_grid(
        n_chord, span_per_bay
    )
    n_span = BAYS * span_per_bay
    forces = pressure_load(nodes, triangles)

    anchor = len(nodes)
    attachments = bridle_attachments(n_chord, span_per_bay, index)
    nodes = np.vstack([nodes, [0.5 * CHORD, 0.0, -8.0]])
    forces = np.vstack([forces, np.zeros(3)])

    frames = np.tile(np.eye(3), (len(nodes), 1, 1))
    frames[leading_edge] = initial_frames_from_polyline(nodes[leading_edge])
    for strut in struts:
        frames[strut] = initial_frames_from_polyline(nodes[strut])

    le_law = InflatableTubeLaw.from_fit(LE_DIAMETER, LE_PRESSURE)
    strut_law = InflatableTubeLaw.from_fit(STRUT_DIAMETER, STRUT_PRESSURE)

    tube = build_inflatable_beam_elements(
        nodes, np.column_stack([leading_edge[:-1], leading_edge[1:]]), le_law, frames,
        axial_stiffness=TUBE_AXIAL, shear_stiffness=TUBE_SHEAR, name="le_tube",
    )
    strut_connectivity = np.vstack(
        [np.column_stack([strut[:-1], strut[1:]]) for strut in struts]
    )
    strut_elements = build_inflatable_beam_elements(
        nodes, strut_connectivity, strut_law, frames,
        axial_stiffness=TUBE_AXIAL, shear_stiffness=TUBE_SHEAR, name="struts",
    )
    canopy = build_membrane_elements(
        nodes, triangles, FABRIC_THICKNESS, FABRIC_MODULUS, FABRIC_POISSON,
        name="canopy",
    )
    hem_connectivity = np.column_stack([trailing_edge[:-1], trailing_edge[1:]])
    hem = build_cable_elements(
        hem_connectivity,
        np.linalg.norm(nodes[hem_connectivity[:, 1]] - nodes[hem_connectivity[:, 0]],
                       axis=1),
        np.full(len(hem_connectivity), HEM_STIFFNESS),
        name="hem",
    )
    bridle_connectivity = np.array([[node, anchor] for node in attachments])
    bridles = build_cable_elements(
        bridle_connectivity,
        np.linalg.norm(nodes[bridle_connectivity[:, 0]] - nodes[anchor], axis=1) * 0.99,
        np.full(len(attachments), BRIDLE_STIFFNESS),
        name="bridles",
    )

    model = StructuralModel(
        nodes,
        [tube, strut_elements, canopy, hem, bridles],
        node_frames=frames,
        fixed_translation_nodes=[anchor],
        fixed_rotation_nodes=[int(leading_edge[len(leading_edge) // 2])],
    )
    counts = {
        "triangles": len(triangles),
        "beams": len(leading_edge) - 1 + len(strut_connectivity),
        "cables": len(hem_connectivity) + len(bridle_connectivity),
        "nodes": len(nodes),
        "n_span": n_span,
    }
    return model, forces, counts


def measure_structural(n_chord: int, span_per_bay: int) -> dict:
    started = time.perf_counter()
    model, forces, counts = build_structural(n_chord, span_per_bay)
    solver = MinimumEnergySolver(model, tolerance=1e-8, max_iterations=2000)
    setup = time.perf_counter() - started

    started = time.perf_counter()
    solution = solver.solve(forces)
    cold = time.perf_counter() - started

    started = time.perf_counter()
    warm = solver.solve(1.05 * forces, state=solution.state)
    warm_time = time.perf_counter() - started

    return {
        "model": "structural",
        "dof": model.layout.n_dof,
        "setup": setup,
        "cold": cold,
        "warm": warm_time,
        "iterations": solution.iterations,
        "residual": solution.residual_norm,
        "converged": solution.converged,
        "detail": (
            f"{counts['triangles']} membrane tri, {counts['beams']} inflatable beams, "
            f"{counts['cables']} cables"
        ),
        **counts,
    }


def measure_kite_fem(n_chord: int, span_per_bay: int, i_stiffness: float) -> dict:
    from kite_fem.FEMStructure import FEM_structure

    started = time.perf_counter()
    nodes, triangles, leading_edge, trailing_edge, struts, index = kite_grid(
        n_chord, span_per_bay
    )
    n_span = BAYS * span_per_bay
    forces = pressure_load(nodes, triangles)

    anchor = len(nodes)
    attachments = bridle_attachments(n_chord, span_per_bay, index)
    nodes = np.vstack([nodes, [0.5 * CHORD, 0.0, -8.0]])
    forces = np.vstack([forces, np.zeros(3)])

    spring_matrix = []
    for node_a, node_b in spring_net_links(n_chord, n_span, index):
        rest = float(np.linalg.norm(nodes[node_b] - nodes[node_a]))
        spring_matrix.append([int(node_a), int(node_b), NET_STIFFNESS, 0.0, rest,
                              "noncompressive"])
    for node_a, node_b in zip(trailing_edge[:-1], trailing_edge[1:]):
        rest = float(np.linalg.norm(nodes[node_b] - nodes[node_a]))
        spring_matrix.append([int(node_a), int(node_b), HEM_STIFFNESS, 0.0, rest,
                              "noncompressive"])
    for node in attachments:
        rest = float(np.linalg.norm(nodes[node] - nodes[anchor])) * 0.99
        spring_matrix.append([int(node), int(anchor), BRIDLE_STIFFNESS, 0.0, rest,
                              "noncompressive"])

    # beam_matrix columns are [n1, n2, d, p, l0] (FEMStructure.__setup_beam_elements)
    beam_matrix = []
    for node_a, node_b in zip(leading_edge[:-1], leading_edge[1:]):
        rest = float(np.linalg.norm(nodes[node_b] - nodes[node_a]))
        beam_matrix.append([int(node_a), int(node_b), LE_DIAMETER, LE_PRESSURE, rest])
    for strut in struts:
        for node_a, node_b in zip(strut[:-1], strut[1:]):
            rest = float(np.linalg.norm(nodes[node_b] - nodes[node_a]))
            beam_matrix.append(
                [int(node_a), int(node_b), STRUT_DIAMETER, STRUT_PRESSURE, rest]
            )

    initial_conditions = [
        [np.asarray(point, dtype=float), np.zeros(3), 1.0, i == anchor]
        for i, point in enumerate(nodes)
    ]
    structure = FEM_structure(
        initial_conditions=initial_conditions,
        spring_matrix=spring_matrix,
        beam_matrix=beam_matrix,
    )
    setup = time.perf_counter() - started

    external = np.zeros(structure.N)
    for node in range(len(nodes)):
        external[6 * node: 6 * node + 3] = forces[node]

    started = time.perf_counter()
    converged, _ = structure.solve(
        external, max_iterations=2000, tolerance=1e-2,
        convergence_criteria="residual", I_stiffness=i_stiffness, print_info=False,
    )
    cold = time.perf_counter() - started

    started = time.perf_counter()
    structure.solve(
        1.05 * external, max_iterations=2000, tolerance=1e-2,
        convergence_criteria="residual", I_stiffness=i_stiffness, print_info=False,
    )
    warm_time = time.perf_counter() - started

    return {
        "model": f"kite_fem (I={i_stiffness:g})",
        "dof": structure.N,
        "setup": setup,
        "cold": cold,
        "warm": warm_time,
        "iterations": len(structure.iteration_history),
        "residual": structure.residual_norm_history[-1],
        "converged": bool(converged),
        "detail": (
            f"{len(spring_matrix)} springs (net + hem + bridle), "
            f"{len(beam_matrix)} inflatable beams"
        ),
        "nodes": len(nodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chord-elements", type=int, default=8)
    parser.add_argument("--span-per-bay", type=int, default=6)
    parser.add_argument("--output", type=Path,
                        default=Path("results/structural/validation/full_kite_cost.md"))
    arguments = parser.parse_args()

    n_chord, span_per_bay = arguments.chord_elements, arguments.span_per_bay
    print(f"[full kite] {n_chord} chordwise x {span_per_bay} per bay x {BAYS} bays")

    rows = []
    try:
        rows.append(measure_structural(n_chord, span_per_bay))
    except Exception as error:  # noqa: BLE001 - report, do not hide
        print(f"  structural failed: {type(error).__name__}: {error}")
    for i_stiffness in (25.0, 0.0):
        try:
            rows.append(measure_kite_fem(n_chord, span_per_bay, i_stiffness))
        except Exception as error:  # noqa: BLE001
            print(f"  kite_fem I={i_stiffness:g} failed: "
                  f"{type(error).__name__}: {error}")

    for row in rows:
        print(
            f"  {row['model']:20s} dof={row['dof']:5d} setup={row['setup']:6.2f}s "
            f"cold={row['cold']:8.2f}s warm={row['warm']:8.2f}s "
            f"it={row['iterations']:5d} res={row['residual']:.2e} "
            f"conv={row['converged']}\n      {row['detail']}"
        )

    lines = [
        "# One full-kite structural solve",
        "",
        f"V3-scale: {SPAN} m span, {CHORD} m chord, {BAYS} bays, {PRESSURE:.0f} Pa.",
        f"Canopy mesh {n_chord} chordwise x {span_per_bay} per bay.",
        "Same nodes, tubes (ASKITE inflatable law) and cables in both; the canopy is a",
        "membrane in `structural` and a noncompressive spring net in `kite_fem`, which",
        "has no membrane element. Not a like-for-like accuracy comparison.",
        "",
        "| model | DOF | setup (s) | cold solve (s) | warm solve (s) | iterations | "
        "residual (N) | converged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['dof']} | {row['setup']:.2f} | {row['cold']:.2f} | "
            f"{row['warm']:.2f} | {row['iterations']} | {row['residual']:.2e} | "
            f"{row['converged']} |"
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  -> {arguments.output}")


if __name__ == "__main__":
    main()

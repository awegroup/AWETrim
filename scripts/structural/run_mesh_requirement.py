# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""How many membrane triangles does a kite canopy actually need?

Chordwise and spanwise resolution are swept **independently**, because they are
not the same problem: a canopy panel billows across the chord, which is what
sets the section the aerodynamics sees, while spanwise it varies slowly between
struts.

The panel is one bay of the LEI-V3 canopy -- chord 2.0 m, strut spacing 1.3 m
(the V3 has 8.28 m span over 10 stations, so ~1.3 m in the centre bays) -- held
on all four edges by the leading-edge tube, the trailing-edge hem and the two
struts, under a uniform 250 Pa differential pressure (about ``q`` at 20 m/s with
``dCp = 1``).

Convergence is judged on quantities the coupling actually consumes:

* **camber** -- peak billow over chord, which is what changes the section the
  VSM sees;
* **section shape** -- RMS deviation of the mid-span chordwise profile from the
  finest mesh, normalised by chord; this is the geometry handed to the aero;
* **edge load** -- resultant transferred into the tube and struts, which is what
  the structure downstream carries.

Peak local strain is deliberately *not* a criterion: in a wrinkling membrane it
does not converge, and it is not what anything downstream reads.

Both a taut panel and a realistically slack one (reference cut 3% oversize) are
run, because slack is the state a real canopy flies in and it is harder.

Run from the project root::

    python scripts/structural/run_mesh_requirement.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from awetrim.plotting import PALETTE
from awetrim.structural import MinimumEnergySolver, StructuralModel
from awetrim.structural.elements import (
    SLACK,
    TAUT,
    WRINKLED,
    build_membrane_elements,
    membrane_regimes,
)

OUTPUT_DIR = Path("results/structural/validation")

CHORD = 2.0  # m
BAY_WIDTH = 1.3  # m, centre-bay strut spacing on the LEI-V3
PRESSURE = 250.0  # Pa

FABRIC_MODULUS = 5.0e8  # Pa
FABRIC_POISSON = 0.3
FABRIC_THICKNESS = 3.0e-4  # m  (E t = 1.5e5 N/m, a technical kite fabric)

#: Spanwise bays on the LEI-V3 canopy, for turning a per-bay count into a total.
V3_BAYS = 9

REGIME_COLORS = ListedColormap(
    [PALETTE["Sky Blue"], PALETTE["Orange"], PALETTE["Vermillion"]]
)
REGIME_LABELS = {SLACK: "slack", WRINKLED: "wrinkled", TAUT: "taut"}
REGIME_NORM = Normalize(-0.5, 2.5)


def panel(n_chord: int, n_span: int, slack: float = 0.0):
    """One canopy bay, clamped on all four edges, under uniform pressure."""
    chord_axis = np.linspace(0.0, CHORD, n_chord + 1)
    span_axis = np.linspace(0.0, BAY_WIDTH, n_span + 1)
    grid_x, grid_y = np.meshgrid(chord_axis, span_axis, indexing="ij")
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])

    def index(i, j):
        return i * (n_span + 1) + j

    triangles = []
    for i in range(n_chord):
        for j in range(n_span):
            triangles.append([index(i, j), index(i + 1, j), index(i + 1, j + 1)])
            triangles.append([index(i, j), index(i + 1, j + 1), index(i, j + 1)])
    triangles = np.array(triangles)

    on_edge = (
        (grid_x.ravel() == 0.0)
        | (grid_y.ravel() == 0.0)
        | (grid_x.ravel() == CHORD)
        | (grid_y.ravel() == BAY_WIDTH)
    )
    edge_nodes = np.flatnonzero(on_edge)

    canopy = build_membrane_elements(
        nodes * (1.0 + slack), triangles, FABRIC_THICKNESS, FABRIC_MODULUS,
        FABRIC_POISSON, name="canopy",
    )
    model = StructuralModel(nodes, [canopy], fixed_translation_nodes=edge_nodes)

    # Uniform pressure lumped by tributary area: each triangle gives a third of
    # its load to each corner, which is the consistent CST distribution.
    forces = np.zeros_like(nodes)
    for triangle in triangles:
        corners = nodes[triangle]
        area = 0.5 * np.linalg.norm(
            np.cross(corners[1] - corners[0], corners[2] - corners[0])
        )
        forces[triangle, 2] += PRESSURE * area / 3.0
    return model, nodes, triangles, edge_nodes, forces, (n_chord, n_span)


def solve_panel(n_chord: int, n_span: int, slack: float = 0.0,
                keep_geometry: bool = False):
    """Solve one bay.

    ``keep_geometry`` retains the deformed mesh for plotting. The sweeps leave
    it off: holding every deformed mesh alive alongside the solvers was enough
    to push the MUMPS factorisation into ``std::bad_alloc``.
    """
    model, nodes, triangles, edge_nodes, forces, shape = panel(n_chord, n_span, slack)

    started = time.perf_counter()
    solver = MinimumEnergySolver(model, tolerance=1e-9)
    build = time.perf_counter() - started

    started = time.perf_counter()
    solution = solver.solve(forces)
    elapsed = time.perf_counter() - started

    positions = solution.state.positions
    # Mid-span chordwise section: the profile the aerodynamics is handed.
    mid = n_span // 2
    section = positions[[i * (n_span + 1) + mid for i in range(n_chord + 1)]]
    regimes = membrane_regimes(positions, model.element_set("canopy"))

    result = {
        "n_chord": n_chord,
        "n_span": n_span,
        "n_triangles": len(triangles),
        "dof": model.layout.n_dof,
        "camber": float(positions[:, 2].max() / CHORD),
        "section_x": section[:, 0] / CHORD,
        "section_z": section[:, 2] / CHORD,
        "edge_load": float(
            np.linalg.norm(solution.internal_forces[edge_nodes].sum(axis=0))
        ),
        "energy": solution.strain_energy,
        "build": build,
        "solve": elapsed,
        "iterations": solution.iterations,
        "converged": solution.converged,
        "wrinkled_fraction": float((regimes["regime"] < 2).mean()),
    }
    if keep_geometry:
        result.update(
            positions=positions, triangles=triangles, regime=regimes["regime"]
        )
    return result


def section_error(coarse: dict, reference: dict) -> float:
    """RMS chordwise-profile difference from the reference mesh, over chord."""
    sample = np.linspace(0.0, 1.0, 201)
    coarse_z = np.interp(sample, coarse["section_x"], coarse["section_z"])
    fine_z = np.interp(sample, reference["section_x"], reference["section_z"])
    return float(np.sqrt(np.mean((coarse_z - fine_z) ** 2)))


def sweep(direction: str, counts, fixed: int, slack: float):
    results = []
    for count in counts:
        n_chord, n_span = (count, fixed) if direction == "chord" else (fixed, count)
        result = solve_panel(n_chord, n_span, slack)
        results.append(result)
        print(
            f"    {direction}={count:3d}  {result['n_triangles']:5d} tri  "
            f"{result['dof']:5d} dof  camber={result['camber'] * 100:6.3f}% c  "
            f"build={result['build']:5.2f}s solve={result['solve']:6.2f}s  "
            f"{result['iterations']:3d} it  wrinkled={result['wrinkled_fraction']:.2f}  "
            f"conv={result['converged']}"
        )
    reference = results[-1]
    for result in results:
        result["section_rms"] = section_error(result, reference)
        result["camber_error"] = abs(result["camber"] / reference["camber"] - 1.0)
        result["load_error"] = abs(result["edge_load"] / reference["edge_load"] - 1.0)
    return results


def recommend(results, label: str, tolerance: float = 0.01) -> int:
    """Coarsest count whose camber and section shape are both within tolerance."""
    for result in results[:-1]:
        if result["camber_error"] < tolerance and result["section_rms"] < tolerance / 10:
            count = result["n_chord"] if label == "chord" else result["n_span"]
            return count
    return results[-1]["n_chord"] if label == "chord" else results[-1]["n_span"]


def draw_panel(axes, result, z_exaggeration: float = 3.0):
    """Deformed bay as a 3-D surface, coloured by wrinkling regime."""
    positions, triangles = result["positions"], result["triangles"]
    polygons = Poly3DCollection(
        [positions[triangle] for triangle in triangles],
        alpha=0.95, edgecolor="0.3", linewidths=0.25,
    )
    polygons.set_facecolor(REGIME_COLORS(REGIME_NORM(result["regime"])))
    axes.add_collection3d(polygons)

    lower, upper = positions.min(axis=0), positions.max(axis=0)
    centre = 0.5 * (lower + upper)
    radius = 0.5 * max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1])) * 1.05
    z_radius = max(0.5 * np.ptp(positions[:, 2]) * 1.15, radius / z_exaggeration)
    axes.set_xlim(centre[0] - radius, centre[0] + radius)
    axes.set_ylim(centre[1] - radius, centre[1] + radius)
    axes.set_zlim(centre[2] - z_radius, centre[2] + z_radius)
    axes.set_box_aspect((1.0, 1.0, min(1.0, z_radius / radius) * z_exaggeration))
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_zticks([])
    axes.view_init(elev=26, azim=-62)


def figure_shapes(output_dir: Path, resolutions=(2, 4, 8, 16)):
    """Uniformly refined bays, side by side, for both the taut and slack cases."""
    cases = {"taut": 0.0, "slack 3%": 0.03}
    figure = plt.figure(figsize=(17.0, 7.4))

    for row, (label, slack) in enumerate(cases.items()):
        sections = []
        for column, count in enumerate(resolutions):
            result = solve_panel(count, count, slack, keep_geometry=True)
            sections.append(result)
            axes = figure.add_subplot(
                2, len(resolutions) + 1, row * (len(resolutions) + 1) + column + 1,
                projection="3d",
            )
            draw_panel(axes, result)
            axes.set_title(
                f"{label}, {count}x{count}\n"
                f"{result['n_triangles']} tri, "
                f"camber {result['camber'] * 100:.2f}% c\n"
                f"{result['build'] + result['solve']:.2f} s",
                fontsize=8,
            )

        overlay = figure.add_subplot(2, len(resolutions) + 1,
                                     row * (len(resolutions) + 1) + len(resolutions) + 1)
        shades = plt.cm.viridis(np.linspace(0.15, 0.9, len(sections)))
        for result, shade in zip(sections, shades):
            overlay.plot(result["section_x"], result["section_z"], color=shade,
                         marker="o", markersize=3,
                         label=f"{result['n_chord']}x{result['n_span']}")
        overlay.set_xlabel("x / c (-)")
        overlay.set_ylabel("z / c (-)")
        overlay.legend(frameon=False, fontsize=7)
        overlay.grid(alpha=0.3)
        overlay.set_title(f"{label}: mid-span section", fontsize=9)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                   color=REGIME_COLORS(REGIME_NORM(code)), label=name)
        for code, name in REGIME_LABELS.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        f"One LEI-V3 canopy bay ({CHORD} x {BAY_WIDTH} m, {PRESSURE:.0f} Pa) "
        "under uniform mesh refinement",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    target = output_dir / "mesh_shapes.png"
    figure.savefig(target, dpi=150)
    plt.close(figure)
    print(f"  -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    chord_counts = [2, 3, 4, 6, 8, 12, 16, 24]
    span_counts = [2, 4, 6, 8, 12, 16, 24]
    cases = {"taut": 0.0, "slack 3%": 0.03}

    chord_sweeps, span_sweeps = {}, {}
    for label, slack in cases.items():
        print(f"  [{label}] chordwise sweep (n_span = 24)")
        chord_sweeps[label] = sweep("chord", chord_counts, 24, slack)
        print(f"  [{label}] spanwise sweep (n_chord = 24)")
        span_sweeps[label] = sweep("span", span_counts, 24, slack)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    colours = {"taut": PALETTE["Vermillion"], "slack 3%": PALETTE["Blue"]}
    markers = {"taut": "o", "slack 3%": "s"}

    for label in cases:
        axes[0].semilogy(
            [r["n_chord"] for r in chord_sweeps[label][:-1]],
            [max(r["camber_error"], 1e-6) for r in chord_sweeps[label][:-1]],
            color=colours[label], marker=markers[label], label=f"{label}, chordwise",
        )
        axes[0].semilogy(
            [r["n_span"] for r in span_sweeps[label][:-1]],
            [max(r["camber_error"], 1e-6) for r in span_sweeps[label][:-1]],
            color=colours[label], marker=markers[label], linestyle="--",
            label=f"{label}, spanwise",
        )
    axes[0].axhline(0.01, color=PALETTE["Black"], linestyle=":", label="1%")
    axes[0].set_xlabel("elements in that direction (-)")
    axes[0].set_ylabel("camber error vs finest mesh (-)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].set_title("Camber: what the aerodynamic section sees", fontsize=10)

    for label in cases:
        axes[1].semilogy(
            [r["n_chord"] for r in chord_sweeps[label][:-1]],
            [max(r["section_rms"], 1e-7) for r in chord_sweeps[label][:-1]],
            color=colours[label], marker=markers[label], label=f"{label}, chordwise",
        )
        axes[1].semilogy(
            [r["n_span"] for r in span_sweeps[label][:-1]],
            [max(r["section_rms"], 1e-7) for r in span_sweeps[label][:-1]],
            color=colours[label], marker=markers[label], linestyle="--",
            label=f"{label}, spanwise",
        )
    axes[1].axhline(1e-3, color=PALETTE["Black"], linestyle=":", label="0.1% chord")
    axes[1].set_xlabel("elements in that direction (-)")
    axes[1].set_ylabel("section RMS error / chord (-)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].set_title("Chordwise profile handed to the aero", fontsize=10)

    reference = chord_sweeps["slack 3%"][-1]
    for label in cases:
        merged = chord_sweeps[label] + span_sweeps[label]
        axes[2].loglog(
            [r["n_triangles"] for r in merged], [r["build"] + r["solve"] for r in merged],
            color=colours[label], marker=markers[label], linestyle="none", label=label,
        )
    axes[2].set_xlabel("triangles in one bay (-)")
    axes[2].set_ylabel("build + solve (s)")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(alpha=0.3, which="both")
    axes[2].set_title("Cost per bay", fontsize=10)

    figure.suptitle(
        f"Canopy mesh requirement: one LEI-V3 bay ({CHORD} x {BAY_WIDTH} m) "
        f"at {PRESSURE:.0f} Pa",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    target = arguments.output_dir / "mesh_requirement.png"
    figure.savefig(target, dpi=160)

    print("\n  Recommendation (1% camber and 0.1% chord section RMS):")
    for label in cases:
        n_chord = recommend(chord_sweeps[label], "chord")
        n_span = recommend(span_sweeps[label], "span")
        per_bay = 2 * n_chord * n_span
        print(
            f"    {label:9s}: {n_chord} chordwise x {n_span} spanwise = "
            f"{per_bay} triangles/bay  ->  {per_bay * V3_BAYS} for a "
            f"{V3_BAYS}-bay V3 canopy"
        )
    print(f"\n  -> {target}")

    print("\n  [shapes] uniform refinement")
    figure_shapes(arguments.output_dir)


if __name__ == "__main__":
    main()

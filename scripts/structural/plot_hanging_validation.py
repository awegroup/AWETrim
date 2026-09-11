# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Figures for the hanging-kite validation: geometry and per-quantity errors.

Three figures, all driven off the ``.npz`` written by
``run_hanging_validation.py`` plus ``kite_fem``'s saved states:

``geometry_<case>.png``  front / top / side projections of the inflatable
                         structure, Billow against kite_fem, laid out like the
                         thesis Figure 7.2 so the two can be read side by side.
``lengths.png``          the twelve measured quantities per case, model over
                         measured, which is the comparison the reference
                         ``shapecorrelation`` metric cannot resolve.
``summary.png``          span and mean relative error across all ten cases.

Run from the project root, after the validation run::

    python scripts/structural/plot_hanging_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from hanging_kite import (LENGTH_NAMES, LOAD_CASES, build_model, canopy_grid,
                          canopy_triangles, kite_fem_state, load_reference)
from awetrim.structural.elements import SLACK, TAUT, WRINKLED, membrane_regimes

BILLOW = "#1f6feb"
SCALED = "#12a150"
MARKER = "#111111"
FABRIC = "#b9c4cc"
TUBE = "#1a1a1a"
KFEM = "#d1495b"
MEASURED = "#111111"
BUILT = "#9aa0a6"
CENTRE_LOAD_CASES = (2, 3, 7, 8)   # neither code reproduces these


def structure_lines(nodes: np.ndarray, grid: np.ndarray):
    """Leading edge and strut polylines -- the inflatable structure only."""
    leading = list(range(1, 58, 2))
    struts = [grid[row].tolist() for row in range(len(grid))
              if int(grid[row, 0]) in (1, 7, 13, 19, 25, 33, 39, 45, 51, 57)]
    return [nodes[leading]] + [nodes[s] for s in struts]


def plot_geometry(axes, nodes, grid, colour, label, planes=("front", "top", "side")):
    projections = {"front": (1, 2, "y (m)", "z (m)"),
                   "top": (1, 0, "y (m)", "x (m)"),
                   "side": (0, 2, "x (m)", "z (m)")}
    for axis, plane in zip(axes, planes):
        i, j, xlabel, ylabel = projections[plane]
        for k, line in enumerate(structure_lines(nodes, grid)):
            axis.plot(line[:, i], line[:, j], color=colour, linewidth=1.4,
                      label=label if k == 0 else None, solid_capstyle="round")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25, linewidth=0.5)


def plot_markers(axes, groups):
    """Scatter the photogrammetry markers, already in the model frame."""
    projections = {"front": (1, 2), "top": (1, 0), "side": (0, 2)}
    for axis, plane in zip(axes, ("front", "top", "side")):
        i, j = projections[plane]
        for index, (name, points) in enumerate(sorted(groups.items())):
            axis.scatter(points[:, i], points[:, j], s=9, c=MARKER, zorder=5,
                         marker="o", linewidths=0,
                         label="measured markers" if index == 0 else None)


def figure_geometry(case, billow_nodes, kfem_nodes, built_nodes, grid, measured,
                    lengths_b, lengths_k, output, scaled_nodes=None,
                    scaled_label=None, markers=None):
    pressure, tip, point = LOAD_CASES[case - 1]
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    plot_geometry(axes, built_nodes, grid, BUILT, "as built")
    plot_geometry(axes, kfem_nodes, grid, KFEM, "kite_fem")
    plot_geometry(axes, billow_nodes, grid, BILLOW, "Billow")
    if scaled_nodes is not None:
        plot_geometry(axes, scaled_nodes, grid, SCALED, scaled_label or "Billow, scaled")
    if markers:
        plot_markers(axes, markers)
    # The kite hangs upside down; plotting z downwards puts it the way it sat
    # in the hangar, and matches the thesis figures.
    for axis, plane in zip(axes, ("front", "top", "side")):
        if plane in ("front", "side"):
            axis.invert_yaxis()

    # The measurement is twelve lengths, not a shape, so the only thing that
    # can be drawn from it is where the trailing-edge tips should have ended up.
    half = 0.5 * measured[9]
    for sign in (-1.0, 1.0):
        axes[0].axvline(sign * half, color=MEASURED, linestyle=":", linewidth=1.0,
                        label="measured span" if sign > 0 else None)
    axes[0].legend(loc="lower right", fontsize=7, framealpha=0.9)
    figure.suptitle(
        f"Load case {case}: {pressure} bar"
        + (f", tip {tip:g} kg" if tip else "")
        + (f", centre {point:g} kg" if point else "")
        + f"   |   span: measured {measured[9]:.2f} m, "
          f"Billow {lengths_b[9]:.2f} m, kite_fem {lengths_k[9]:.2f} m",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=200)
    plt.close(figure)


REGIME_COLOUR = {TAUT: "#2b7bba", WRINKLED: "#f2c14e", SLACK: "#c0392b"}
REGIME_NAME = {TAUT: "taut", WRINKLED: "wrinkled", SLACK: "slack"}


def shade(faces, base, light=(0.35, -0.55, -0.75), ambient=0.45):
    """Lambertian shading per triangle, so the surface reads as a solid object.

    A flat fill hides every fold; with facet lighting the billow between struts
    and the sag of the canopy are visible directly, which is what makes the
    render worth looking at instead of a regime map.
    """
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths < 1e-15, 1.0, lengths)
    direction = np.asarray(light, dtype=float)
    direction /= np.linalg.norm(direction)
    intensity = ambient + (1.0 - ambient) * np.abs(normals @ direction)
    rgb = np.asarray(matplotlib.colors.to_rgb(base))
    return np.clip(intensity[:, None] * rgb[None, :], 0.0, 1.0)


def figure_canopy(case, positions, grid, reference, output, markers=None,
                  pressure=0.15, elev=14.0, azim=-22.0, style="fabric"):
    """The deformed kite in 3D.

    ``style="fabric"`` renders a shaded canopy so the shape can be judged by
    eye against the hangar photographs. ``style="regime"`` colours each triangle
    by its tension-field state instead -- taut, wrinkled or slack -- which is a
    physical prediction Billow makes and a spring net cannot.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    triangles = canopy_triangles(grid)
    faces = positions[triangles]

    regimes = None
    if style == "regime":
        model, _, _ = build_model(reference, grid, pressure=pressure,
                                  canopy_stiffness=5000.0)
        regimes = membrane_regimes(positions, model.element_set("canopy"))["regime"]
        colours = [REGIME_COLOUR.get(int(r), "#999999") for r in regimes]
        edge, width = "#ffffff", 0.15
    else:
        colours = shade(faces, FABRIC)
        edge, width = "none", 0.0

    figure = plt.figure(figsize=(10.0, 5.4))
    axis = figure.add_subplot(111, projection="3d")
    collection = Poly3DCollection(faces, facecolors=colours, edgecolors=edge,
                                  linewidths=width,
                                  alpha=1.0 if style == "fabric" else 0.92)
    axis.add_collection3d(collection)

    for line in structure_lines(positions, grid):
        axis.plot(line[:, 0], line[:, 1], line[:, 2], color=TUBE,
                  linewidth=3.4, solid_capstyle="round", zorder=6)

    if markers:
        pts = np.vstack(list(markers.values()))
        axis.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=11, c=MARKER,
                     depthshade=False, zorder=10, label="measured markers")

    handles = []
    if regimes is not None:
        counts = {k: int((regimes == k).sum()) for k in (TAUT, WRINKLED, SLACK)}
        handles = [plt.Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                              markerfacecolor=REGIME_COLOUR[k],
                              markeredgecolor="none",
                              label=f"{REGIME_NAME[k]} ({counts[k]})")
                   for k in (TAUT, WRINKLED, SLACK) if counts[k]]
    if markers:
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="none",
                                  markersize=5, color=MARKER,
                                  label="measured markers"))
    if handles:
        axis.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    everything = np.vstack([positions[grid.ravel()],
                            *([np.vstack(list(markers.values()))] if markers else [])])
    low, high = everything.min(axis=0), everything.max(axis=0)
    pad = 0.06 * (high - low).max()
    low, high = low - pad, high + pad
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(high[2], low[2])          # z downwards: the kite hangs
    # Box aspect from the true extents, so the kite fills the frame without
    # being distorted -- equal limits would leave it small and edge-on.
    axis.set_box_aspect(tuple(high - low))
    axis.set_xlabel("x (m)", labelpad=2)
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.xaxis.set_major_locator(plt.MaxNLocator(3))
    axis.tick_params(axis="x", labelsize=7, pad=-2)
    axis.view_init(elev=elev, azim=azim)
    axis.set_proj_type("ortho")

    p, tip, point = LOAD_CASES[case - 1]
    axis.set_title(
        f"Load case {case}: {p} bar"
        + (f", tip {tip:g} kg" if tip else "")
        + (f", centre {point:g} kg" if point else "")
        + ("   |   canopy shaded by wrinkling regime" if style == "regime"
           else "   |   solved shape, measured markers overlaid"), fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def figure_lengths(cases, lengths_b, lengths_k, measured, output,
                   lengths_s=None, scaled_label=None):
    figure, axes = plt.subplots(2, 5, figsize=(14.0, 5.4), sharex=True)
    index = np.arange(len(LENGTH_NAMES))
    for column, case in enumerate(cases):
        axis = axes[column // 5, column % 5]
        axis.plot(index, measured[column], "o-", color=MEASURED, markersize=3,
                  linewidth=1.2, label="measured")
        axis.plot(index, lengths_k[column], "s--", color=KFEM, markersize=3,
                  linewidth=1.0, label="kite_fem")
        axis.plot(index, lengths_b[column], "^--", color=BILLOW, markersize=3,
                  linewidth=1.0, label="Billow")
        if lengths_s is not None:
            axis.plot(index, lengths_s[column], "v--", color=SCALED, markersize=3,
                      linewidth=1.0, label=scaled_label or "Billow, scaled")
        axis.set_yscale("log")
        axis.set_title(f"case {case}", fontsize=8)
        axis.set_xticks(index)
        axis.set_xticklabels(LENGTH_NAMES, rotation=90, fontsize=6)
        axis.grid(alpha=0.25, linewidth=0.5)
        if column == 0:
            axis.set_ylabel("length (m)")
            axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def figure_summary(cases, lengths_b, lengths_k, measured, output,
                   lengths_s=None, scaled_label=None):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    width = 0.27
    index = np.arange(len(cases))

    # Shade the centre-load cases: neither code reproduces them, so they should
    # not be read as an ordinary accuracy comparison.
    for axis in axes:
        for position, case in zip(index, cases):
            if int(case) in CENTRE_LOAD_CASES:
                axis.axvspan(position - 0.5, position + 0.5, color="#f2c14e",
                             alpha=0.18, zorder=0, linewidth=0)

    n = 4 if lengths_s is not None else 3
    width = 0.8 / n
    offsets = (np.arange(n) - (n - 1) / 2) * width
    axes[0].bar(index + offsets[0], measured[:, 9], width, color=MEASURED, label="measured")
    axes[0].bar(index + offsets[1], lengths_k[:, 9], width, color=KFEM, label="kite_fem")
    axes[0].bar(index + offsets[2], lengths_b[:, 9], width, color=BILLOW, label="Billow")
    if lengths_s is not None:
        axes[0].bar(index + offsets[3], lengths_s[:, 9], width, color=SCALED,
                    label=scaled_label or "Billow, scaled")
    axes[0].axhline(8.305, color=BUILT, linestyle="--", linewidth=1.0, label="as built")
    axes[0].set_ylabel("span (m)")
    axes[0].set_xlabel("load case")
    axes[0].set_xticks(index)
    axes[0].set_xticklabels(cases)
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.5)

    rel_b = np.abs((lengths_b - measured) / measured).mean(axis=1) * 100
    rel_k = np.abs((lengths_k - measured) / measured).mean(axis=1) * 100
    m = 3 if lengths_s is not None else 2
    bar = 0.8 / m
    shift = (np.arange(m) - (m - 1) / 2) * bar
    axes[1].bar(index + shift[0], rel_k, bar, color=KFEM, label="kite_fem")
    axes[1].bar(index + shift[1], rel_b, bar, color=BILLOW, label="Billow")
    if lengths_s is not None:
        rel_s = np.abs((lengths_s - measured) / measured).mean(axis=1) * 100
        axes[1].bar(index + shift[2], rel_s, bar, color=SCALED,
                    label=scaled_label or "Billow, scaled")
    axes[1].set_ylabel("mean relative error (%)")
    axes[1].set_xlabel("load case")
    axes[1].set_xticks(index)
    axes[1].set_xticklabels(cases)
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.5)

    axes[0].set_title("shaded: centre-load cases -- local cross-section collapse,\noutside beam-element kinematics",
                      fontsize=7, color="#7a5c00")

    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("results/structural/hanging/validation_thesis.npz"))
    parser.add_argument("--canopy-cases", type=int, nargs="*", default=[1, 5],
                        help="cases to render in 3D with the canopy shaded")
    parser.add_argument("--shape", type=Path, default=None,
                        help="shape_comparison.npz, to overlay the markers")
    parser.add_argument("--scaled", type=Path, default=None,
                        help="second Billow run to overlay, e.g. a scaled tube stiffness")
    parser.add_argument("--scaled-label", default=None)
    parser.add_argument("--no-kite-fem", action="store_true",
                        help="omit kite_fem geometry from the projections")
    parser.add_argument("--figures", type=Path,
                        default=Path("results/structural/hanging/figures"))
    arguments = parser.parse_args()

    data = np.load(arguments.results, allow_pickle=True)
    cases = data["cases"]
    positions = data["positions"]
    lengths_b = data["lengths"]
    lengths_k = data["kite_fem"]
    measured = data["measured"]
    built = data["reference_nodes"]

    scaled = None
    scaled_label = arguments.scaled_label
    if arguments.scaled is not None and arguments.scaled.exists():
        extra = np.load(arguments.scaled, allow_pickle=True)
        scaled = {"positions": extra["positions"], "lengths": extra["lengths"]}
        if scaled_label is None:
            factor = float(extra["stiffness_factor"]) if "stiffness_factor" in extra else None
            scaled_label = (f"Billow, $EI\\times${factor:g}" if factor
                            else "Billow, scaled")

    marker_sets: dict = {}
    if arguments.shape is not None and arguments.shape.exists():
        shape = np.load(arguments.shape, allow_pickle=True)
        for key in shape.files:
            if key.startswith("markers_"):
                _, group, case = key.split("_", 2)
                marker_sets.setdefault(int(case), {})[group] = shape[key]

    reference = load_reference()
    grid = canopy_grid(reference.nodes)
    arguments.figures.mkdir(parents=True, exist_ok=True)

    for column, case in enumerate(cases):
        kfem_nodes = built if arguments.no_kite_fem else kite_fem_state(int(case))
        figure_geometry(int(case), positions[column], kfem_nodes, built, grid,
                        measured[column], lengths_b[column], lengths_k[column],
                        arguments.figures / f"geometry_{case}.png",
                        scaled_nodes=None if scaled is None else scaled["positions"][column],
                        scaled_label=scaled_label,
                        markers=marker_sets.get(int(case)))

    for case in arguments.canopy_cases:
        if case not in list(cases):
            continue
        column = list(cases).index(case)
        pressure = LOAD_CASES[case - 1][0]
        source = scaled["positions"][column] if scaled is not None else positions[column]
        figure_canopy(int(case), source, grid, reference,
                      arguments.figures / f"canopy_{case}.png",
                      markers=marker_sets.get(int(case)), pressure=pressure,
                      style="fabric")
        figure_canopy(int(case), source, grid, reference,
                      arguments.figures / f"regime_{case}.png",
                      markers=None, pressure=pressure, style="regime")

    lengths_s = None if scaled is None else scaled["lengths"]
    figure_lengths(cases, lengths_b, lengths_k, measured,
                   arguments.figures / "lengths.png",
                   lengths_s=lengths_s, scaled_label=scaled_label)
    figure_summary(cases, lengths_b, lengths_k, measured,
                   arguments.figures / "summary.png",
                   lengths_s=lengths_s, scaled_label=scaled_label)
    print(f"  -> {arguments.figures}")


if __name__ == "__main__":
    main()

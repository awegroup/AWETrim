# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Demo cases for the standalone minimum-energy structural model.

Four cases, each a figure, each answering one question about whether a richer
structural model still fits inside an NLP:

1. ``wrinkling``  -- fabric clamped into a frame smaller than itself. Does the
   relaxed (tension-field) energy find the wrinkled state, and what would the
   unrelaxed law have done instead?
2. ``cantilever`` -- a slender tube bent to large deflection. Does the
   geometrically exact Timoshenko beam reproduce linear theory in the small and
   stiffen correctly in the large?
3. ``sail``       -- a batten-stiffened sail: inflatable-tube spar + wrinkling
   canopy + bridle cables, all in one energy. Beams, membranes, cables and
   pulleys solved together.
4. ``scaling``    -- build time, solve time and iteration count against problem
   size, from tens to tens of thousands of DOF.
5. ``canopy_model`` -- how the canopy is modelled: the noncompressive spring net
   that ``kite_fem``'s ``FEM_canopy_section.py`` uses, against a CST membrane, on
   identical meshes. Both are built here with this package, so the only
   difference is the constitutive model.

Run from the project root::

    python scripts/structural/run_demo_cases.py
    python scripts/structural/run_demo_cases.py --case sail --show
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
    BeamSection,
    build_beam_elements,
    build_cable_elements,
    build_membrane_elements,
    initial_frames_from_polyline,
    membrane_regimes,
)

OUTPUT_DIR = Path("results/structural/demo")

FABRIC_MODULUS = 5.0e8  # Pa
FABRIC_POISSON = 0.3
FABRIC_THICKNESS = 3.0e-4  # m

REGIME_COLORS = ListedColormap(
    [PALETTE["Sky Blue"], PALETTE["Orange"], PALETTE["Vermillion"]]
)
REGIME_LABELS = {SLACK: "slack", WRINKLED: "wrinkled", TAUT: "taut"}


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------


def square_mesh(n_side: int, size: float = 1.0):
    """Flat square grid, its triangles, and the indices of its edge nodes."""
    axis = np.linspace(0.0, size, n_side)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])

    triangles = []
    for i in range(n_side - 1):
        for j in range(n_side - 1):
            a, b = i * n_side + j, (i + 1) * n_side + j
            triangles.append([a, b, b + 1])
            triangles.append([a, b + 1, a + 1])

    on_edge = (
        (grid_x.ravel() == 0.0)
        | (grid_y.ravel() == 0.0)
        | (grid_x.ravel() == size)
        | (grid_y.ravel() == size)
    )
    return nodes, np.array(triangles), np.flatnonzero(on_edge)


def draw_surface(axes, positions, triangles, facevalues, cmap, norm, alpha=0.95):
    """Draw a triangulated surface coloured per element."""
    polygons = Poly3DCollection(
        [positions[triangle] for triangle in triangles],
        alpha=alpha,
        edgecolor="0.35",
        linewidths=0.15,
    )
    polygons.set_facecolor(cmap(norm(facevalues)))
    axes.add_collection3d(polygons)
    return polygons


def equalize(axes, positions, pad=0.05, z_exaggeration=1.0):
    """Frame a 3-D axes around the deformed shape.

    ``z_exaggeration`` > 1 stretches the vertical box so a shallow billow stays
    readable; it changes only the drawing, never the solution.
    """
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    centre = 0.5 * (lower + upper)
    span = upper - lower
    radius = 0.5 * max(span[:2].max(), 1e-6) * (1.0 + pad)
    z_radius = max(0.5 * span[2] * (1.0 + pad), radius / z_exaggeration, 1e-6)

    axes.set_xlim(centre[0] - radius, centre[0] + radius)
    axes.set_ylim(centre[1] - radius, centre[1] + radius)
    axes.set_zlim(centre[2] - z_radius, centre[2] + z_radius)
    axes.set_box_aspect((1.0, 1.0, min(1.0, z_radius / radius) * z_exaggeration))
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m)")


# ---------------------------------------------------------------------------
# Case 1 -- wrinkling fabric
# ---------------------------------------------------------------------------


def solve_loose_panel(n_side=21, slack=0.06, total_load=250.0, wrinkling=True):
    """Oversized fabric clamped into a smaller frame, then pushed sideways."""
    nodes, triangles, edge_nodes = square_mesh(n_side)
    canopy = build_membrane_elements(
        nodes * (1.0 + slack),
        triangles,
        FABRIC_THICKNESS,
        FABRIC_MODULUS,
        FABRIC_POISSON,
        wrinkling=wrinkling,
    )
    model = StructuralModel(nodes, [canopy], fixed_translation_nodes=edge_nodes)

    interior = np.setdiff1d(np.arange(len(nodes)), edge_nodes)
    forces = np.zeros_like(nodes)
    forces[interior, 2] = total_load / len(interior)

    started = time.perf_counter()
    solution = MinimumEnergySolver(model, tolerance=1e-10).solve(forces)
    return solution, model, triangles, time.perf_counter() - started


def case_wrinkling():
    figure = plt.figure(figsize=(12.5, 5.4))
    figure.suptitle(
        "Fabric clamped into a frame 6% smaller than itself, then pushed out of plane",
        fontsize=12,
    )

    for column, wrinkling in enumerate([True, False]):
        solution, model, triangles, elapsed = solve_loose_panel(wrinkling=wrinkling)
        positions = solution.state.positions
        diagnostic = membrane_regimes(positions, model.element_set("canopy"))

        axes = figure.add_subplot(1, 2, column + 1, projection="3d")
        draw_surface(
            axes,
            positions,
            triangles,
            diagnostic["regime"],
            REGIME_COLORS,
            Normalize(-0.5, 2.5),
        )
        equalize(axes, positions, z_exaggeration=2.2)
        counts = {
            name: int((diagnostic["regime"] == code).sum())
            for code, name in REGIME_LABELS.items()
        }
        law = "relaxed (tension field)" if wrinkling else "unrelaxed SVK"
        axes.set_title(
            f"{law}\n"
            f"peak billow {positions[:, 2].max() * 1e3:.1f} mm, "
            f"U = {solution.strain_energy:.3f} J\n"
            f"{counts} - {solution.iterations} iters, {elapsed:.2f} s",
            fontsize=9,
        )
        axes.view_init(elev=24, azim=-58)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                   color=REGIME_COLORS(Normalize(-0.5, 2.5)(code)), label=name)
        for code, name in REGIME_LABELS.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


# ---------------------------------------------------------------------------
# Case 2 -- large-deflection cantilever
# ---------------------------------------------------------------------------


def cantilever(n_elements=24, length=1.0, diameter=0.02, wall=1.0e-3):
    modulus, shear_modulus = 70.0e9, 26.0e9
    nodes = np.column_stack(
        [np.linspace(0.0, length, n_elements + 1), np.zeros(n_elements + 1),
         np.zeros(n_elements + 1)]
    )
    frames = initial_frames_from_polyline(nodes)
    section = BeamSection.from_tube(diameter, wall, modulus, shear_modulus)
    connectivity = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    beams = build_beam_elements(nodes, connectivity, section, frames, name="spar")
    model = StructuralModel(
        nodes,
        [beams],
        node_frames=frames,
        fixed_translation_nodes=[0],
        fixed_rotation_nodes=[0],
    )
    second_moment = np.pi * (0.5 * diameter) ** 3 * wall
    return model, modulus * second_moment, length


def case_cantilever():
    model, bending_stiffness, length = cantilever()
    solver = MinimumEnergySolver(model)
    loads = np.linspace(0.0, 900.0, 19)[1:]

    shapes, tips, iterations = [], [], []
    state = None
    for load in loads:
        forces = np.zeros((model.n_nodes, 3))
        forces[-1, 2] = load
        solution = solver.solve(forces, state=state)
        assert solution.converged, f"{load} N: {solution.status}"
        state = solution.state  # continuation: each load warm-starts the next
        shapes.append(solution.state.positions.copy())
        tips.append(solution.state.positions[-1, 2])
        iterations.append(solution.iterations)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colours = plt.cm.viridis(np.linspace(0.1, 0.95, len(shapes)))
    for shape, colour in zip(shapes, colours):
        left.plot(shape[:, 0], shape[:, 2], color=colour, linewidth=1.4)
    left.plot([0, length], [0, 0], color="0.6", linestyle=":", linewidth=1)
    left.set_aspect("equal")
    left.set_xlabel("x (m)")
    left.set_ylabel("z (m)")
    left.set_title(
        "Deflected shapes, tip load 50 to 900 N\n"
        "geometrically exact: the beam shortens in x as it bends",
        fontsize=10,
    )

    linear = loads * length**3 / (3.0 * bending_stiffness)
    right.plot(loads, np.array(tips) / length, color=PALETTE["Black"],
               marker="o", markersize=3.5, label="geometrically exact")
    right.plot(loads, linear / length, color=PALETTE["Vermillion"],
               linestyle="--", label=r"linear Euler $PL^3/3EI$")
    right.set_xlabel("tip load (N)")
    right.set_ylabel("tip deflection / length (-)")
    right.legend(frameon=False)
    right.grid(alpha=0.3)
    softening = 100.0 * (1.0 - tips[-1] / linear[-1])
    right.set_title(
        f"Linear theory overshoots by {softening:.0f}% at the largest load\n"
        f"median {int(np.median(iterations))} IPOPT iterations per load step "
        "(warm started)",
        fontsize=10,
    )
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Case 3 -- batten-stiffened sail: beams + fabric + cables in one energy
# ---------------------------------------------------------------------------


def build_sail(n_span=13, n_chord=9, span=3.0, chord=1.2):
    """A spar along the leading edge, fabric behind it, bridles below.

    Deliberately kite-shaped without being the LEI-V3: one inflatable-tube spar
    carrying bending and torsion, a wrinkling canopy hanging off it, and three
    bridle cables down to a fixed attachment point.
    """
    span_axis = np.linspace(-0.5 * span, 0.5 * span, n_span)
    chord_axis = np.linspace(0.0, chord, n_chord)
    # Slight leading-edge sweep and arc, so the spar is curved as built.
    grid_y, grid_x = np.meshgrid(span_axis, chord_axis, indexing="ij")
    arc = -0.25 * (grid_y / (0.5 * span)) ** 2
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel(), arc.ravel()])

    def index(i, j):
        return i * n_chord + j

    triangles = []
    for i in range(n_span - 1):
        for j in range(n_chord - 1):
            triangles.append([index(i, j), index(i + 1, j), index(i + 1, j + 1)])
            triangles.append([index(i, j), index(i + 1, j + 1), index(i, j + 1)])

    spar_nodes = np.array([index(i, 0) for i in range(n_span)])
    frames = np.tile(np.eye(3), (len(nodes), 1, 1))
    frames[spar_nodes] = initial_frames_from_polyline(nodes[spar_nodes])

    section = BeamSection.from_tube(0.12, 4.0e-4, 3.0e9, 1.1e9)
    spar = build_beam_elements(
        nodes,
        np.column_stack([spar_nodes[:-1], spar_nodes[1:]]),
        section,
        frames,
        name="spar",
    )
    canopy = build_membrane_elements(
        nodes, triangles, FABRIC_THICKNESS, FABRIC_MODULUS, FABRIC_POISSON, name="canopy"
    )

    # Bridle: three lines from spanwise stations down to one anchor node.
    anchor = len(nodes)
    nodes = np.vstack([nodes, [0.5 * chord, 0.0, -2.5]])
    frames = np.vstack([frames, np.eye(3)[None]])
    attachments = [index(2, n_chord - 1), index(n_span // 2, n_chord - 1),
                   index(n_span - 3, n_chord - 1)]
    bridle_connectivity = np.array([[node, anchor] for node in attachments])
    rest_lengths = np.linalg.norm(nodes[bridle_connectivity[:, 0]] - nodes[anchor], axis=1)
    bridles = build_cable_elements(
        bridle_connectivity, rest_lengths * 0.995, np.full(3, 4.0e5), name="bridles"
    )

    # Trailing-edge hem: a real sail is not a raw membrane edge. Without it the
    # free edge festoons into scallops between bridle attachments -- correct for
    # a hemless membrane, but not what a sail does.
    hem_nodes = np.array([index(i, n_chord - 1) for i in range(n_span)])
    hem_connectivity = np.column_stack([hem_nodes[:-1], hem_nodes[1:]])
    hem_lengths = np.linalg.norm(
        nodes[hem_connectivity[:, 1]] - nodes[hem_connectivity[:, 0]], axis=1
    )
    hem = build_cable_elements(
        hem_connectivity, hem_lengths, np.full(len(hem_lengths), 2.0e5), name="hem"
    )

    model = StructuralModel(
        nodes,
        [spar, canopy, bridles, hem],
        node_frames=frames,
        fixed_translation_nodes=[anchor],
        fixed_rotation_nodes=[int(spar_nodes[n_span // 2])],
    )
    return model, triangles, spar_nodes, anchor


def case_sail():
    model, triangles, spar_nodes, anchor = build_sail()
    canopy_nodes = np.arange(anchor)

    # Kite-like: aero pushes the sail away from the anchor, so the bridles go
    # taut and carry the load down to it. Loading the other way would just drop
    # the sail onto a slack bridle and nothing would resist.
    forces = np.zeros((model.n_nodes, 3))
    forces[canopy_nodes, 2] = 1200.0 / len(canopy_nodes)
    forces[canopy_nodes, 0] = 220.0 / len(canopy_nodes)

    started = time.perf_counter()
    solution = MinimumEnergySolver(model, tolerance=1e-9).solve(forces)
    elapsed = time.perf_counter() - started

    positions = solution.state.positions
    diagnostic = membrane_regimes(positions, model.element_set("canopy"))

    figure = plt.figure(figsize=(12.5, 5.6))
    figure.suptitle(
        "Batten-stiffened sail: Timoshenko spar + wrinkling canopy + bridle cables, "
        "one energy, one solve",
        fontsize=12,
    )

    left = figure.add_subplot(1, 2, 1, projection="3d")
    draw_surface(left, positions, triangles, diagnostic["regime"],
                 REGIME_COLORS, Normalize(-0.5, 2.5), alpha=0.9)
    left.plot(*positions[spar_nodes].T, color=PALETTE["Black"], linewidth=3.0,
              label="spar (beam)")
    for node in model.element_set("bridles").connectivity[:, 0]:
        segment = positions[[node, anchor]]
        left.plot(*segment.T, color=PALETTE["Bluish Green"], linewidth=1.4)
    left.plot([], [], color=PALETTE["Bluish Green"], linewidth=1.4, label="bridles")
    hem_nodes = model.element_set("hem").connectivity
    for node_a, node_b in hem_nodes:
        left.plot(*positions[[node_a, node_b]].T, color=PALETTE["Blue"], linewidth=1.8)
    left.plot([], [], color=PALETTE["Blue"], linewidth=1.8, label="trailing-edge hem")
    left.scatter(*positions[anchor], color=PALETTE["Vermillion"], s=45, label="anchor")
    equalize(left, positions, z_exaggeration=1.6)
    left.view_init(elev=18, azim=-64)
    left.legend(loc="upper left", frameon=False, fontsize=8)
    left.set_title(
        f"{model.layout.n_dof} DOF "
        f"({model.layout.n_rotational_nodes} nodes carry a frame)\n"
        f"{solution.iterations} iterations, {elapsed:.2f} s, "
        f"residual {solution.residual_norm:.1e} N",
        fontsize=9,
    )

    right = figure.add_subplot(1, 2, 2, projection="3d")
    strain_ceiling = float(np.percentile(diagnostic["principal_1"], 97))
    draw_surface(right, positions, triangles, diagnostic["principal_1"],
                 plt.cm.magma, Normalize(0.0, max(strain_ceiling, 1e-9)))
    right.plot(*model.nodes[spar_nodes].T, color="0.55", linewidth=1.6,
               linestyle="--", label="undeformed spar")
    right.plot(*positions[spar_nodes].T, color=PALETTE["Black"], linewidth=3.0,
               label="deformed spar")
    equalize(right, positions, z_exaggeration=1.6)
    right.view_init(elev=18, azim=-64)
    right.legend(loc="upper left", frameon=False, fontsize=8)
    tip_droop = positions[spar_nodes[0], 2] - model.nodes[spar_nodes[0], 2]
    right.set_title(
        f"Major principal strain; spar tip moves {tip_droop * 1e3:.0f} mm\n"
        f"strain energy {solution.strain_energy:.1f} J",
        fontsize=9,
    )

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                   color=REGIME_COLORS(Normalize(-0.5, 2.5)(code)), label=name)
        for code, name in REGIME_LABELS.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


# ---------------------------------------------------------------------------
# Case 4 -- does it still scale?
# ---------------------------------------------------------------------------


def case_scaling(sides=(9, 13, 17, 23, 31, 41, 55)):
    build_times, solve_times, dofs, elements, iterations = [], [], [], [], []

    for n_side in sides:
        nodes, triangles, edge_nodes = square_mesh(n_side)
        canopy = build_membrane_elements(
            nodes, triangles, FABRIC_THICKNESS, FABRIC_MODULUS, FABRIC_POISSON
        )
        model = StructuralModel(nodes, [canopy], fixed_translation_nodes=edge_nodes)

        started = time.perf_counter()
        solver = MinimumEnergySolver(model, tolerance=1e-9)
        build_times.append(time.perf_counter() - started)

        interior = np.setdiff1d(np.arange(len(nodes)), edge_nodes)
        forces = np.zeros_like(nodes)
        forces[interior, 2] = 200.0 / len(interior)

        started = time.perf_counter()
        solution = solver.solve(forces)
        solve_times.append(time.perf_counter() - started)

        dofs.append(model.layout.n_dof)
        elements.append(len(triangles))
        iterations.append(solution.iterations)
        print(
            f"  {len(triangles):6d} triangles  {model.layout.n_dof:6d} DOF  "
            f"build {build_times[-1]:6.2f} s  solve {solve_times[-1]:6.2f} s  "
            f"{solution.iterations:3d} iters  converged={solution.converged}"
        )

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    left.loglog(dofs, build_times, marker="o", color=PALETTE["Blue"],
                label="build graph + IPOPT setup")
    left.loglog(dofs, solve_times, marker="s", color=PALETTE["Vermillion"],
                label="solve")
    reference = np.array(dofs, dtype=float)
    left.loglog(reference, solve_times[0] * reference / reference[0],
                color="0.6", linestyle=":", label="linear in DOF")
    left.set_xlabel("degrees of freedom (-)")
    left.set_ylabel("wall time (s)")
    left.legend(frameon=False, fontsize=9)
    left.grid(alpha=0.3, which="both")
    left.set_title(
        "Mapped element kernels: the graph is built once per element TYPE,\n"
        "so build cost tracks the solve rather than the element count",
        fontsize=10,
    )

    right.semilogx(dofs, iterations, marker="o", color=PALETTE["Black"])
    right.set_xlabel("degrees of freedom (-)")
    right.set_ylabel("IPOPT iterations (-)")
    right.grid(alpha=0.3, which="both")
    right.set_title(
        "Iteration count is nearly mesh independent\n"
        f"{elements[0]} to {elements[-1]} triangles",
        fontsize=10,
    )
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Case 5 -- spring net vs membrane: how should the canopy be modelled?
# ---------------------------------------------------------------------------


def spring_net(nodes, n_side, stiffness, name="net"):
    """Orthogonal noncompressive spring net -- the kite_fem canopy recipe.

    ``kite_fem/examples/FEM_canopy_section.py`` links each node to its right and
    upper neighbour with a ``noncompressive`` spring of fixed stiffness. There
    are no diagonals, which is why that example carries a commented-out block of
    "fictional springs constraining the canopy from collapse".
    """
    connectivity, rest_lengths = [], []
    for i in range(n_side):
        for j in range(n_side):
            here = i * n_side + j
            for neighbour in ([here + n_side] if i < n_side - 1 else []) + (
                [here + 1] if j < n_side - 1 else []
            ):
                connectivity.append([here, neighbour])
                rest_lengths.append(
                    float(np.linalg.norm(nodes[neighbour] - nodes[here]))
                )
    return build_cable_elements(
        connectivity, rest_lengths, np.full(len(rest_lengths), stiffness),
        name=name, tension_only=True,
    )


def canopy_energy(model, positions) -> float:
    """Strain energy of an imposed configuration -- no solve, no equilibrium."""
    energy = MinimumEnergySolver(model).energy
    parameters = energy.pack_parameters(
        model.initial_state(), model.element_sets,
        np.zeros_like(positions), None, positions,
    )
    return float(energy.total_energy(energy.pack_unknowns(positions), parameters))


def case_canopy_model():
    sides = (5, 7, 9, 13, 17, 21)
    sheet_stiffness = FABRIC_MODULUS * FABRIC_THICKNESS  # N/m, k = E t for a square grid

    inflation = {"membrane": [], "net (k = E t)": [], "net (k = 5e4, kite_fem)": []}
    biaxial = {"membrane": [], "net (k = E t)": []}

    for n_side in sides:
        nodes, triangles, edge_nodes = square_mesh(n_side)
        interior = np.setdiff1d(np.arange(len(nodes)), edge_nodes)
        forces = np.zeros_like(nodes)
        forces[interior, 2] = 200.0 / len(interior)
        centre = (n_side * n_side - 1) // 2

        variants = {
            "membrane": build_membrane_elements(
                nodes, triangles, FABRIC_THICKNESS, FABRIC_MODULUS, FABRIC_POISSON
            ),
            "net (k = E t)": spring_net(nodes, n_side, sheet_stiffness),
            "net (k = 5e4, kite_fem)": spring_net(nodes, n_side, 5.0e4),
        }
        for label, elements in variants.items():
            model = StructuralModel(nodes, [elements], fixed_translation_nodes=edge_nodes)
            solution = MinimumEnergySolver(model, tolerance=1e-9).solve(forces)
            inflation[label].append(solution.state.positions[centre, 2])

            if label in biaxial:
                # Equibiaxial stretch, imposed rather than solved.
                stretch = 1.01
                biaxial[label].append(canopy_energy(model, nodes * stretch))

    # Simple shear x -> x + gamma y, at the finest mesh.
    nodes, triangles, edge_nodes = square_mesh(sides[-1])
    shear_models = {
        "membrane": StructuralModel(
            nodes,
            [build_membrane_elements(
                nodes, triangles, FABRIC_THICKNESS, FABRIC_MODULUS, FABRIC_POISSON
            )],
        ),
        "net (k = E t)": StructuralModel(
            nodes, [spring_net(nodes, sides[-1], sheet_stiffness)]
        ),
    }
    gammas = np.logspace(-3, -1, 9)
    shear = {label: [] for label in shear_models}
    for label, model in shear_models.items():
        for gamma in gammas:
            sheared = nodes.copy()
            sheared[:, 0] += gamma * nodes[:, 1]
            shear[label].append(canopy_energy(model, sheared))

    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(15.0, 4.6))
    figure.suptitle(
        "How to model the canopy: noncompressive spring net (the kite_fem recipe) "
        "vs CST membrane",
        fontsize=12,
    )

    styles = {
        "membrane": (PALETTE["Vermillion"], "o", "-"),
        "net (k = E t)": (PALETTE["Blue"], "s", "--"),
        "net (k = 5e4, kite_fem)": (PALETTE["Sky Blue"], "^", ":"),
    }
    for label, values in inflation.items():
        colour, marker, style = styles[label]
        left.plot(sides, np.array(values) * 1e3, color=colour, marker=marker,
                  linestyle=style, label=label)
    left.set_xlabel("nodes per side (-)")
    left.set_ylabel("centre deflection (mm)")
    left.legend(frameon=False, fontsize=8)
    left.grid(alpha=0.3)
    gap = 100.0 * (inflation["net (k = E t)"][-1] / inflation["membrane"][-1] - 1.0)
    left.set_title(
        "Clamped panel, uniform out-of-plane load\n"
        f"both converge, to answers {gap:+.0f}% apart",
        fontsize=10,
    )

    ratio = [n / m for n, m in zip(biaxial["net (k = E t)"], biaxial["membrane"])]
    middle.plot(sides, ratio, color=PALETTE["Blue"], marker="s")
    middle.axhline(1.0 - FABRIC_POISSON, color=PALETTE["Black"], linestyle="--",
                   label=r"$1-\nu$")
    middle.set_xlabel("nodes per side (-)")
    middle.set_ylabel("net energy / membrane energy (-)")
    middle.set_ylim(0.0, 1.1)
    middle.legend(frameon=False, fontsize=9)
    middle.grid(alpha=0.3)
    middle.set_title(
        "Equibiaxial stretch (1%)\n"
        r"the net misses Poisson coupling: too soft by exactly $1-\nu$",
        fontsize=10,
    )

    for label, values in shear.items():
        colour, marker, style = styles[label]
        right.loglog(gammas, np.maximum(values, 1e-30), color=colour, marker=marker,
                     linestyle=style, label=label)
    slopes = {
        label: np.polyfit(np.log(gammas), np.log(np.maximum(values, 1e-30)), 1)[0]
        for label, values in shear.items()
    }
    right.set_xlabel(r"shear $\gamma$ (-)")
    right.set_ylabel("strain energy (J)")
    right.legend(frameon=False, fontsize=8)
    right.grid(alpha=0.3, which="both")
    right.set_title(
        "In-plane simple shear\n"
        rf"membrane $\propto\gamma^{{{slopes['membrane']:.1f}}}$, "
        rf"net $\propto\gamma^{{{slopes['net (k = E t)']:.1f}}}$ "
        "- no shear stiffness at leading order",
        fontsize=10,
    )

    print(f"  inflation gap at finest mesh: {gap:+.1f}%")
    print(f"  biaxial energy ratio: {ratio[-1]:.4f} (1-nu = {1 - FABRIC_POISSON:.2f})")
    print(f"  shear energy slopes: {slopes}")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


CASES = {
    "wrinkling": case_wrinkling,
    "canopy_model": case_canopy_model,
    "cantilever": case_cantilever,
    "sail": case_sail,
    "scaling": case_scaling,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), action="append",
                        help="run only these cases (default: all)")
    parser.add_argument("--show", action="store_true", help="open the figures")
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

# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Compare the solved kite shape against the 3D photogrammetry markers.

The twelve published lengths are only six independent numbers, with no height
and no chordwise measure among them, so they cannot decide whether a model has
the right *shape*. Haanen's marker clouds can: 62--76 three-dimensional points
per load case on the leading edge, the eight struts and the canopy.

    https://github.com/pimjhaanen/photogrammetry_thesis

Alignment
---------
The markers are in the stereo rig's frame, so model and measurement must be
brought together before anything can be compared. Only a *rigid* transform is
allowed --- rotation and translation, never scale --- because scale is exactly
the thing under test.

Correspondence comes from structure, not from fitting: measured ``strut0``
through ``strut7`` are ordered across the span and their first marker sits at
the leading edge, so their roots pair with the model's eight strut leading-edge
nodes. That gives eight anchors for an initial Kabsch fit, tried in both
spanwise orderings because the kite is near-symmetric and the marker numbering
direction is not documented. The fit is then refined by ICP against the model
*curves* (each measured marker to the nearest point on its own polyline), which
removes the assumption that a marker sits exactly on a model node.

What is reported
----------------
Point-to-curve RMS per marker group. This is the honest error measure for a
shape: it does not require a marker to correspond to a node, and it cannot be
reduced by sliding markers along the structure.

    python scripts/structural/run_hanging_shape_comparison.py \\
        --markers <path to>/static_test_output
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from hanging_kite import canopy_grid, load_reference

#: Measured file stem for each of the ten load cases, in case order.
CASE_FILES = ("P1_S", "P1_PL1", "P1_PL2", "P1_TL1", "P1_TL2",
              "P2_S", "P2_PL1", "P2_PL2", "P2_TL1", "P2_TL2")

#: The eight real struts (the model's two tip closures are not instrumented).
STRUT_LE_NODES = (7, 13, 19, 25, 33, 39, 45, 51)


def load_markers(directory: Path, stem: str) -> dict[str, np.ndarray]:
    """Marker positions by group, each ordered by ``idx_in_group``."""
    rows = list(csv.DictReader(open(directory / f"{stem}.csv")))
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(
            (int(row["idx_in_group"]),
             float(row["x"]), float(row["y"]), float(row["z"]))
        )
    return {name: np.array([p[1:] for p in sorted(points)])
            for name, points in groups.items()}


def model_curves(positions: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    """Leading-edge and strut polylines from a solved configuration."""
    sections = {int(grid[row, 0]): grid[row] for row in range(len(grid))}
    curves = {"LE": positions[list(range(1, 58, 2))]}
    for index, node in enumerate(STRUT_LE_NODES):
        curves[f"strut{index}"] = positions[sections[node]]
    return curves


def curve_radii(reference, grid) -> dict[str, np.ndarray]:
    """Tube radius at each polyline node.

    The markers are stuck to the outside of the tubes while the model polyline
    is the tube *axis*, so a raw point-to-curve distance carries a systematic
    offset of about one radius -- 55 to 100 mm on this kite, which
    is most of the residual floor. Comparing to the surface removes it.

    Radii come from the beam elements; a node takes the mean of the elements
    meeting there, and nodes with no beam (canopy interior of a strut section)
    inherit the nearest beam radius along the section.
    """
    diameter = {}
    for (node_a, node_b), d in zip(reference.beams, reference.beam_diameter):
        diameter.setdefault(int(node_a), []).append(d)
        diameter.setdefault(int(node_b), []).append(d)
    radius_of = {n: 0.5 * float(np.mean(v)) for n, v in diameter.items()}

    sections = {int(grid[row, 0]): grid[row] for row in range(len(grid))}
    radii = {}

    leading = list(range(1, 58, 2))
    radii["LE"] = np.array([radius_of.get(n, 0.0) for n in leading])
    for index, node in enumerate(STRUT_LE_NODES):
        section = sections[node]
        values = np.array([radius_of.get(int(n), np.nan) for n in section])
        # carry the last known radius aft over any node without a beam
        known = np.where(~np.isnan(values))[0]
        if len(known):
            for j in range(len(values)):
                if np.isnan(values[j]):
                    values[j] = values[known[np.argmin(np.abs(known - j))]]
        else:
            values[:] = 0.0
        radii[f"strut{index}"] = values
    return radii


def project_to_polyline(points: np.ndarray, polyline: np.ndarray,
                       radii: np.ndarray | None = None):
    """Closest point on a polyline, and optionally the radius there."""
    start, end = polyline[:-1], polyline[1:]
    segment = end - start
    length2 = np.einsum("ij,ij->i", segment, segment)
    length2 = np.where(length2 < 1e-15, 1e-15, length2)

    delta = points[:, None, :] - start[None, :, :]
    t = np.einsum("ijk,jk->ij", delta, segment) / length2[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = start[None, :, :] + t[:, :, None] * segment[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - closest, axis=2)
    pick = np.argmin(distance, axis=1)
    nearest = closest[np.arange(len(points)), pick]
    if radii is None:
        return nearest
    edge = 0.5 * (radii[:-1] + radii[1:])
    return nearest, edge[pick]


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform taking ``source`` onto ``target``; no scaling."""
    centre_s, centre_t = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - centre_s).T @ (target - centre_t)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])          # never allow a reflection
    rotation = vt.T @ correction @ u.T
    return rotation, centre_t - rotation @ centre_s


def align(markers, curves, radii, *, reverse: bool, iterations: int = 60):
    """Kabsch on the strut roots, then ICP against the model curves."""
    anchors_measured, anchors_model = [], []
    order = range(7, -1, -1) if reverse else range(8)
    for measured_index, model_index in enumerate(order):
        name = f"strut{measured_index}"
        if name in markers and len(markers[name]):
            anchors_measured.append(markers[name][0])
            anchors_model.append(curves[f"strut{model_index}"][0])
    rotation, translation = kabsch(np.array(anchors_model),
                                   np.array(anchors_measured))

    pairs = [(markers["LE"], curves["LE"])] if "LE" in markers else []
    for measured_index, model_index in enumerate(order):
        name = f"strut{measured_index}"
        if name in markers and len(markers[name]):
            pairs.append((markers[name], curves[f"strut{model_index}"]))

    for _ in range(iterations):
        moved, targets = [], []
        for measured, curve in pairs:
            placed = curve @ rotation.T + translation
            moved.append(project_to_polyline(measured, placed))
            targets.append(measured)
        source = np.vstack(moved)
        # Bring the projected model points back to model frame before refitting.
        source = (source - translation) @ rotation
        rotation, translation = kabsch(source, np.vstack(targets))

    names = (["LE"] if "LE" in markers else []) + [
        f"strut{i}" for i in range(8) if f"strut{i}" in markers]
    model_names = ["LE"] if "LE" in markers else []
    model_names += [f"strut{m}" for i, m in enumerate(order)
                    if f"strut{i}" in markers]

    residual = {}
    for name, model_name, (measured, curve) in zip(names, model_names, pairs):
        placed = curve @ rotation.T + translation
        closest, edge = project_to_polyline(measured, placed, radii[model_name])
        # Distance to the tube SURFACE: the markers are on the outside.
        residual[name] = np.abs(np.linalg.norm(measured - closest, axis=1) - edge)
    return rotation, translation, residual


def refine_on_leading_edge(markers, curves, rotation, translation,
                           iterations: int = 40):
    """Refine an existing rigid fit using the leading-edge markers only.

    Used for the *figures*: registering on the leading edge puts the two leading
    edges on top of each other, so the remaining disagreement shows up where it
    physically is -- in the struts and the canopy -- instead of being spread
    over the whole structure by a global fit.

    It only ever *refines*. Seeding correspondence by marker index instead was
    tried and fails: the measured numbering does not always run the same way
    along the span, so the initial pairing can be reversed and ICP then locks
    onto a badly wrong pose. Starting from the converged all-curve alignment
    removes that failure mode, and the result is rejected if it drifts far from
    where it started.
    """
    measured, curve = markers["LE"], curves["LE"]
    start = (rotation.copy(), translation.copy())
    for _ in range(iterations):
        placed = curve @ rotation.T + translation
        projected = project_to_polyline(measured, placed)
        rotation, translation = kabsch((projected - translation) @ rotation, measured)

    # Guard: a good refinement moves the model a little, not metres.
    drift = float(np.linalg.norm(translation - start[1]))
    placed = curve @ rotation.T + translation
    residual = np.linalg.norm(
        measured - project_to_polyline(measured, placed), axis=1).mean()
    placed0 = curve @ start[0].T + start[1]
    residual0 = np.linalg.norm(
        measured - project_to_polyline(measured, placed0), axis=1).mean()
    if drift > 1.0 or residual > residual0:
        return start
    return rotation, translation


def compare(positions, grid, markers, radii):
    """Best of the two spanwise orderings, by overall RMS."""
    curves = model_curves(positions, grid)
    best = None
    for reverse in (False, True):
        rotation, translation, residual = align(markers, curves, radii,
                                                reverse=reverse)
        everything = np.concatenate(list(residual.values()))
        rms = float(np.sqrt(np.mean(everything**2)))
        if best is None or rms < best[0]:
            best = (rms, reverse, rotation, translation, residual)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=Path, required=True,
                        help="directory holding P1_S.csv etc.")
    parser.add_argument("--results", type=Path, nargs="+", default=[
        Path("results/structural/hanging/validation_aikept.npz"),
        Path("results/structural/hanging/validation_scaled.npz"),
    ])
    parser.add_argument("--labels", nargs="+", default=["EI x1.0", "EI x0.36"])
    parser.add_argument("--output", type=Path,
                        default=Path("results/structural/hanging/shape_comparison.npz"))
    arguments = parser.parse_args()

    reference = load_reference()
    grid = canopy_grid(reference.nodes)
    radii = curve_radii(reference, grid)
    datasets = [np.load(path, allow_pickle=True) for path in arguments.results]

    print(f"{'case':>5} {'file':>8} {'markers':>8} " +
          " ".join(f"{label:>12}" for label in arguments.labels) +
          f" {'as built':>10}")
    table = []
    aligned: dict = {}
    for index, stem in enumerate(CASE_FILES):
        path = arguments.markers / f"{stem}.csv"
        if not path.exists():
            continue
        markers = load_markers(arguments.markers, stem)
        used = sum(len(v) for k, v in markers.items() if k != "CAN")

        row, transform = [], None
        for data in datasets:
            case_index = list(data["cases"]).index(index + 1)
            rms, _, rotation, translation, _ = compare(
                data["positions"][case_index], grid, markers, radii)
            row.append(rms)
            if transform is None:                # first dataset sets the frame
                transform = (rotation, translation)
        built, *_ = compare(reference.nodes, grid, markers, radii)

        # Markers carried into the MODEL frame so they can be plotted on top of
        # every curve at once. The transform is the first dataset's alignment,
        # chosen so the overlay does not flatter whichever model came second.
        # For plotting, register on the leading edge alone (see docstring).
        curves = model_curves(datasets[0]["positions"][
            list(datasets[0]["cases"]).index(index + 1)], grid)
        rotation, translation = transform
        if "LE" in markers and len(markers["LE"]) >= 3:
            rotation, translation = refine_on_leading_edge(
                markers, curves, rotation, translation)
        for group, points in markers.items():
            aligned.setdefault(group, {})[index + 1] = (
                (points - translation) @ rotation)
        table.append((index + 1, stem, used, row, built))
        print(f"{index + 1:>5} {stem:>8} {used:>8} " +
              " ".join(f"{value * 1000:>9.0f} mm" for value in row) +
              f" {built * 1000:>7.0f} mm")

    means = np.array([r[3] for r in table])
    print("\n  mean point-to-curve RMS over the ten cases:")
    for column, label in enumerate(arguments.labels):
        print(f"     {label:>10}: {means[:, column].mean() * 1000:6.0f} mm")
    print(f"     {'as built':>10}: {np.mean([r[4] for r in table]) * 1000:6.0f} mm")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        cases=np.array([r[0] for r in table]),
        rms=means,
        as_built=np.array([r[4] for r in table]),
        labels=np.array(arguments.labels),
        **{f"markers_{group}_{case}": points
           for group, cases_ in aligned.items()
           for case, points in cases_.items()},
    )
    print(f"\n  -> {arguments.output}")


if __name__ == "__main__":
    main()

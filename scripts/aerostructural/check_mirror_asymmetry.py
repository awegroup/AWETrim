"""Left-right asymmetry of a converged Billow shape: rigid rotation vs shape.

The unsteered, unactuated kite at the centre of the wind window with gravity off
is mirror-symmetric in every input, so any left-right difference in its solved
shape is model error. Measuring it in the GLOBAL frame, as ``run_ab_canopy_pattern``
does, conflates two things:

``rigid``      the whole kite rotated slightly -- still symmetric, about a tilted
               plane. The kite is pinned at ONE node (the KCU) and the aero load
               is frozen as a dead load during each structural solve, so a small
               yaw/roll about the line through the KCU costs almost no energy.
``intrinsic``  a genuinely asymmetric shape.

The mirrored-and-relabelled shape ``Y_i = M x_p(i)`` of a rigidly rotated
symmetric shape ``X`` is exactly ``R X + t`` for a proper rotation ``R``, so a
best-fit rigid alignment (Kabsch) leaves only the intrinsic part. Pairs are taken
over ALL nodes (bridle and quad centres included) from the exactly symmetric
built model; pairing only the wing grid silently mis-pairs the rest.

Also printed per section pair: the intrinsic mismatch next to the ELASTIC
deformation there (rigid motion from the built shape removed), which says how
much the asymmetry matters locally.

Usage (from project root):
    python scripts/aerostructural/check_mirror_asymmetry.py cross_traction
    python scripts/aerostructural/check_mirror_asymmetry.py cross cross_traction --pattern cross
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from common import DEFAULT_KITE_NAME
from plot_billow_geometry import rebuild

MIRROR = np.array([1.0, -1.0, 1.0])


def kabsch(source, target):
    """Proper rotation ``R`` and translation ``t`` minimising ``sum |R x + t - y|^2``."""
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, ct - rotation @ cs


def final_positions(folder):
    """Node positions at the last recorded coupled iteration."""
    with h5py.File(folder / "sim_output.h5", "r") as handle:
        residual = np.asarray(handle["tracking/residual_norm"])
        positions = np.asarray(handle["tracking/positions"])
    last = int(np.flatnonzero(residual)[-1]) if np.any(residual) else len(residual) - 1
    return positions[last]


def mirror_partners(nodes):
    """Index of each node's mirror partner in an exactly symmetric node set."""
    distances = np.linalg.norm(nodes[None] - (nodes * MIRROR)[:, None], axis=2)
    partner = distances.argmin(axis=1)
    worst = float(distances.min(axis=1).max())
    if worst > 1e-9:
        raise ValueError(f"the built model is not mirror-symmetric ({worst:.3e} m)")
    return partner


def decompose(positions, built, grid, partner):
    """Global and intrinsic mismatch, the rigid rotation, and per-pair details."""
    mirrored = positions[partner] * MIRROR
    rotation, shift = kabsch(positions, mirrored)
    aligned = positions @ rotation.T + shift
    rows = grid.shape[0]
    wing = np.unique(grid)
    rigid_rotation, rigid_shift = kabsch(positions[wing], built[wing])
    elastic = positions @ rigid_rotation.T + rigid_shift

    per_pair = []
    for i in range(rows // 2):
        left, right = grid[i], grid[rows - 1 - i]
        global_mm = max(np.linalg.norm(positions[a] - positions[b] * MIRROR)
                        for a, b in zip(left, right))
        intrinsic = max(np.linalg.norm(aligned[a] - mirrored[a]) for a in left)
        deformation = max(np.linalg.norm(elastic[a] - built[a]) for a in np.r_[left, right])
        per_pair.append((i, float(built[left[0], 1]), global_mm, intrinsic, deformation))

    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    axis = np.degrees(np.array([rotation[2, 1] - rotation[1, 2],
                                rotation[0, 2] - rotation[2, 0],
                                rotation[1, 0] - rotation[0, 1]]) / 2.0)
    return per_pair, angle, axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", help="result folders under billow_canopy_ab/")
    parser.add_argument("--pattern", default="cross")
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    structure = rebuild(project, args.kite, args.panels_per_section,
                        {"canopy_pattern": args.pattern, "canopy_refinement": args.refine})
    built = structure.model.nodes.copy()
    partner = mirror_partners(built)
    root = project / "results" / args.kite / "aerostructural" / "billow_canopy_ab"

    for name in args.folders:
        per_pair, angle, axis = decompose(final_positions(root / name), built,
                                          structure.grid, partner)
        glob = max(p[2] for p in per_pair)
        intr = max(p[3] for p in per_pair)
        print(f"\n{name}: global {1e3 * glob:.2f} mm, intrinsic {1e3 * intr:.2f} mm, "
              f"rigid rotation {angle:.4f} deg "
              f"(about x {axis[0]:+.4f}, y {axis[1]:+.4f}, z {axis[2]:+.4f} deg)")
        print(f"  {'pair':>6} {'y (m)':>7} | {'global mm':>9} {'intrinsic mm':>12} "
              f"| {'elastic def. mm':>15} {'ratio':>6}")
        for i, y, g, m, d in per_pair:
            print(f"  {i:2d}-{structure.grid.shape[0] - 1 - i:<3d} {y:+7.2f} | "
                  f"{1e3 * g:9.1f} {1e3 * m:12.1f} | {1e3 * d:15.1f} {m / d:6.1%}")


if __name__ == "__main__":
    main()

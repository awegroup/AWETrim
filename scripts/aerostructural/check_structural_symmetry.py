"""Structure alone under an exactly symmetric load: does the solve stay symmetric?

No aerodynamics, no coupling. The first-iteration aerodynamic load of the
unsteered kite is made EXACTLY mirror-symmetric (to ~1e-14 N), put on the
exactly symmetric built model through the traction transfer, scaled, and solved
once from the built state per scale.

How the answer reads:

- A linear solve with a mirror-symmetric stiffness and a mirror-symmetric load
  has a unique, symmetric solution. Asymmetry at SMALL load therefore means the
  model is not mirror-symmetric, or the start is singular enough (slack fabric,
  slack cables) that the solution is not unique.
- Asymmetry that appears only ABOVE some load is a bifurcation: the symmetric
  equilibrium lost stability there.

Measured 2026-09-11 on the coarse ``cross`` canopy: 25 mm at 1% load (22.5 N),
then 25 / 99 / 35 / 235 mm at 3 / 10 / 30 / 100% -- present at every load and
erratic, which is neither clean case.

Usage (from project root):
    python scripts/aerostructural/check_structural_symmetry.py
    python scripts/aerostructural/check_structural_symmetry.py --scales 0.01 0.1
"""

import argparse
import time
from pathlib import Path

import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural.billow import structural_billow as sb
from awetrim.aerostructural.fem import aero2struc
from common import DEFAULT_KITE_NAME

from check_load_transfer import build, transfer
from check_mirror_asymmetry import MIRROR, decompose, mirror_partners
from plot_billow_geometry import rebuild
from run_chain_depower_BILLOW import build_once


def symmetric_load(shared, pattern, refine, built, grid, triangles):
    """The first-iteration aero load, symmetrised, as nodal forces on ``built``."""
    state = build(shared, pattern, refine)
    forces = state["panel_forces"]
    n_panels = len(forces)
    _, _, points = transfer("sections", state)
    points = points.reshape(n_panels, -1, 3)
    # the per-panel chordwise weights, recovered from f_kj = w_kj F_k
    weights = (np.einsum("kjc,kc->kj", points, forces)
               / np.einsum("kc,kc->k", forces, forces)[:, None])
    if not aero2struc._panels_follow_grid(state["panels"], state["struc_nodes"], grid):
        forces, weights = forces[::-1], weights[::-1]
    forces = 0.5 * (forces + (forces * MIRROR)[::-1])
    weights = 0.5 * (weights + weights[::-1])
    return aero2struc.map_aero_traction_to_membrane(
        forces, weights, np.linspace(0.0, 1.0, weights.shape[1]), built, grid, triangles
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--wind", type=float, default=4.2)
    parser.add_argument("--pattern", default="cross")
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    settings = {"canopy_pattern": args.pattern, "canopy_refinement": args.refine}
    shared = build_once(project, args.kite, args.panels_per_section,
                        {"aerodynamic": {"gamma_loop_type": "base"}})
    shared["config"]["wind_speed_wind_ref"] = float(args.wind)

    probe = rebuild(project, args.kite, args.panels_per_section, settings)
    built = probe.model.nodes.copy()
    grid = probe.grid
    partner = mirror_partners(built)
    load = symmetric_load(shared, args.pattern, args.refine, built, grid,
                          probe.model.element_set(sb.CANOPY).connectivity)
    total = float(np.linalg.norm(load.sum(axis=0)))
    print(f"\nload: total {total:.1f} N, mirror mismatch "
          f"{np.linalg.norm(load - load[partner] * MIRROR, axis=1).max():.2e} N")
    print(f"{'scale':>6} {'load N':>8} | {'global mm':>9} {'intrinsic mm':>12} "
          f"{'rigid deg':>9} | {'residual N':>10} {'conv':>5} {'s':>5}")

    config = dict(shared["config"].get("structural_billow") or {})
    config.update(settings)
    for scale in args.scales:
        structure = rebuild(project, args.kite, args.panels_per_section, settings)
        started = time.perf_counter()
        structure, converged, positions, f_int = sb.run_billow(
            structure, (scale * load).ravel(), config
        )
        positions = np.asarray(positions).reshape(-1, 3)
        # run_billow returns f_int in the driver convention f_res = f_int + f_ext
        residual = np.linalg.norm(np.asarray(f_int).reshape(-1, 3) + scale * load, axis=1)
        per_pair, angle, _ = decompose(positions, built, grid, partner)
        print(f"{scale:6.2f} {scale * total:8.1f} | {1e3 * max(p[2] for p in per_pair):9.2f} "
              f"{1e3 * max(p[3] for p in per_pair):12.2f} {angle:9.4f} | "
              f"{residual.max():10.3e} {str(bool(converged)):>5} "
              f"{time.perf_counter() - started:5.1f}", flush=True)


if __name__ == "__main__":
    main()

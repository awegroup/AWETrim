"""Are the converged Billow shapes attached on the TRUE polars, and on both sides?

With artificial viscosity on, a panel pushed past its stall onset can lock onto
a stalled branch -- and at the tips it often does so on ONE side only. That
breaks the symmetry of the aerodynamic load however symmetric the structure is,
and every load transfer then faithfully hands the asymmetry to the canopy.

For each result folder this re-trims the converged shape on the true polars
(artificial viscosity as configured) and reports, per panel, the stall margin
``onset - alpha_eff``: negative means stalled. A result solved on attached
polars (``aerodynamic.attached_polars``) is an exact solution of the real model
exactly when every margin here is positive.

Usage (from project root):
    python scripts/aerostructural/check_tip_stall.py cross_traction cross_traction_attached
"""

import argparse
import copy
from pathlib import Path

import h5py
import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.mapping import LinearStructuralToAeroMapper
from awetrim.aerostructural.utils import calculate_cg
from awetrim.aerodynamics.vsm_quasi_steady import _panel_stall_onsets_rad
from common import DEFAULT_KITE_NAME, build_system_model

from plot_billow_geometry import rebuild
from run_chain_depower_BILLOW import build_once

MIRROR = np.array([1.0, -1.0, 1.0])


def converged_positions(folder):
    with h5py.File(folder / "sim_output.h5", "r") as handle:
        residual = np.asarray(handle["tracking/residual_norm"])
        positions = np.asarray(handle["tracking/positions"])
    last = int(np.flatnonzero(residual)[-1]) if np.any(residual) else len(residual) - 1
    return positions[last]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+",
                        help="result folders under billow_canopy_ab/")
    parser.add_argument("--pattern", default="cross")
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--wind", type=float, default=4.2)
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    # TRUE polars: no attached_polars override, whatever the run used.
    shared = build_once(project, args.kite, args.panels_per_section,
                        {"aerodynamic": {"gamma_loop_type": "base"}})
    config = shared["config"]
    config["wind_speed_wind_ref"] = float(args.wind)
    structure = rebuild(project, args.kite, args.panels_per_section,
                        {"canopy_pattern": args.pattern,
                         "canopy_refinement": args.refine})
    le_indices, te_indices = structure.grid[:, 0], structure.grid[:, -1]
    onsets = np.rad2deg(_panel_stall_onsets_rad(shared["body_aero"]))
    root = project / "results" / args.kite / "aerostructural" / "billow_canopy_ab"

    for name in args.folders:
        nodes = converged_positions(root / name)
        update = LinearStructuralToAeroMapper().map(
            nodes, le_indices, te_indices, args.panels_per_section
        )
        forces, _, results = aerodynamic_vsm.run_vsm_package(
            body_aero=copy.deepcopy(shared["body_aero"]),
            solver=copy.deepcopy(shared["vsm_solver"]),
            system_model=build_system_model(
                shared["system_config_path"], shared["tether"], structure.masses, config
            ),
            center_of_gravity=calculate_cg(struc_nodes=nodes, m_arr=structure.masses),
            le_arr=update.leading_edge_points,
            te_arr=update.trailing_edge_points,
            aero_input_type="reuse_initial_polar_data",
            initial_polar_data=copy.deepcopy(shared["polars"]),
            include_gravity=config["is_with_gravity"],
            is_with_plot=False,
        )
        alpha = np.rad2deg(np.ravel(np.asarray(results["alpha_at_ac"], dtype=float)))
        margin = onsets - alpha
        forces = np.asarray(forces, dtype=float)
        n = len(margin)
        print(f"\n{name}: true-polar re-trim of the converged shape")
        print(f"  min stall margin {margin.min():+6.2f} deg, stalled panels "
              f"{int((margin <= 0).sum())}/{n}, sum F_y {forces[:, 1].sum():+8.3f} N")
        print("  outermost panels, one tip vs the other (margin deg | alpha deg):")
        for k in range(3):
            print(f"    panel {k:2d} {margin[k]:+6.2f} | {alpha[k]:5.2f}      "
                  f"panel {n - 1 - k:2d} {margin[n - 1 - k]:+6.2f} | {alpha[n - 1 - k]:5.2f}")
        mirror = np.linalg.norm(forces - (forces * MIRROR)[::-1], axis=1)
        print(f"  panel-force mirror mismatch max {mirror.max():.3e} N "
              f"(panel {int(mirror.argmax())})")


if __name__ == "__main__":
    main()

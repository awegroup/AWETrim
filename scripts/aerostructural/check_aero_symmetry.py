"""Is the aerodynamic solve mirror-symmetric on an exactly symmetric wing?

The structural side is now provably symmetric: the built geometry mirrors to
0.000000 mm and the canopy triangulation is mirror-symmetric by construction.
Yet the coupled solve still develops ~56 mm of left-right mismatch, and it grows
rather than shrinks as the residual is tightened, so it is a converged property
of the model rather than iteration error. That leaves the aerodynamics.

This isolates it. No structure, no coupling, no deformation: take the undeformed
symmetric wing, run the aerodynamic solve once, and compare each panel with its
mirror partner. On a symmetric wing at zero azimuth and zero elevation with
gravity off, panel k and panel n-1-k must carry mirrored forces.

Two levels are measured, because they fail for different reasons:

``vsm``    the circulation solve alone, at a fixed apparent wind. Any mismatch
           here is the aerodynamic solver: panel geometry, the gamma loop's
           termination, or the artificial-viscosity/stall treatment.
``trim``   the full quasi-steady trim the coupled loop actually calls, which
           wraps the above in an optimiser. Mismatch that appears only here is
           the trim, not the aerodynamics.

Usage (from project root):
    python scripts/aerostructural/check_aero_symmetry.py
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml as _yaml

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.fem import read_struc_geometry_yaml
from awetrim.aerostructural.mapping import LinearStructuralToAeroMapper
from awetrim.aerostructural.utils import calculate_cg, load_yaml, rotate_geometry
from awetrim.system.tether import RigidLumpedTether
from awetrim.utils.system_config import get_tether
from common import (
    DEFAULT_KITE_NAME,
    build_system_model,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)

STRUC_GEOMETRY_FILENAME = "struc_geometry_FEM_full.yaml"
MIRROR = np.array([1.0, -1.0, 1.0])


def mirror_pairs(points):
    """Index of each row's mirror partner, matched on the reflected position."""
    reflected = points * MIRROR
    partner = np.empty(len(points), dtype=int)
    for index, target in enumerate(reflected):
        partner[index] = int(np.argmin(np.linalg.norm(points - target, axis=1)))
    return partner


def report(name, points, vectors, scale):
    """Mirror mismatch of a per-panel vector field, as a fraction of ``scale``."""
    partner = mirror_pairs(points)
    residual = np.linalg.norm(vectors - vectors[partner] * MIRROR, axis=1)
    pairing = np.linalg.norm(points - points[partner] * MIRROR, axis=1)
    print(f"  {name:22s} max {residual.max():10.4g}  mean {residual.mean():10.4g}"
          f"   ({100 * residual.max() / scale:6.3f}% of {scale:.4g})")
    if pairing.max() > 1e-9:
        print(f"    ! panel pairing itself is off by {pairing.max():.3e} m --"
              f" the geometry is not symmetric, so the above is not conclusive")
    return residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--wind", type=float, default=4.2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    config_path, aero_geometry_path, _ = resolve_kite_paths(project, args.kite)
    struc_path = project / "data" / args.kite / STRUC_GEOMETRY_FILENAME
    system_path = project / "data" / args.kite / "system.yaml"
    with system_path.open(encoding="utf-8") as handle:
        system_config = _yaml.safe_load(handle)

    config = load_yaml(config_path)
    config["structural_solver"] = "billow"
    config["wind_speed_wind_ref"] = float(args.wind)
    config["aerodynamic"]["n_aero_panels_per_struc_section"] = args.panels_per_section
    config.setdefault("aero2struc", {})["chordwise_distribution"] = "moment_matched"

    struc_geometry = load_yaml(struc_path)
    n_ribs = len(struc_geometry["wing_particles"]["data"]) / 2
    n_panels = (n_ribs - 1) * args.panels_per_section
    body_aero, vsm_solver, vel_app, polars = aerodynamic_vsm.initialize(
        aero_geometry_path, config, n_panels, bridle_path=None
    )

    reader = read_struc_geometry_yaml.main(
        struc_geometry, config=config, system_config=system_config
    )
    nodes, masses, le_indices, te_indices = reader[0], reader[1], reader[2], reader[3]
    nodes = rotate_geometry(nodes, **resolve_initial_geometry_rotation_kwargs(config))

    # The wing the aerodynamics actually sees.
    update = LinearStructuralToAeroMapper().map(
        nodes, le_indices, te_indices, args.panels_per_section
    )
    le, te = update.leading_edge_points, update.trailing_edge_points
    print(f"wing: {len(le)} sections -> {len(le) - 1} panels, "
          f"v_w = {args.wind} m/s\n")

    print("input geometry (must be exact before anything below means anything):")
    for label, pts in (("leading edge", le), ("trailing edge", te)):
        partner = mirror_pairs(pts)
        worst = np.linalg.norm(pts - pts[partner] * MIRROR, axis=1).max()
        print(f"  {label:22s} mirror mismatch {worst:.3e} m")
    print()

    tether_struct = get_tether(system_config)["structure"]
    system_model = build_system_model(
        system_path,
        RigidLumpedTether(diameter=tether_struct["diameter"],
                          density=tether_struct.get("density", 970.0)),
        masses, config,
    )

    forces, solved_body, results = aerodynamic_vsm.run_vsm_package(
        body_aero=copy.deepcopy(body_aero),
        solver=copy.deepcopy(vsm_solver),
        system_model=system_model,
        center_of_gravity=calculate_cg(struc_nodes=nodes, m_arr=masses),
        le_arr=le, te_arr=te,
        aero_input_type="reuse_initial_polar_data",
        initial_polar_data=copy.deepcopy(polars),
        include_gravity=config["is_with_gravity"],
        is_with_plot=False,
    )
    forces = np.asarray(forces, dtype=float)
    centres = np.array([
        0.25 * (p.LE_point_1 + p.LE_point_2 + p.TE_point_1 + p.TE_point_2)
        for p in solved_body.panels
    ])

    print("trimmed aerodynamic solve (what the coupled loop calls):")
    scale = float(np.linalg.norm(forces, axis=1).max())
    report("panel force [N]", centres, forces, scale)

    cp = np.asarray(results.get("panel_cp_locations"), dtype=float)
    if cp.ndim == 2 and len(cp) == len(centres):
        report("centre of pressure [m]", centres, cp - centres,
               float(np.linalg.norm(cp - centres, axis=1).max()))

    alpha = np.asarray(results.get("alpha_at_ac"), dtype=float).ravel()
    if alpha.size == len(centres):
        partner = mirror_pairs(centres)
        spread = np.abs(alpha - alpha[partner])
        print(f"  {'alpha at AC [deg]':22s} max {np.degrees(spread.max()):10.4g}"
              f"  mean {np.degrees(spread.mean()):10.4g}")

    total = forces.sum(axis=0)
    print(f"\n  resultant force        [{total[0]:9.3f} {total[1]:9.3f} {total[2]:9.3f}] N")
    print(f"  spanwise component     {total[1]:.4g} N  "
          f"({100 * abs(total[1]) / np.linalg.norm(total):.3f}% of the resultant)")
    print("    on a mirror-symmetric wing this must be zero")


if __name__ == "__main__":
    main()

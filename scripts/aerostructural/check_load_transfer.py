"""Is the aero-to-structure load transfer conservative?

The aerodynamic solver produces point loads on the canopy. Whatever maps them
onto the structural mesh must not invent or destroy load: the resultant force
and the resultant moment integrated over the FE mesh have to equal the ones
integrated over the aero mesh. Force is the easy half -- any partition of unity
gets it exactly. Moment is the real test, because it fails as soon as a load is
moved off its line of action.

Two routes are compared on the same aerodynamic state:

``sections``   the lattice mapping: each load is spread over the two bracketing
               chordwise node chains. It reaches only nodes lying on a chain, so
               the quad-centre nodes of the ``cross`` pattern get nothing however
               much canopy surrounds them.
``nearest_element``
               each point load through its nearest triangle, split over the
               three vertices by barycentric weights. Exact force and in-plane
               moment, but a point reaches only three nodes.
``traction``   the load as a FIELD over the canopy, integrated over every
               element with its shape functions. Every canopy node is loaded by
               the elements around it, at any refinement.

Also counted: how many structural nodes actually receive load. That number is
why conservation alone is not enough -- the totals can balance perfectly while
part of the mesh is held by the membrane alone.

Usage (from project root):
    python scripts/aerostructural/check_load_transfer.py
    python scripts/aerostructural/check_load_transfer.py --patterns cross --refine 3
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.billow import structural_billow
from awetrim.aerostructural.fem import aero2struc, read_struc_geometry_yaml
from awetrim.aerostructural.mapping import (
    BilinearAeroToStructuralLoadMapper,
    LinearStructuralToAeroMapper,
)
from awetrim.aerostructural.utils import calculate_cg, rotate_geometry
from common import (
    DEFAULT_KITE_NAME,
    build_system_model,
    resolve_initial_geometry_rotation_kwargs,
)

from run_chain_depower_BILLOW import build_once


def resultants(points, forces, reference):
    """Resultant force [N] and moment [N m] about ``reference`` of point loads."""
    points, forces = np.asarray(points, float), np.asarray(forces, float)
    return forces.sum(axis=0), np.cross(points - reference, forces).sum(axis=0)


def transfer(route, state):
    """One mapping route, as ``(nodal forces, aero points, aero forces)``."""
    config = copy.deepcopy(state["config"])
    config["aero2struc"]["load_transfer"] = route
    nodal, debug = aero2struc.main(
        config["aero2struc"]["coupling_method"],
        state["panel_forces"],
        state["struc_nodes"],
        state["panel_cp_locations"],
        state["mapping"],
        False,
        config["aero2struc"],
        state["canopy_sections"],
        [],
        state["panels"],
        canopy_triangles=None if route == "sections" else state["triangles"],
        canopy_grid=state["grid"],
        return_distributed_aero=True,
    )
    return nodal, debug["points"], debug["forces"]


def build(shared, pattern, refine):
    """The Billow model and the aerodynamic state to transfer onto it."""
    config = copy.deepcopy(shared["config"])
    config["structural_billow"] = {
        **(config.get("structural_billow") or {}),
        "canopy_pattern": pattern,
        "canopy_refinement": refine,
    }
    # The reader mutates the geometry it is handed, so each pattern gets a copy.
    geometry = copy.deepcopy(shared["struc_geometry"])
    reader = read_struc_geometry_yaml.main(
        geometry, config=config, system_config=shared["system_config"]
    )
    (struc_nodes, m_arr, le_indices, te_indices, _pt, _st, _pn,
     canopy_sections, strut_sections, _sbp, connectivity, _bc, _bd,
     l0_arr, k_arr, c_arr, link_types, pulley_line_indices, _pd) = reader
    struc_nodes = rotate_geometry(
        struc_nodes, **resolve_initial_geometry_rotation_kwargs(config)
    )
    structure = structural_billow.instantiate(
        config=config, struc_geometry=geometry, struc_nodes=struc_nodes,
        kite_connectivity_arr=connectivity, l0_arr=l0_arr, k_arr=k_arr,
        c_arr=c_arr, m_arr=m_arr, linktype_arr=link_types,
        pulley_line_indices=pulley_line_indices,
        canopy_sections=canopy_sections, strut_sections=strut_sections,
    )
    nodes = structure.model.nodes.copy()

    update = LinearStructuralToAeroMapper().map(
        nodes, le_indices, te_indices,
        config["aerodynamic"]["n_aero_panels_per_struc_section"],
    )
    panel_forces, solved_body, results = aerodynamic_vsm.run_vsm_package(
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
    # The trim returns loads in ITS frame; the driver rotates the structure
    # into that frame before mapping (aerostructural_coupled_solver, right after
    # the aero solve). Skip this and every load is mapped onto a mesh rotated
    # by the trim attitude -- which shows up as a large moment error shared
    # identically by every route, because it is not the mapping's.
    roll, pitch, yaw = results["opt_x"][1:4]
    nodes = rotate_geometry(nodes, angle_deg=[roll, pitch, yaw])
    return dict(
        config=config,
        struc_nodes=nodes,
        panel_forces=np.asarray(panel_forces, dtype=float),
        panel_cp_locations=np.array(results["panel_cp_locations"]),
        panels=solved_body.panels,
        # Exactly what run_chain hands the mapper: the membrane's own grid.
        canopy_sections=[list(map(int, row)) for row in structure.fine_grid],
        grid=structure.fine_grid,
        triangles=structure.model.element_set(structural_billow.CANOPY).connectivity,
        mapping=(BilinearAeroToStructuralLoadMapper()
                 .initialize(shared["body_aero"].panels, nodes,
                             le_indices, te_indices).panel_corner_map),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind", type=float, default=4.2)
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--patterns", nargs="+", default=["cross"])
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    shared = build_once(project, args.kite, args.panels_per_section,
                        {"aerodynamic": {"gamma_loop_type": "base"}})
    shared["config"]["wind_speed_wind_ref"] = float(args.wind)

    print(f"\nunsteered, v_w = {args.wind} m/s, canopy refinement x{args.refine}")
    print("resultants about the origin; the aero side is the reference.\n")
    print(f"{'pattern':9s} {'route':15s} {'canopy':>6} {'loaded':>7} | "
          f"{'|dF| (N)':>10} {'rel':>9} | {'|dM| (N m)':>11} {'rel':>9}")
    print("-" * 90)

    for pattern in args.patterns:
        state = build(shared, pattern, args.refine)
        for route in ("sections", "nearest_element", "traction"):
            nodal, points, forces = transfer(route, state)
            f_ref, m_ref = resultants(points, forces, np.zeros(3))
            f_got, m_got = resultants(state["struc_nodes"], nodal, np.zeros(3))
            canopy = np.unique(state["triangles"])
            loaded = int((np.linalg.norm(nodal[canopy], axis=1) > 0).sum())
            df = float(np.linalg.norm(f_got - f_ref))
            dm = float(np.linalg.norm(m_got - m_ref))
            print(f"{pattern:9s} {route:15s} {len(canopy):6d} "
                  f"{loaded:7d} | {df:10.3e} {df / np.linalg.norm(f_ref):9.2e} | "
                  f"{dm:11.3e} {dm / np.linalg.norm(m_ref):9.2e}", flush=True)
        print()

    print("force must be conserved exactly by either route (the weights sum to")
    print("one). moment is the discriminating check: it is lost as soon as a")
    print("load is moved off its line of action.")


if __name__ == "__main__":
    main()

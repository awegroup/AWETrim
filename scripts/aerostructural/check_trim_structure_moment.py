"""Does the structure receive the moment the trim balanced?

The trim solves a rigid-body balance about the reference point, and the
structure is PINNED at that same point. A pin carries force, not moment, so
the loads the driver hands the structure have to carry zero moment about it.
Whatever they do carry, the structure can only shed by swinging rigidly --
which the trim then rotates back, every iteration, for as long as the
disagreement stands.

Neither existing check sees this. ``check_moment_preservation`` and
``check_load_transfer.py`` both compare the mapped nodal loads against the
already-distributed aerodynamic loads, so they measure the mapping; a term the
trim balances and the structure never receives (or receives at a different wind
speed, or at a different CG) is invisible to both. This one closes the loop:
one trim, then every moment on both sides about the bridle point, term by term,
and the rigid rotation that would null what the structure is given.

``--backend`` picks which coupled driver's load assembly to replicate:

``billow``  `fem/aerostructural_coupled_solver.py` on `struc_geometry_FEM_full`
            -- the Billow (and kite_fem) path, whose aero loads are spread over
            10 chordwise nodes before being mapped.
``pss``     `pss/aerostructural_coupled_solver_qsm.py` on the PSM
            photogrammetry geometry -- each panel load applied at its OWN
            centre of pressure through the bilinear corner map, no chordwise
            spread, and the trim carrying the tether when the config says so.

It found the bridle-line drag being taken at the freestream ``vel_app`` on the
structural side while the trim carried it at the apparent wind -- -326 N m of
pitch and a 0.76 deg swing per iteration (fixed 2026-09-12; see
``aerostructural/AGENTS.md``).

Usage (from the project root):
    python scripts/aerostructural/check_trim_structure_moment.py
    python scripts/aerostructural/check_trim_structure_moment.py --backend pss
    python scripts/aerostructural/check_trim_structure_moment.py --no-bridle
    python scripts/aerostructural/check_trim_structure_moment.py \
        --from-result billow_steering/cross_steer_p0050mm_bridlefix
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml as _yaml
from scipy.optimize import least_squares

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural import aerodynamic_vsm, aerodynamic_bridle_line_drag
from awetrim.aerostructural.billow import structural_billow
from awetrim.aerostructural.fem import read_struc_geometry_yaml
from awetrim.aerostructural.fem.aerostructural_coupled_solver import (
    _bridle_line_drag,
    _bridle_line_specs_for_vsm,
    _map_aero_to_structure,
)
from awetrim.aerostructural.forces import distribute_total_force_by_particle_mass
from awetrim.aerostructural.mapping import (
    BilinearAeroToStructuralLoadMapper,
    LinearStructuralToAeroMapper,
)
from awetrim.aerostructural.pss import structural_geometry_io
from awetrim.aerostructural.pss.aerostructural_coupled_solver_qsm import (
    _bridle_node_pairs,
    _map_aero_loads_to_structure,
)
from awetrim.aerostructural.utils import (
    calculate_cg,
    load_yaml,
    rotate_geometry,
    rotation_matrix_from_angles,
)
from awetrim.system.tether import RigidLumpedTether
from awetrim.utils.system_config import get_tether
from common import (
    DEFAULT_KITE_NAME,
    build_system_model,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)

from run_chain_depower_BILLOW import build_once


def project_root():
    return Path(__file__).resolve().parents[2]


def moment(points, forces, reference=np.zeros(3)):
    """Resultant moment [N m] of point loads about ``reference``."""
    points, forces = np.asarray(points, float), np.asarray(forces, float)
    return np.cross(points - reference, forces).sum(axis=0)


def fmt(vector, unit="N m"):
    vector = np.asarray(vector, float)
    return (f"[{vector[0]:9.1f} {vector[1]:9.1f} {vector[2]:9.1f}] "
            f"|.| = {np.linalg.norm(vector):8.1f} {unit}")


def _starting_positions(args, struc_nodes):
    """A stored converged (and for a steered run asymmetric) shape, or None."""
    if not args.from_result:
        return None
    from check_mirror_asymmetry import final_positions

    positions = final_positions(
        project_root() / "results" / args.kite / "aerostructural" / args.from_result
    )
    if positions.shape != struc_nodes.shape:
        raise ValueError(
            f"the result has {len(positions)} nodes, the model {len(struc_nodes)}: "
            "pass the --pattern/--refine (or --backend) the run used"
        )
    return positions.copy()


def build_billow(args):
    """The Billow model, one trim on it, and the loads the driver would build."""
    shared = build_once(project_root(), args.kite, args.panels_per_section,
                        {"aerodynamic": {"gamma_loop_type": args.gamma_loop}})
    config = copy.deepcopy(shared["config"])
    config["wind_speed_wind_ref"] = float(args.wind)
    config["structural_billow"] = {
        **(config.get("structural_billow") or {}),
        "canopy_pattern": args.pattern,
        "canopy_refinement": args.refine,
    }
    if args.no_bridle:
        config["is_with_aero_bridle"] = False

    # The reader mutates the geometry it is handed (strut padding).
    geometry = copy.deepcopy(shared["struc_geometry"])
    (struc_nodes, m_arr, le_indices, te_indices, _pt, _sti, _pn,
     canopy_sections, strut_sections, _sbp, connectivity, bridle_connectivity,
     bridle_diameter, l0_arr, k_arr, c_arr, link_types, pulley_line_indices,
     _pd) = read_struc_geometry_yaml.main(
        geometry, config=config, system_config=shared["system_config"]
    )
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
    # instantiate() may ADD nodes, and relax_bridles moves them: the coupled
    # loop starts from the model's own state, so this has to as well.
    struc_nodes = structure.model.nodes.copy()
    m_arr = structure.masses
    _from_result = _starting_positions(args, struc_nodes)
    if _from_result is not None:
        struc_nodes = _from_result

    body_aero = copy.deepcopy(shared["body_aero"])
    if args.no_bridle:
        body_aero._bridle_line_system = None
    mapping = (BilinearAeroToStructuralLoadMapper()
               .initialize(body_aero.panels, struc_nodes, le_indices, te_indices)
               .panel_corner_map)
    update = LinearStructuralToAeroMapper().map(
        struc_nodes, le_indices, te_indices,
        config["aerodynamic"]["n_aero_panels_per_struc_section"],
    )
    cg = calculate_cg(struc_nodes=struc_nodes, m_arr=m_arr)
    # Same call the driver makes -- including handing it no `config`.
    f_panels, solved_body, results = aerodynamic_vsm.run_vsm_package(
        body_aero=body_aero,
        solver=copy.deepcopy(shared["vsm_solver"]),
        system_model=build_system_model(
            shared["system_config_path"], shared["tether"], m_arr, config
        ),
        center_of_gravity=cg,
        le_arr=update.leading_edge_points,
        te_arr=update.trailing_edge_points,
        aero_input_type="reuse_initial_polar_data",
        initial_polar_data=copy.deepcopy(shared["polars"]),
        include_gravity=config["is_with_gravity"],
        is_with_plot=False,
        struc_nodes=struc_nodes,
        bridle_line_specs=_bridle_line_specs_for_vsm(body_aero, bridle_connectivity),
    )
    # The trim returns its loads in ITS frame; the driver rotates the structure
    # into that frame before mapping. Skip it and every load lands on a mesh
    # rotated by the trim attitude.
    roll, pitch, yaw = results["opt_x"][1:4]
    nodes = rotate_geometry(struc_nodes, angle_deg=[roll, pitch, yaw])

    f_wing, debug = _map_aero_to_structure(
        config, np.asarray(f_panels, float), nodes, results, mapping,
        [list(map(int, row)) for row in structure.fine_grid], [],
        solved_body.panels, structure, is_with_conservation_check=False,
    )
    f_bridle = _bridle_line_drag(
        config, nodes, bridle_connectivity, bridle_diameter, results,
        shared["vel_app"],
    )
    return _assemble(config, results, solved_body, nodes, cg, (roll, pitch, yaw),
                     debug, f_wing, f_bridle, m_arr, "chordwise placement")


def build_pss(args):
    """The same, for the PSS/QSM driver on the PSM photogrammetry geometry."""
    config_path, aero_geometry_path, struc_geometry_path = resolve_kite_paths(
        project_root(), args.kite
    )
    system_config_path = project_root() / "data" / args.kite / "system_flown_2019.yaml"
    system_config = _yaml.safe_load(system_config_path.read_text(encoding="utf-8"))

    config = load_yaml(config_path)
    config["structural_solver"] = "pss"
    config["wind_speed_wind_ref"] = float(args.wind)
    config["aerodynamic"]["gamma_loop_type"] = args.gamma_loop
    if args.no_bridle:
        config["is_with_aero_bridle"] = False
    cp_rel = config.get("aero2struc", {}).get("cp_distribution_path")
    if cp_rel:
        config["aero2struc"]["cp_distribution_path"] = str(project_root() / cp_rel)

    struc_geometry = load_yaml(struc_geometry_path)
    n_ribs = len(struc_geometry["wing_particles"]["data"]) / 2
    n_panels = (n_ribs - 1) * config["aerodynamic"]["n_aero_panels_per_struc_section"]
    bridle_path = (
        struc_geometry_path if config.get("is_with_aero_bridle", False) else None
    )
    body_aero, vsm_solver, vel_app, polars = aerodynamic_vsm.initialize(
        aero_geometry_path, config, n_panels, bridle_path=bridle_path
    )
    bridle_line_specs = aerodynamic_vsm.parse_bridle_line_specs(struc_geometry, config)

    (struc_nodes, m_arr, le_indices, te_indices, _pt, _sti, _pn, _cs, _ss, _sbp,
     _conn, bridle_connectivity, bridle_diameter, *_rest) = structural_geometry_io.main(
        struc_geometry, config=config, system_config=system_config
    )
    struc_nodes = rotate_geometry(
        struc_nodes, **resolve_initial_geometry_rotation_kwargs(config)
    )
    _from_result = _starting_positions(args, struc_nodes)
    if _from_result is not None:
        struc_nodes = _from_result

    mapping = (BilinearAeroToStructuralLoadMapper()
               .initialize(body_aero.panels, struc_nodes, le_indices, te_indices)
               .panel_corner_map)
    pairs = (_bridle_node_pairs(bridle_line_specs, struc_nodes, body_aero)
             if config["is_with_aero_bridle"] else None)
    update = LinearStructuralToAeroMapper().map(
        struc_nodes, le_indices, te_indices,
        config["aerodynamic"]["n_aero_panels_per_struc_section"],
    )
    cg = calculate_cg(struc_nodes=struc_nodes, m_arr=m_arr)

    # Tether CLASS from the config, as run_simulation_PSM does: this driver
    # passes `config` to the trim, so `tether.include_in_trim` is live here and
    # the tether-aware trim type-checks the model it is given.
    tether_struct = get_tether(system_config)["structure"]
    tether_cfg = config.get("tether", {}) or {}
    if str(tether_cfg.get("model", "rigid_lumped")).lower() == "williams":
        from awetrim.system.williams_tether import WilliamsTether

        tether = WilliamsTether(
            diameter=tether_struct["diameter"],
            density=tether_struct.get("density", 970.0),
            n_elements=int(tether_cfg.get("n_elements", 10)),
            elastic=bool(tether_cfg.get("is_elastic", False)),
            cf=float(tether_cfg.get("cf", 0.01)),
        )
    else:
        tether = RigidLumpedTether(
            diameter=tether_struct["diameter"],
            density=tether_struct.get("density", 970.0),
        )

    f_panels, solved_body, results = aerodynamic_vsm.run_vsm_package(
        body_aero=body_aero,
        solver=vsm_solver,
        system_model=build_system_model(system_config_path, tether, m_arr, config),
        center_of_gravity=cg,
        le_arr=update.leading_edge_points,
        te_arr=update.trailing_edge_points,
        aero_input_type="reuse_initial_polar_data",
        initial_polar_data=polars,
        include_gravity=config["is_with_gravity"],
        is_with_plot=False,
        struc_nodes=struc_nodes,
        bridle_line_specs=bridle_line_specs,
        config=config,
    )
    roll, pitch, yaw = results["opt_x"][1:4]
    nodes = rotate_geometry(struc_nodes, angle_deg=[roll, pitch, yaw])

    panel_cps = np.array(results["panel_cp_locations"])
    f_wing = _map_aero_loads_to_structure(
        np.asarray(f_panels, float), nodes, panel_cps, mapping
    )
    if config["is_with_aero_bridle"]:
        f_bridle = aerodynamic_bridle_line_drag.main(
            nodes, bridle_connectivity, bridle_diameter, vel_app,
            config["rho"], config["aerodynamic_bridle"]["cd_cable"],
            config["aerodynamic_bridle"]["cf_cable"],
            body_aero=solved_body, bridle_node_pairs=pairs,
        )
    else:
        f_bridle = np.zeros_like(nodes)
    # No chordwise spread on this path: the panel load IS applied at its own
    # centre of pressure, so the panel loads are the "distributed" ones.
    debug = {"points": panel_cps, "forces": np.asarray(f_panels, float)}
    return _assemble(config, results, solved_body, nodes, cg, (roll, pitch, yaw),
                     debug, f_wing, f_bridle, m_arr, "cp placement")


def _assemble(config, results, body, nodes, cg, attitude, debug, f_wing, f_bridle,
              m_arr, placement_label):
    """The inertial/gravity split both drivers share, and the state to report."""
    f_inertial = distribute_total_force_by_particle_mass(
        np.asarray(results.get("inertial_force", np.zeros(3)), float), m_arr
    )
    if config["is_with_gravity"] and "gravity_force" in results:
        f_gravity = distribute_total_force_by_particle_mass(
            np.asarray(results["gravity_force"], float), m_arr
        )
    else:
        f_gravity = np.zeros_like(nodes)
    return dict(
        config=config, results=results, body=body, nodes=nodes, cg=cg,
        attitude=attitude, distributed=debug, placement_label=placement_label,
        f_wing=f_wing, f_bridle=f_bridle, f_inertial=f_inertial,
        f_gravity=f_gravity, f_ext=f_wing + f_bridle + f_inertial + f_gravity,
    )


def budget(state):
    """Every moment both sides carry about the bridle point."""
    config, results, body = state["config"], state["results"], state["body"]
    nodes = state["nodes"]

    # The trim's residual, back in N m. Its balance is
    # aero + cg_arm x (inertial + gravity) + arm_kcu x kcu_drag = residual,
    # and arm_kcu is zero whenever the reference point is the bridle point
    # (every shipped configuration), so the aero side follows by difference.
    q_inf = 0.5 * float(config["rho"]) * float(results["Umag"]) ** 2
    denom = (q_inf * float(body.wings[0].compute_projected_area())
             * max(float(panel.chord) for panel in body.panels))
    m_residual = np.asarray(results["cm"], float) * denom

    cg_arm = rotation_matrix_from_angles(angle_deg=state["attitude"]) @ state["cg"]
    f_inertial_total = np.asarray(results.get("inertial_force", np.zeros(3)), float)
    m_inertial_trim = np.cross(cg_arm, f_inertial_total)
    if config["is_with_gravity"]:
        m_inertial_trim += np.cross(
            cg_arm, np.asarray(results["gravity_force"], float)
        )

    # The VSM's own bridle-line drag -- the term the trim balanced -- on the
    # segments and apparent wind it actually used. The 50/50 endpoint lumping
    # on the structural side is moment-identical to this midpoint application,
    # so with the same segments and the same va the two agree exactly.
    va = np.asarray(results["va_vel_world"], float)
    f_bridle_vsm, m_bridle_vsm = np.zeros(3), np.zeros(3)
    for line in getattr(body, "_bridle_line_system", None) or []:
        force = body.compute_line_aerodynamic_force(
            va, line,
            cd_cable=config["aerodynamic_bridle"]["cd_cable"],
            cf_cable=config["aerodynamic_bridle"]["cf_cable"],
            rho=config["rho"],
        )
        f_bridle_vsm += force
        m_bridle_vsm += np.cross(
            0.5 * (np.asarray(line[0], float) + np.asarray(line[1], float)), force
        )

    m_aero_trim = m_residual - m_inertial_trim
    return dict(
        residual=m_residual,
        wing_trim=m_aero_trim - m_bridle_vsm,
        bridle_trim=m_bridle_vsm,
        bridle_force_trim=f_bridle_vsm,
        inertial_trim=m_inertial_trim,
        wing_struct=moment(nodes, state["f_wing"]),
        bridle_struct=moment(nodes, state["f_bridle"]),
        inertial_struct=moment(nodes, state["f_inertial"] + state["f_gravity"]),
        total_struct=moment(nodes, state["f_ext"]),
        distributed=moment(state["distributed"]["points"],
                           state["distributed"]["forces"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="billow", choices=("billow", "pss"),
                        help="which coupled driver's load assembly to replicate")
    parser.add_argument("--wind", type=float, default=4.2)
    parser.add_argument("--panels-per-section", type=int, default=2,
                        help="billow only; the pss path takes the config's value")
    parser.add_argument("--pattern", default="cross", help="billow only")
    parser.add_argument("--refine", type=int, default=1, help="billow only")
    parser.add_argument("--gamma-loop", default="base", choices=("base", "anderson"))
    parser.add_argument("--no-bridle", action="store_true",
                        help="drop the bridle-line drag from BOTH sides")
    parser.add_argument("--from-result", default=None,
                        help="result folder under results/<kite>/aerostructural/: "
                             "run the budget on its converged (deformed, and for "
                             "a steered run asymmetric) shape instead of the built one")
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    state = build_billow(args) if args.backend == "billow" else build_pss(args)
    m = budget(state)
    results, roll, pitch, yaw = state["results"], *state["attitude"]
    mesh = (f"{args.pattern} x{args.refine}" if args.backend == "billow"
            else "PSM photogrammetry")

    print("\n" + "=" * 78)
    print(f"{args.backend}   v_w = {args.wind} m/s   v_a = {results['Umag']:.2f} m/s   "
          f"{len(state['body'].panels)} panels   {len(state['nodes'])} nodes   {mesh}")
    print(f"trim attitude  roll {roll:+.3f}  pitch {pitch:+.3f}  yaw {yaw:+.3f} deg"
          f"   course rate {results['opt_x'][4]:+.4f} rad/s"
          f"   gravity {'on' if state['config']['is_with_gravity'] else 'off'}"
          f"   bridle drag {'off' if args.no_bridle else 'on'}")
    print("=" * 78)

    print("\nwhat the TRIM balanced (about the bridle point)")
    print(f"  aero, wing               {fmt(m['wing_trim'])}")
    print(f"  aero, bridle lines       {fmt(m['bridle_trim'])}   "
          f"|F| = {np.linalg.norm(m['bridle_force_trim']):.1f} N")
    print(f"  inertial (+gravity)      {fmt(m['inertial_trim'])}")
    print(f"  ------------------------ residual {fmt(m['residual'])}")

    print("\nwhat the STRUCTURE receives (same point, same state)")
    print(f"  aero, wing, mapped       {fmt(m['wing_struct'])}")
    print(f"  aero, bridle lines       {fmt(m['bridle_struct'])}   "
          f"|F| = {np.linalg.norm(state['f_bridle'].sum(axis=0)):.1f} N")
    print(f"  inertial (+gravity)      {fmt(m['inertial_struct'])}")
    print(f"  ------------------------ net      {fmt(m['total_struct'])}")

    print("\nthe disagreement, term by term")
    print(f"  wing transfer            {fmt(m['wing_struct'] - m['wing_trim'])}")
    print(f"    {state['placement_label']:22s} {fmt(m['distributed'] - m['wing_trim'])}")
    print(f"    spatial mapping        {fmt(m['wing_struct'] - m['distributed'])}")
    print(f"  bridle drag              {fmt(m['bridle_struct'] - m['bridle_trim'])}")
    print(f"  inertial/gravity         "
          f"{fmt(m['inertial_struct'] - m['inertial_trim'])}")
    print(f"  ------------------------ total    "
          f"{fmt(m['total_struct'] - m['residual'])}")

    print("\nforce check (the pin carries the difference, at zero arm)")
    print(f"  trim aero force          "
          f"{fmt(np.asarray(results['total_aero_force_vec'], float), 'N')}")
    print(f"  f_ext total              {fmt(state['f_ext'].sum(axis=0), 'N')}")
    print(f"  KCU drag (trim only)     "
          f"{fmt(np.asarray(results.get('kcu_drag_force_vsm', np.zeros(3)), float), 'N')}")

    # A rigid rotation about the pin strains nothing, so the structure's only
    # equilibrium condition on it is that the applied moment vanish. With the
    # loads frozen (they are, for one structural solve) this IS the swing.
    nodes, f_ext = state["nodes"], state["f_ext"]
    swing = least_squares(
        lambda angles: moment(rotate_geometry(nodes, angle_deg=angles), f_ext),
        np.zeros(3), xtol=1e-14, ftol=1e-14,
    ).x
    print("\nrigid swing that nulls what the structure receives (frozen loads)")
    print(f"  roll {swing[0]:+.3f}  pitch {swing[1]:+.3f}  yaw {swing[2]:+.3f} deg"
          f"   |angle| = {np.linalg.norm(swing):.3f} deg")
    print()


if __name__ == "__main__":
    main()

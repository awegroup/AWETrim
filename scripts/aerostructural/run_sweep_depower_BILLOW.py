"""Depower sweep at the centre of the wind window, over several apparent speeds.

A small two-dimensional grid: depower tape extension (up and down from the
geometry's own length) crossed with target apparent speed. The window centre
geometry is as_config's -- zero elevation, zero azimuth, 90 degree course --
so the only things varying are the tape and the wind.

Two things this does that a plain loop over ``run_simulation_BILLOW`` would not:

* **VSM is initialised once** and deep-copied per point. Building the body and
  its polars costs about as much as a whole coupled solve, so paying it once
  turns a grid into roughly the cost of its solves.
* **The wind is iterated onto a target apparent speed**, not fixed. The
  apparent-speed-to-wind ratio is the trimmed L/D, which moves substantially
  over a depower sweep -- so a fixed-wind grid slides its v_a with u_dp and the
  rows are not comparable. This is the one-shot version of the ratio table
  ``run_center_sweeps_continuation.sh`` uses.

Why target v_a at all, rather than trusting v-squared self-similarity: the
inflatable tube law saturates at ``moment_max``, so beyond it the tubes hinge
at constant moment and the shape stops scaling with load. Load decides whether
the tubes are inside their calibrated range, so it has to be controlled.

Each point actuates in steps from the built tape length and re-converges at
each step, which is continuation rather than a single jump.

**Prefer ``run_chain_depower_BILLOW.py`` for anything but a handful of points.**
Every point here re-walks the tape continuation from the built length, and does
so again on each wind pass, so the same grid costs about twenty times more:
measured 14 minutes per point against 26 points in 19 minutes for the chains.
What this script still buys is per-point v_a targeting, which the chains give up
(they hold the wind and let v_a slide). Use it when a few points must sit at an
exact apparent speed.

Usage (from project root):
    python scripts/aerostructural/run_sweep_depower_BILLOW.py
    python scripts/aerostructural/run_sweep_depower_BILLOW.py --va 15 20 --dl -0.1 0 0.1
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import yaml as _yaml

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural.mapping import BilinearAeroToStructuralLoadMapper
from awetrim.aerostructural.results import (
    aerostructural_results_root,
    build_deformed_aero_geometry,
    build_deformed_struc_geometry,
    save_geometry_snapshot,
    save_input_snapshot,
    save_sim_output,
)
from awetrim.aerostructural.utils import load_yaml, rotate_geometry
from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.billow import structural_billow
from awetrim.aerostructural.fem import aerostructural_coupled_solver, read_struc_geometry_yaml
from awetrim.system.tether import RigidLumpedTether
from awetrim.utils.system_config import get_tether
from common import (
    DEFAULT_KITE_NAME,
    build_system_model,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)

STRUC_GEOMETRY_FILENAME = "struc_geometry_FEM_full.yaml"

#: Depower map, single-sourced from the kite's own calibration:
#: l_dp = -1.724 u^2 + 6.624 u + 0.192  [m].
DEPOWER_QUADRATIC = (-1.724, 6.624, 0.192)

#: Starting guess for v_a / v_wind -- the trimmed L/D, measured 5.8 at the
#: window centre on the unactuated kite. Each point corrects it from its own
#: first solve, so this only sets how many passes a point needs.
INITIAL_SPEED_RATIO = 5.8

#: Accept a point's wind when the achieved apparent speed is this close.
SPEED_TOLERANCE = 0.04


def depower_input(tape_length: float) -> float:
    """Invert the depower map: tape length [m] -> u_dp.

    The quadratic has two roots; the physical branch is the lower one, which is
    the only one inside the 0-1 actuation range.
    """
    a, b, c = DEPOWER_QUADRATIC
    discriminant = b**2 + 4.0 * a * (tape_length - c)
    if discriminant < 0.0:
        return float("nan")
    return float((b - np.sqrt(discriminant)) / (-2.0 * a))


def build_once(project_dir, kite_name):
    """Everything that does not change across the grid."""
    config_path, aero_geometry_path, _ = resolve_kite_paths(project_dir, kite_name)
    struc_geometry_path = project_dir / "data" / kite_name / STRUC_GEOMETRY_FILENAME
    system_config_path = project_dir / "data" / kite_name / "system.yaml"
    with system_config_path.open("r", encoding="utf-8") as handle:
        system_config = _yaml.safe_load(handle)

    config = load_yaml(config_path)
    config["structural_solver"] = "billow"
    config.setdefault("aero2struc", {})["chordwise_distribution"] = "moment_matched"
    config["aerodynamic"]["n_aero_panels_per_struc_section"] = 2
    config["steering_tape_final_extension"] = 0.0
    config["steering_tape_extension_step"] = 0.0
    cp_rel = config.get("aero2struc", {}).get("cp_distribution_path")
    if cp_rel:
        config["aero2struc"]["cp_distribution_path"] = str(project_dir / cp_rel)

    struc_geometry = load_yaml(struc_geometry_path)
    aero_geometry = load_yaml(aero_geometry_path)
    n_struc_ribs = len(struc_geometry["wing_particles"]["data"]) / 2
    n_panels_aero = (n_struc_ribs - 1) * config["aerodynamic"][
        "n_aero_panels_per_struc_section"
    ]
    bridle_path = (
        struc_geometry_path if config.get("is_with_aero_bridle", False) else None
    )
    body_aero, vsm_solver, vel_app, polars = aerodynamic_vsm.initialize(
        aero_geometry_path, config, n_panels_aero, bridle_path=bridle_path
    )

    tether_struct = get_tether(system_config)["structure"]
    tether = RigidLumpedTether(
        diameter=tether_struct["diameter"],
        density=tether_struct.get("density", 970.0),
    )
    return dict(
        config=config,
        struc_geometry=struc_geometry,
        struc_geometry_path=struc_geometry_path,
        aero_geometry=aero_geometry,
        system_config=system_config,
        system_config_path=system_config_path,
        body_aero=body_aero,
        vsm_solver=vsm_solver,
        vel_app=vel_app,
        polars=polars,
        tether=tether,
    )


def solve_point(shared, wind_speed, tape_extension, tape_step, results_dir):
    """One coupled solve at a given wind and depower tape extension."""
    config = copy.deepcopy(shared["config"])
    config["wind_speed_wind_ref"] = float(wind_speed)
    config["power_tape_final_extension"] = float(tape_extension)
    config["power_tape_extension_step"] = float(tape_step) if tape_extension else 0.0

    # read_struc_geometry_yaml.main MUTATES the geometry it is given:
    # initialize_particles inserts the strut padding nodes straight into
    # strut_tubes' node_indices lists. Calling it twice on one dict therefore
    # pads an already-padded strut and walks off the end. A single-run script
    # never sees this; a sweep does on its second point.
    geometry = copy.deepcopy(shared["struc_geometry"])
    reader = read_struc_geometry_yaml.main(
        geometry,
        config=config,
        system_config=shared["system_config"],
    )
    (struc_nodes, m_arr, le_indices, te_indices, power_tape_index, _steer, _pn,
     canopy_sections, strut_sections, _sbp, connectivity, bridle_connectivity,
     bridle_diameter, l0_arr, k_arr, c_arr, link_types, pulley_line_indices,
     pulley_dict) = reader

    struc_nodes = rotate_geometry(
        struc_nodes, **resolve_initial_geometry_rotation_kwargs(config)
    )
    structure = structural_billow.instantiate(
        config=config,
        struc_geometry=geometry,
        struc_nodes=struc_nodes,
        kite_connectivity_arr=connectivity,
        l0_arr=l0_arr,
        k_arr=k_arr,
        c_arr=c_arr,
        m_arr=m_arr,
        linktype_arr=link_types,
        pulley_line_indices=pulley_line_indices,
        canopy_sections=canopy_sections,
        strut_sections=strut_sections,
    )
    struc_nodes = structure.model.nodes.copy()

    mapping = (
        BilinearAeroToStructuralLoadMapper()
        .initialize(shared["body_aero"].panels, struc_nodes, le_indices, te_indices)
        .panel_corner_map
    )
    n_steps = (
        int(abs(tape_extension) / abs(tape_step)) if tape_extension and tape_step else 0
    )
    system_model = build_system_model(
        shared["system_config_path"], shared["tether"], m_arr, config
    )

    tracking, meta = aerostructural_coupled_solver.main(
        m_arr=m_arr,
        struc_nodes=struc_nodes,
        struc_nodes_initial=struc_nodes.copy(),
        system_model=system_model,
        config=config,
        initial_length_power_tape=l0_arr[power_tape_index],
        n_power_tape_steps=n_steps,
        power_tape_final_extension=config["power_tape_final_extension"],
        power_tape_extension_step=config["power_tape_extension_step"],
        kite_connectivity_arr=connectivity,
        bridle_connectivity_arr=bridle_connectivity,
        pulley_line_indices=pulley_line_indices,
        pulley_line_to_other_node_pair_dict=pulley_dict,
        struc_node_le_indices=le_indices,
        struc_node_te_indices=te_indices,
        body_aero=copy.deepcopy(shared["body_aero"]),
        vsm_solver=copy.deepcopy(shared["vsm_solver"]),
        vel_app=shared["vel_app"],
        initial_polar_data=copy.deepcopy(shared["polars"]),
        bridle_diameter_arr=bridle_diameter,
        aero2struc_mapping=mapping,
        power_tape_index=power_tape_index,
        billow_structure=structure,
        canopy_sections=canopy_sections,
        strut_sections=strut_sections,
    )

    final = np.asarray(tracking["positions"][meta["n_iter"] - 1])
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        save_input_snapshot(config=config, results_dir=results_dir)
        save_sim_output(tracking, meta, results_dir)
        save_geometry_snapshot(
            config,
            build_deformed_struc_geometry(geometry, final),
            build_deformed_aero_geometry(
                shared["aero_geometry"], final, le_indices, te_indices
            ),
            results_dir,
        )
    return (tracking, meta, final, structure,
            l0_arr[power_tape_index] + tape_extension, geometry)


def span_metrics(structure, positions, struc_geometry):
    """Tip-to-tip span, leading-edge arc length and arch rise [m]."""
    chain = structural_billow.leading_edge_chain(struc_geometry)
    edge = positions[chain]
    start, end = edge[0], edge[-1]
    axis = end - start
    span = float(np.linalg.norm(axis))
    axis = axis / span
    relative = edge - start
    rise = float(np.linalg.norm(relative - np.outer(relative @ axis, axis), axis=1).max())
    arc = float(np.linalg.norm(np.diff(edge, axis=0), axis=1).sum())
    return span, arc, rise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--va", type=float, nargs="+", default=[15.0, 20.0, 25.0],
                        help="target apparent speeds [m/s]")
    parser.add_argument("--dl", type=float, nargs="+",
                        default=[-0.2, -0.1, 0.0, 0.1, 0.2],
                        help="depower tape extensions [m], negative powers up")
    parser.add_argument("--tape-step", type=float, default=0.05,
                        help="actuation step [m]; the tape walks to its target")
    parser.add_argument("--speed-passes", type=int, default=2,
                        help="wind corrections per point before accepting v_a")
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    shared = build_once(project_dir, args.kite)
    root = aerostructural_results_root(project_dir, args.kite) / "billow_depower_sweep"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "sweep.csv"

    fields = ["target_va", "achieved_va", "kite_speed", "wind_speed",
              "tape_extension", "tape_length", "u_dp", "converged", "residual",
              "iterations", "span", "le_arc", "arch_rise", "aero_force",
              "tether_force", "seconds"]
    rows = []
    started = time.perf_counter()

    print(f"grid: {len(args.va)} apparent speeds x {len(args.dl)} tape extensions "
          f"= {len(args.va) * len(args.dl)} points\n")
    print(f"{'target':>7} {'dl':>6} {'u_dp':>6} | {'wind':>6} {'v_a':>6} | "
          f"{'res':>7} {'it':>3} | {'span':>6} {'rise':>6} | {'F':>7} | {'s':>5}")
    print("-" * 88)

    for target in args.va:
        wind = target / INITIAL_SPEED_RATIO
        for extension in args.dl:
            point_start = time.perf_counter()
            folder = root / f"va{target:04.1f}_dl{extension:+05.2f}".replace(".", "p")
            tracking = meta = final = structure = None
            achieved = float("nan")

            # Each attempt is saved, so the accepted one is always on disk and
            # the loop can stop the moment it is on target rather than burning
            # every pass. L/D moves enough over a depower sweep (6.5 -> 4.1 for
            # 0.1 m of tape, measured) that the first guess is rarely close.
            wind_used = wind
            for attempt in range(max(1, args.speed_passes)):
                last = attempt == args.speed_passes - 1
                wind_used = wind
                tracking, meta, final, structure, tape_length, geometry = solve_point(
                    shared, wind_used, extension, args.tape_step, folder,
                )
                # speed_apparent is the real v_a; speed_tangential is the kite
                # speed, which at this window centre is within a few percent of
                # it. Prefer the former and fall back on the latter.
                achieved = float(meta.get("speed_apparent", np.nan))
                if not np.isfinite(achieved):
                    achieved = float(meta.get("speed_tangential", np.nan))
                if not np.isfinite(achieved) or achieved <= 0.0:
                    break
                if abs(achieved - target) / target <= SPEED_TOLERANCE or last:
                    break
                wind = wind_used * target / achieved

            span, arc, rise = span_metrics(structure, final, geometry)
            force = float(np.linalg.norm(
                np.asarray(tracking["f_ext"][meta["n_iter"] - 1]).sum(axis=0)
            ))
            elapsed = time.perf_counter() - point_start
            row = dict(
                target_va=target, achieved_va=achieved,
                kite_speed=float(meta.get("speed_tangential", np.nan)),
                tether_force=float(meta.get("tether_force", np.nan)),
                wind_speed=wind_used,
                tape_extension=extension, tape_length=tape_length,
                u_dp=depower_input(tape_length), converged=bool(meta["converged"]),
                residual=float(np.asarray(tracking["residual_norm"])[meta["n_iter"] - 1]),
                iterations=int(meta["n_iter"] - 1), span=span, le_arc=arc,
                arch_rise=rise, aero_force=force, seconds=elapsed,
            )
            rows.append(row)
            print(f"{target:7.1f} {extension:+6.2f} {row['u_dp']:6.3f} | "
                  f"{wind_used:6.2f} {achieved:6.2f} | {row['residual']:7.2f} "
                  f"{row['iterations']:3d} | {span:6.3f} {rise:6.3f} | "
                  f"{force:7.0f} | {elapsed:5.0f}", flush=True)

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    print("-" * 88)
    print(f"{len(rows)} points in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"written to {csv_path}")


if __name__ == "__main__":
    main()

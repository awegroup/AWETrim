"""Depower continuation chains at the centre of the wind window.

One chain per wind speed, walking the depower tape away from the length the
geometry stores and converging at every step. Each step warm-starts the next, so
a chain costs about one tape walk rather than one walk per point --- the
difference between roughly forty minutes and three and a half hours for the same
grid. This is how ``run_center_sweeps_continuation.sh`` does it.

Two chains are run per wind: one shortening the tape (powering the kite up) and
one lengthening it (depowering), both starting from the built length. The
coupled driver already walks the tape and re-converges at each step inside a
single call, so a chain is one call; the converged states are then sliced out of
the tracking record by tape length.

Within a chain the wind is held, so the apparent speed slides as the depower
changes --- measured L/D runs about 7.9 powered to 5.2 depowered, which is
exactly why the campaigns keep a ratio table. Here v_a is recorded per point
rather than targeted, so a chain is a curve through the v_a-u_dp plane rather
than a line of constant v_a. Cross-chain interpolation onto constant v_a is left
to the analysis.

Usage (from project root):
    python scripts/aerostructural/studies/run_chain_depower_BILLOW.py
    python scripts/aerostructural/studies/run_chain_depower_BILLOW.py --wind 3 4 5 --reach 0.2
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import h5py
import numpy as np
import yaml as _yaml

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural.mapping import BilinearAeroToStructuralLoadMapper
from awetrim.aerostructural.results import aerostructural_results_root, save_sim_output
from awetrim.aerostructural.utils import load_yaml, rotate_geometry
from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.billow import structural_billow
from awetrim.aerostructural.coupled import coupled_solver, read_struc_geometry_yaml
from awetrim.system.tether import RigidLumpedTether
from awetrim.utils.system_config import get_tether
from awetrim.aerostructural.case import (
    DEFAULT_KITE_NAME,
    build_system_model,
    build_tether,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)

STRUC_GEOMETRY_FILENAME = "struc_geometry_FEM_full.yaml"

#: l_dp = -1.724 u^2 + 6.624 u + 0.192  [m], the kite's own depower calibration.
DEPOWER_QUADRATIC = (-1.724, 6.624, 0.192)


def depower_input(tape_length):
    """Invert the depower map: tape length [m] -> u_dp. Lower root is physical."""
    a, b, c = DEPOWER_QUADRATIC
    discriminant = b**2 + 4.0 * a * (np.asarray(tape_length, float) - c)
    return np.where(
        discriminant < 0.0,
        np.nan,
        (b - np.sqrt(np.maximum(discriminant, 0.0))) / (-2.0 * a),
    )


def build_once(project_dir, kite_name, panels_per_section, overrides=None,
               system_config_path=None):
    """Everything that does not change across the chains.

    ``overrides`` is applied to the config BEFORE the VSM solver is built.
    It has to be: aerodynamic_vsm.initialize constructs the Solver from the
    config it is handed, so a key set afterwards is silently ignored and the
    run quietly keeps the old value -- which looks like a result, not a bug.

    ``system_config_path`` selects a system.yaml variant (e.g. the as-flown
    ``system_flown_2019.yaml``); default the kite's own ``system.yaml``. The
    KCU mass is read from it, by the geometry reader and the system model both.
    """
    config_path, aero_geometry_path, _ = resolve_kite_paths(project_dir, kite_name)
    struc_geometry_path = project_dir / "data" / kite_name / STRUC_GEOMETRY_FILENAME
    system_config_path = Path(
        system_config_path or project_dir / "data" / kite_name / "system.yaml"
    )
    with system_config_path.open("r", encoding="utf-8") as handle:
        system_config = _yaml.safe_load(handle)

    config = load_yaml(config_path)
    config["structural_solver"] = "billow"
    config.setdefault("aero2struc", {})["chordwise_distribution"] = "moment_matched"
    config["aerodynamic"]["n_aero_panels_per_struc_section"] = panels_per_section
    config["steering_tape_final_extension"] = 0.0
    config["steering_tape_extension_step"] = 0.0
    cp_rel = config.get("aero2struc", {}).get("cp_distribution_path")
    if cp_rel:
        config["aero2struc"]["cp_distribution_path"] = str(project_dir / cp_rel)

    for section, values in (overrides or {}).items():
        config.setdefault(section, {}).update(values)

    struc_geometry = load_yaml(struc_geometry_path)
    n_struc_ribs = len(struc_geometry["wing_particles"]["data"]) / 2
    n_panels = (n_struc_ribs - 1) * panels_per_section
    bridle_path = (
        struc_geometry_path if config.get("is_with_aero_bridle", False) else None
    )
    body_aero, vsm_solver, vel_app, polars = aerodynamic_vsm.initialize(
        aero_geometry_path, config, n_panels, bridle_path=bridle_path
    )
    tether_struct = get_tether(system_config)["structure"]
    return dict(
        config=config,
        struc_geometry=struc_geometry,
        system_config=system_config,
        system_config_path=system_config_path,
        body_aero=body_aero,
        vsm_solver=vsm_solver,
        vel_app=vel_app,
        polars=polars,
        tether=build_tether(config, system_config),
    )


def prepare_structure(shared, config):
    """Read the geometry and build the Billow model once, ready to be solved.

    Returned as a dict so a caller can hand the SAME structure to several
    coupled calls (``coupled_call``): Billow owns positions, frames and rest
    lengths across solves, so the second call warm-starts from the first one's
    equilibrium with its actuated tapes in place.
    """
    # The reader MUTATES the geometry it is handed (initialize_particles inserts
    # the strut padding into strut_tubes' node_indices), so every chain gets its
    # own copy or the second one walks off the end of an already-padded strut.
    geometry = copy.deepcopy(shared["struc_geometry"])
    reader = read_struc_geometry_yaml.main(
        geometry, config=config, system_config=shared["system_config"]
    )
    (struc_nodes, m_arr, le_indices, te_indices, power_tape_index,
     steering_tape_indices, _pn, canopy_sections, strut_sections, _sbp,
     connectivity, bridle_connectivity, bridle_diameter, l0_arr, k_arr, c_arr,
     link_types, pulley_line_indices, pulley_dict) = reader

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
    # instantiate() may ADD nodes (the cross canopy pattern puts one at each
    # quad centre), so the node count downstream is the model's, not the
    # reader's. Take the masses from the structure for the same reason: every
    # array the driver sizes -- external forces, tracking, the CG -- has to
    # agree with model.nodes or they silently misalign.
    struc_nodes = structure.model.nodes.copy()
    m_arr = structure.masses
    # The aerodynamic load is mapped onto these sections. Use the MEMBRANE's
    # grid, not the reader's: with a refined canopy the two differ, and mapping
    # onto the coarse one leaves every added node unloaded. aero2struc's
    # bracketing is symmetry-preserving for any section list, so a finer list
    # costs nothing but resolution gained.
    canopy_sections = [list(map(int, row)) for row in structure.fine_grid]
    strut_sections = []
    mapping = (
        BilinearAeroToStructuralLoadMapper()
        .initialize(shared["body_aero"].panels, struc_nodes, le_indices, te_indices)
        .panel_corner_map
    )
    return dict(
        geometry=geometry, structure=structure, struc_nodes=struc_nodes,
        built_nodes=struc_nodes.copy(), m_arr=m_arr, le_indices=le_indices,
        te_indices=te_indices, power_tape_index=power_tape_index,
        steering_tape_indices=steering_tape_indices, connectivity=connectivity,
        bridle_connectivity=bridle_connectivity, bridle_diameter=bridle_diameter,
        l0_arr=l0_arr, pulley_line_indices=pulley_line_indices,
        pulley_dict=pulley_dict, canopy_sections=canopy_sections,
        strut_sections=strut_sections, mapping=mapping,
        # One aero body per structure, not a fresh copy per call: the body is
        # mutated onto the deformed shape every iteration, so reusing it across
        # calls is simply more of the same iteration.
        body_aero=copy.deepcopy(shared["body_aero"]),
        vsm_solver=copy.deepcopy(shared["vsm_solver"]),
        polars=copy.deepcopy(shared["polars"]),
    )


def coupled_call(shared, prepared, config, reach=0.0, step=0.0, steer=0.0):
    """One coupled solve on a prepared structure, from wherever it stands now.

    The depower tape walks ``reach`` [m] from its CURRENT rest length in
    ``step`` increments (0 = one step) and the steering half-difference
    ``steer`` [m] from its current one, in ``config``'s
    ``steering_tape_extension_step``. The wind is ``config``'s, so a caller
    can change it between calls on the same structure.
    """
    config["power_tape_final_extension"] = float(reach)
    config["power_tape_extension_step"] = float(step or reach)
    config["steering_tape_final_extension"] = float(steer)
    structure = prepared["structure"]
    index = prepared["power_tape_index"]
    current_tape = structure_billow_rest_length(structure, index)
    # A structure that has been solved starts from its own state; a fresh one
    # from the reader's nodes. Billow's state IS the positions the next
    # minimum-energy solve starts from, so they are the consistent guess.
    struc_nodes = np.asarray(structure.state.positions, dtype=float).copy()
    n_steps = int(np.ceil(abs(reach) / abs(step) - 1e-9)) if reach and step else int(bool(reach))
    m_arr = prepared["m_arr"]
    return coupled_solver.main(
        m_arr=m_arr,
        struc_nodes=struc_nodes,
        struc_nodes_initial=prepared["built_nodes"],
        system_model=build_system_model(
            shared["system_config_path"], shared["tether"], m_arr, config
        ),
        config=config,
        initial_length_power_tape=current_tape,
        n_power_tape_steps=n_steps,
        power_tape_final_extension=config["power_tape_final_extension"],
        power_tape_extension_step=config["power_tape_extension_step"],
        kite_connectivity_arr=prepared["connectivity"],
        bridle_connectivity_arr=prepared["bridle_connectivity"],
        pulley_line_indices=prepared["pulley_line_indices"],
        pulley_line_to_other_node_pair_dict=prepared["pulley_dict"],
        struc_node_le_indices=prepared["le_indices"],
        struc_node_te_indices=prepared["te_indices"],
        body_aero=prepared["body_aero"],
        vsm_solver=prepared["vsm_solver"],
        vel_app=shared["vel_app"],
        initial_polar_data=prepared["polars"],
        bridle_diameter_arr=prepared["bridle_diameter"],
        aero2struc_mapping=prepared["mapping"],
        power_tape_index=index,
        # Only a steered call hands the driver its tapes: without them the
        # driver never actuates steering, whatever the config says.
        steering_tape_indices=prepared["steering_tape_indices"] if steer else None,
        billow_structure=structure,
        canopy_sections=prepared["canopy_sections"],
        strut_sections=prepared["strut_sections"],
    )


def structure_billow_rest_length(structure, element_index):
    """Live rest length [m] of one reader element on a Billow structure."""
    return float(structural_billow.get_rest_length(structure, element_index))


def run_chain(shared, wind_speed, reach, step, results_dir, steer=0.0, steer_step=0.0):
    """Walk the tape from its built length to ``reach``, converging at each step.

    ``steer`` [m] is a steering tape half-difference (first tape shortened,
    second lengthened), walked in ``steer_step`` increments -- 0 for one step
    -- once the depower walk has arrived. 0 leaves the kite unsteered.
    """
    config = copy.deepcopy(shared["config"])
    config["wind_speed_wind_ref"] = float(wind_speed)
    config["steering_tape_extension_step"] = float(steer_step)
    prepared = prepare_structure(shared, config)
    tracking, meta = coupled_call(shared, prepared, config, reach, step, steer)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_sim_output(tracking, meta, results_dir)
    return tracking, meta, prepared["geometry"], prepared["structure"]


def converged_steps(tracking, n_iter, tolerance):
    """Indices of the converged iterate at each distinct tape length.

    The driver only actuates once the residual gate is met, so the iteration
    immediately BEFORE a tape change is a converged state. Taking the last
    iterate at each length therefore takes the converged one, and the final
    iterate closes the last step.
    """
    lengths = np.asarray(tracking["tape_length"])[:n_iter]
    residual = np.asarray(tracking["residual_norm"])[:n_iter]
    picks = []  # (index, converged)
    for index in range(len(lengths)):
        if not np.isfinite(lengths[index]):
            continue
        is_last = index == len(lengths) - 1
        changes = not is_last and not np.isclose(
            lengths[index + 1], lengths[index], atol=1e-9
        )
        if changes or is_last:
            # Report the verdict rather than filtering: a step dropped for
            # missing the gate looks identical to a step that was never run,
            # and a silently short chain reads as a complete one.
            picks.append((index, bool(residual[index] <= tolerance)))
    return picks


def span_metrics(positions, geometry):
    """Tip-to-tip span, leading-edge arc length and arch rise [m]."""
    chain = structural_billow.leading_edge_chain(geometry)
    edge = positions[chain]
    start, end = edge[0], edge[-1]
    axis = end - start
    span = float(np.linalg.norm(axis))
    axis = axis / span
    relative = edge - start
    rise = float(np.linalg.norm(relative - np.outer(relative @ axis, axis), axis=1).max())
    return span, float(np.linalg.norm(np.diff(edge, axis=0), axis=1).sum()), rise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind", type=float, nargs="+", default=[3.0, 4.2, 5.5],
                        help="wind speeds at the reference height [m/s]")
    parser.add_argument("--reach", type=float, default=0.2,
                        help="how far the tape walks each way [m]")
    parser.add_argument("--step", type=float, default=0.05, help="tape step [m]")
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[3]
    shared = build_once(project_dir, args.kite, args.panels_per_section)
    gate = float(shared["config"]["aero_structural_solver"]["tol"])
    root = aerostructural_results_root(project_dir, args.kite) / "billow_depower_chains"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "chains.csv"

    fields = ["wind_speed", "direction", "tape_length", "u_dp", "speed_apparent",
              "residual", "span", "le_arc", "arch_rise", "aero_force", "converged"]
    rows, started = [], time.perf_counter()

    print(f"{len(args.wind)} winds x 2 directions, tape +/-{args.reach} m "
          f"in {args.step} m steps\n")
    print(f"{'wind':>5} {'dir':>5} {'l_dp':>6} {'u_dp':>6} {'v_a':>6} | "
          f"{'res':>6} | {'span':>6} {'rise':>6} | {'F':>7}")
    print("-" * 74)

    for wind in args.wind:
        for direction, reach in (("down", -abs(args.reach)), ("up", +abs(args.reach))):
            folder = root / f"wind{wind:04.1f}_{direction}".replace(".", "p")
            chain_started = time.perf_counter()
            try:
                tracking, meta, geometry, _ = run_chain(
                    shared, wind, reach, args.step, folder
                )
            except Exception as error:                      # keep the grid going
                print(f"{wind:5.2f} {direction:>5}  chain FAILED: {error}", flush=True)
                continue

            n_iter = int(meta["n_iter"])
            for index, converged in converged_steps(tracking, n_iter, gate):
                positions = np.asarray(tracking["positions"][index])
                span, arc, rise = span_metrics(positions, geometry)
                length = float(np.asarray(tracking["tape_length"])[index])
                row = dict(
                    wind_speed=wind, direction=direction, tape_length=length,
                    u_dp=float(depower_input(length)),
                    speed_apparent=float(np.asarray(tracking["speed_apparent"])[index]),
                    residual=float(np.asarray(tracking["residual_norm"])[index]),
                    span=span, le_arc=arc, arch_rise=rise,
                    aero_force=float(np.linalg.norm(
                        np.asarray(tracking["f_ext"][index]).sum(axis=0))),
                    converged=converged,
                )
                rows.append(row)
                print(f"{wind:5.2f} {direction:>5} {length:6.3f} {row['u_dp']:6.3f} "
                      f"{row['speed_apparent']:6.2f} | {row['residual']:6.2f} | "
                      f"{span:6.3f} {rise:6.3f} | {row['aero_force']:7.0f}"
                      f"{'' if converged else '   NOT CONVERGED'}", flush=True)

            print(f"      -> chain in {(time.perf_counter() - chain_started)/60:.1f} min",
                  flush=True)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    print("-" * 74)
    print(f"{len(rows)} converged points in "
          f"{(time.perf_counter() - started)/60:.1f} min -> {csv_path}")


if __name__ == "__main__":
    main()

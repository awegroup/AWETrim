"""
2-D parameter sweep: wind reference speed × steering tape extension.

Wind speed is taken from the kite aerostructural config.
Steering values are generated from STEERING_START_M, STEERING_END_M, and
STEERING_N_VALUES. If STEERING_N_VALUES is None, STEERING_STEP_M is used.
Every result row is appended to SUMMARY_CSV_NAME inside
results/aerostructural/<kite_name>/.

The CSV includes:
  - wind_speed_wind_ref, steering, depower actuation
  - angle of attack (aoa_deg), side-slip (side_slip_deg)
  - lift and drag coefficients (cl, cd)
  - quasi-steady optimised state (kite speed, roll, pitch, yaw, course rate)
  - convergence flag

NOTE on AOA calculation:
  The AOA in results is `aoa_deg` = atan2(v_z, v_x) in the course frame, where:
    - v_x is apparent velocity along course direction (horizontal)
    - v_z is apparent velocity along radial direction (vertical/lift)
  This gives the pitch angle of the apparent wind relative to the kite.
  This value accounts for the kite's trim attitude rotations and includes
  induced velocity effects from the VSM solver.
"""

import copy
import logging
from pathlib import Path

import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403  (sets up root logger)
from awetrim.aerostructural.actuation import update_steering_tape_actuation
from awetrim.aerostructural.mapping import BilinearAeroToStructuralLoadMapper
from awetrim.aerostructural.results import (
    aerostructural_results_root,
    append_sweep_csv_row,
    build_sweep_csv_row,
    save_input_snapshot,
    save_sim_output,
    steering_values_from_count_or_step,
)
from awetrim.aerostructural.utils import (
    load_yaml,
    rotate_geometry,
)
from awetrim.aerostructural import (
    aerodynamic_vsm,
    aerostructural_coupled_solver_qsm,
    structural_geometry_io,
    structural_pss,
)
from awetrim.system.system_model import SystemModel
from awetrim.system.tether import RigidLumpedTether
from awesio.validator import validate as awesio_validate
from common import (
    CONFIG_DEFAULTS,
    DEFAULT_KITE_NAME,
    build_actuation_case_folder,
    configure_system_model_from_config,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
    resolve_starting_rest_lengths,
    resolve_starting_struc_nodes,
)

# ── Sweep parameters ─────────────────────────────────────────────────────────
# Wind speed is taken from config file (wind_speed_wind_ref)

STEERING_START_M: float = 0.0  # m
STEERING_END_M: float = 0.4  # m
STEERING_N_VALUES: int | None = 9  # inclusive count from START to END
STEERING_STEP_M: float = 0.05  # used only when STEERING_N_VALUES is None

# Depower tape extension applied to every run in the sweep.
# Set DEPOWER_STEP_M = 0 to jump in a single step (recommended for sweep use).
DEPOWER_M: float = 0.0  # m  — final power-tape extension
DEPOWER_STEP_M: float = 0.0  # m  — actuation step size (0 = single step)

SUMMARY_CSV_NAME: str = "sweep_wind_steering.csv"
# ─────────────────────────────────────────────────────────────────────────────


def _parse_steering_from_case_folder(case_dir_or_name: str) -> float:
    """
    Extract steering amount (in meters) from case folder name.
    E.g., "depower_p0000mm_steer_p0150mm" -> 0.150 m
    """
    case_name = str(case_dir_or_name).split("/")[-1]  # Handle both Path and string
    if "steer_" not in case_name:
        return 0.0
    try:
        steer_part = case_name.split("steer_")[1]  # e.g., "p0150mm"
        sign_char = steer_part[0]  # 'p' or 'm'
        milli_str = steer_part[1:5]  # e.g., "0150"
        milli = int(milli_str)
        value_m = milli / 1000.0
        if sign_char == "m":
            value_m = -value_m
        return value_m
    except (IndexError, ValueError):
        return 0.0


# ── Main sweep ────────────────────────────────────────────────────────────────


def main() -> None:
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    kite_name = DEFAULT_KITE_NAME

    # Resolve standard kite paths (config, aero_geometry, struc_geometry)
    config_path, aero_geometry_path, struc_geometry_path = resolve_kite_paths(
        PROJECT_DIR, kite_name
    )

    # Build steering values
    steering_values = steering_values_from_count_or_step(
        STEERING_START_M,
        STEERING_END_M,
        n_values=STEERING_N_VALUES,
        step_m=STEERING_STEP_M,
    )

    results_root = aerostructural_results_root(PROJECT_DIR, kite_name)
    summary_csv_path = results_root / SUMMARY_CSV_NAME

    total_runs = len(steering_values)
    run_idx = 0

    # ── One-time: load base config and geometry ───────────────────────────────
    system_config_path = Path(PROJECT_DIR) / "data" / kite_name / "system.yaml"
    import yaml as _yaml

    with system_config_path.open("r", encoding="utf-8") as _f:
        system_config = _yaml.safe_load(_f)
    awesio_validate(system_config, restrictive=False)

    base_config = load_yaml(config_path)
    struc_geometry = load_yaml(struc_geometry_path)

    # Structural arrays from geometry (shared – steering actuation modifies
    # spring lengths inside psystem at runtime, not these arrays directly)
    (
        struc_nodes_base,
        m_arr,
        struc_node_le_indices,
        struc_node_te_indices,
        power_tape_index,
        steering_tape_indices,
        pulley_node_indices,
        kite_connectivity_arr,
        bridle_connectivity_arr,
        bridle_diameter_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        pulley_line_indices,
        pulley_line_to_other_node_pair_dict,
    ) = structural_geometry_io.main(
        struc_geometry, config=base_config, system_config=system_config
    )

    # Apply initial geometry rotation once
    struc_nodes_base = rotate_geometry(
        struc_nodes_base,
        **resolve_initial_geometry_rotation_kwargs(base_config),
    )

    # ── One-time: initialise aerodynamic solver ───────────────────────────────
    n_wing_struc_nodes = len(struc_geometry["wing_particles"]["data"])
    n_struc_ribs = n_wing_struc_nodes / 2
    n_panels_aero = (n_struc_ribs - 1) * base_config["aerodynamic"][
        "n_aero_panels_per_struc_section"
    ]
    bridle_path = (
        struc_geometry_path if base_config.get("is_with_aero_bridle", False) else None
    )
    body_aero_init, vsm_solver_init, vel_app, initial_polar_data_init = (
        aerodynamic_vsm.initialize(
            aero_geometry_path,
            base_config,
            n_panels_aero,
            bridle_path=bridle_path,
        )
    )

    # ── Aero–structure mapping (created per-run with current geometry) ────────────
    # Note: Must be created inside loop to match run_simulation_level_qsm.py behavior
    # which creates it after geometry setup/recovery

    # ── Track previous case for recovery sequence ──────────────────────────────
    previous_case_dir = None

    # Extract wind speed from config (single wind speed per sweep)
    wind_speed = base_config.get("wind_speed_wind_ref", 6.0)

    # ── Loop: steering values ─────────────────────────────────────────────────
    for steering in steering_values:
        run_idx += 1
        logging.info(
            "\n=== Sweep run %d/%d: wind=%.2f m/s  depower=%.4f m  steering=%.4f m ===",
            run_idx,
            total_runs,
            wind_speed,
            DEPOWER_M,
            steering,
        )

        # Build a per-run config copy with the current wind/steering/depower values
        cfg = copy.deepcopy(base_config)
        cfg["wind_speed_wind_ref"] = wind_speed
        cfg["steering_tape_final_extension"] = float(steering)
        cfg["power_tape_final_extension"] = float(DEPOWER_M)
        # Step size: use DEPOWER_STEP_M if > 0, otherwise a single step equal
        # to the full extension (avoids divide-by-zero in the coupled solver).
        cfg["power_tape_extension_step"] = (
            float(DEPOWER_STEP_M)
            if DEPOWER_STEP_M > 0
            else float(DEPOWER_M) if DEPOWER_M != 0 else 0.0
        )

        # Determine output directory for this run
        case_folder = build_actuation_case_folder(cfg)
        case_dir = results_root / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)

        # Persist effective config, and input geometries when enabled in config.
        results_dir = save_input_snapshot(
            config=cfg,
            struc_geometry_path=struc_geometry_path,
            aero_geometry_path=aero_geometry_path,
            results_dir=case_dir,
        )

        # Recover final struc_nodes and rest_lengths from PREVIOUS run in sequence
        # (not from current case folder; only after first run)
        if run_idx > 1 and previous_case_dir is not None:
            struc_nodes_recovered = resolve_starting_struc_nodes(
                previous_case_dir, struc_nodes_base
            )
            l0_arr_active = resolve_starting_rest_lengths(previous_case_dir, l0_arr)
        else:
            # First run: use defaults
            struc_nodes_recovered = None
            l0_arr_active = l0_arr
        initial_length_power_tape = l0_arr_active[power_tape_index]

        # Use recovered nodes if available, else start fresh
        struc_nodes = (
            struc_nodes_recovered.copy()
            if struc_nodes_recovered is not None
            else struc_nodes_base.copy()
        )
        psystem, pss_initial_conditions, pss_params, struc_nodes_initial = (
            structural_pss.instantiate(
                cfg,
                struc_nodes,
                m_arr,
                kite_connectivity_arr,
                l0_arr_active,
                k_arr,
                c_arr,
                linktype_arr,
                pulley_line_to_other_node_pair_dict,
            )
        )

        if cfg["is_with_initial_structure_plot"]:
            structural_pss.plot_3d_kite_structure(
                struc_nodes,
                kite_connectivity_arr,
                power_tape_index,
                k_arr=k_arr,
                c_arr=c_arr,
                linktype_arr=linktype_arr,
                pulley_nodes=pulley_node_indices,
            )

        # Apply steering actuation
        steering_tape_extension_step = cfg.get("steering_tape_extension_step", 0.0)
        steering_tape_final_extension = cfg.get("steering_tape_final_extension", 0.0)

        # When recovering from a previous run, apply INCREMENTAL steering
        # (not the absolute value from config)
        steering_to_apply = steering_tape_final_extension
        if run_idx > 1 and previous_case_dir is not None:
            previous_steering = _parse_steering_from_case_folder(previous_case_dir.name)
            steering_to_apply = steering_tape_final_extension - previous_steering
            logging.info(
                f"Steering adjustment: current={steering_tape_final_extension:.4f}m - "
                f"previous={previous_steering:.4f}m = incremental={steering_to_apply:.4f}m"
            )

        # Apply steering if adjustment != 0 (step=0 means single application, step>0 means gradual)
        if abs(float(steering_to_apply)) > 1e-9:
            # Use step size from config, or incremental extension for single application (step=0)
            effective_step = (
                float(steering_tape_extension_step)
                if steering_tape_extension_step != 0
                else steering_to_apply
            )
            update_steering_tape_actuation(
                psystem=psystem,
                steering_tape_indices=steering_tape_indices,
                steering_tape_extension_step=effective_step,
                initial_length_steering_left=l0_arr_active[steering_tape_indices[0]],
                initial_length_steering_right=l0_arr_active[steering_tape_indices[1]],
                steering_tape_final_extension=steering_to_apply,
            )

        # Power-tape actuation
        power_tape_extension_step = cfg.get("power_tape_extension_step", 0.0)
        power_tape_final_extension = cfg.get("power_tape_final_extension", 0.0)
        n_power_tape_steps = (
            int(power_tape_final_extension / power_tape_extension_step)
            if power_tape_extension_step != 0
            else 0
        )

        # ── SystemModel ───────────────────────────────────────────────────
        tether_struct = system_config["components"]["tether"]["structure"]
        tether = RigidLumpedTether(
            diameter=tether_struct["diameter"],
            density=tether_struct.get("density", 970.0),
        )
        system_model = SystemModel(tether=tether)
        system_model.mass_wing = float(np.sum(m_arr))
        configure_system_model_from_config(system_model, cfg)

        # ── Aero–structure mapping (created per-run with current geometry) ────────
        aero2struc_mapping = (
            BilinearAeroToStructuralLoadMapper()
            .initialize(
                body_aero_init.panels,
                struc_nodes,  # Use current struc_nodes (may be recovered)
                struc_node_le_indices,
                struc_node_te_indices,
            )
            .panel_corner_map
        )

        # ── Run coupled solver ────────────────────────────────────────────
        # Deepcopy aero objects so each run starts from a clean initial state.
        tracking_data, meta = aerostructural_coupled_solver_qsm.main(
            m_arr=m_arr,
            struc_nodes=struc_nodes,
            struc_nodes_initial=struc_nodes_initial,
            system_model=system_model,
            config=cfg,
            # Actuation
            initial_length_power_tape=initial_length_power_tape,
            n_power_tape_steps=n_power_tape_steps,
            power_tape_final_extension=power_tape_final_extension,
            power_tape_extension_step=power_tape_extension_step,
            # Connectivity
            kite_connectivity_arr=kite_connectivity_arr,
            bridle_connectivity_arr=bridle_connectivity_arr,
            pulley_line_indices=pulley_line_indices,
            pulley_line_to_other_node_pair_dict=pulley_line_to_other_node_pair_dict,
            # Struc → Aero
            struc_node_le_indices=struc_node_le_indices,
            struc_node_te_indices=struc_node_te_indices,
            # Aero
            body_aero=copy.deepcopy(body_aero_init),
            vsm_solver=copy.deepcopy(vsm_solver_init),
            vel_app=vel_app,
            initial_polar_data=copy.deepcopy(initial_polar_data_init),
            bridle_diameter_arr=bridle_diameter_arr,
            # Aero → Struc
            aero2struc_mapping=aero2struc_mapping,
            power_tape_index=power_tape_index,
            # Struc
            psystem=psystem,
        )

        # Save simulation output
        save_sim_output(tracking_data, meta, results_dir)

        # Structural warm-start: PSS particle positions (pre-rotation) for next step.
        final_nodes = meta.get("final_struc_nodes")
        if final_nodes is not None and np.all(np.isfinite(final_nodes)):
            struc_nodes_warm = np.asarray(final_nodes, dtype=float)

        # QSM warm-start: carry converged opt_x to the next steering step.
        opt_x = np.asarray(meta.get("opt_x", []), dtype=float)

        logging.info(
            "Run complete: wind=%.2f m/s  steering=%.4f m  opt_x=%s",
            wind_speed,
            steering,
            opt_x,
        )

        # Append to sweep summary CSV
        row = build_sweep_csv_row(
            wind_speed=wind_speed,
            steering=float(steering),
            config=cfg,
            meta=meta,
            case_folder=case_folder,
            results_dir=results_dir,
            config_defaults=CONFIG_DEFAULTS,
            power_tape_index=power_tape_index,
            steering_tape_indices=steering_tape_indices,
        )
        append_sweep_csv_row(summary_csv_path, row)

        logging.info(
            "Run %d/%d done | depower=%.4f m  steering=%.4f m | "
            "aoa=%.2f deg  side_slip=%.2f deg  converged=%s",
            run_idx,
            total_runs,
            DEPOWER_M,
            float(steering),
            row["aoa_deg"],
            row["side_slip_deg"],
            row["converged"],
        )

        # Track this case for next run's recovery
        previous_case_dir = case_dir

    logging.info(
        "\nSweep complete (wind=%.2f m/s). Summary CSV: %s",
        wind_speed,
        summary_csv_path,
    )


if __name__ == "__main__":
    main()

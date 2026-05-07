"""
3-D parameter sweep: course angle × steering tape extension × depower tape extension.

Configurable sweep parameters: easily adjust COURSE_ANGLES_DEG, STEERING_*, and DEPOWER_*
at the top of this file to change sweep scope without rewriting code.

All combinations are run in-process and every result row is appended to a CSV inside
results/aerostructural/<kite_name>/. The CSV filename includes the sweep counts (e.g.,
sweep_c1_s6_d1.csv for 1 course × 6 steering × 1 depower = 6 total cases).

The CSV includes:
  - course angle, steering, depower actuation
  - angle of attack (aoa_deg), side-slip (side_slip_deg)
  - lift and drag coefficients (cl, cd)
  - quasi-steady optimised state (kite speed, roll, pitch, yaw, course rate)
  - convergence flag

Examples:
  - Single depower only:  Set DEPOWER_START_M = DEPOWER_END_M = 0.0
  - 5-point steering sweep: Set STEERING_START_M = 0.0, STEERING_END_M = 0.20, STEERING_N_VALUES = 5
  - Single course angle: Set COURSE_ANGLES_DEG = [90.0]
"""

import copy
import logging
from pathlib import Path

import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403  (sets up root logger)
from awetrim.aerostructural.mapping import BilinearAeroToStructuralLoadMapper
from awetrim.aerostructural.results import (
    aerostructural_results_root,
    append_sweep_csv_row,
    build_sweep_csv_row,
    candidate_case_dirs,
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
    configure_system_model_from_config,
    format_length_tag,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
    resolve_starting_rest_lengths,
    resolve_starting_struc_nodes,
)

# ── Sweep parameters ─────────────────────────────────────────────────────────
# Easily configure each parameter:
#   - COURSE_ANGLES_DEG: List of course angles (degrees)
#   - STEERING: start, end, step (meters)
#   - DEPOWER: start, end, step (meters)

# Example configurations:
#   Single depower, multiple steering: set DEPOWER_START_M == DEPOWER_END_M
#   Single steering, multiple depower: set STEERING_START_M == STEERING_END_M
#   Single course: COURSE_ANGLES_DEG = [90.0]

COURSE_ANGLES_DEG: list = [90.0]  # Course angle in degrees
STEERING_START_M: float = 0.0  # m
STEERING_END_M: float = 0.5  # m
STEERING_N_VALUES: int | None = 11  # inclusive count from START to END
STEERING_STEP_M: float = 0.05  # m, used only when STEERING_N_VALUES is None
DEPOWER_START_M: float = 0.0  # m|
DEPOWER_END_M: float = 0.8  # m (only depower=0.0)
DEPOWER_STEP_M: float = 0.2  # m (step size; ignored if START==END)

# Calculate total cases (will be shown at runtime)
_n_course = len(COURSE_ANGLES_DEG)
_steering_values = steering_values_from_count_or_step(
    STEERING_START_M,
    STEERING_END_M,
    n_values=STEERING_N_VALUES,
    step_m=STEERING_STEP_M,
)
_depower_values = (
    np.arange(DEPOWER_START_M, DEPOWER_END_M + 0.5 * DEPOWER_STEP_M, DEPOWER_STEP_M)
    if DEPOWER_STEP_M > 0
    else [DEPOWER_START_M]
)
_n_steering = len(_steering_values)
_n_depower = len(_depower_values)
_total_cases = _n_course * _n_steering * _n_depower

# CSV filename with sweep parameters (editable for custom names)
SUMMARY_CSV_NAME: str = f"sweep_c{_n_course}_s{_n_steering}_d{_n_depower}.csv"

# ─────────────────────────────────────────────────────────────────────────────


def _parse_length_tag_mm(tag: str) -> float:
    """Parse signed mm tag like p0200mm/m0050mm into meters."""
    clean = str(tag).strip()
    if len(clean) < 4 or not clean.endswith("mm"):
        raise ValueError(f"Invalid length tag format: {tag}")

    sign_char = clean[0]
    if sign_char == "p":
        sign = 1.0
    elif sign_char == "m":
        sign = -1.0
    else:
        raise ValueError(f"Invalid length tag sign in: {tag}")

    milli = int(clean[1:-2])
    return sign * (milli / 1000.0)


def _infer_source_extensions_from_subdir(sim_subdir: str) -> tuple[float, float]:
    """Infer (depower, steering) source extensions [m] from folder naming."""
    if str(sim_subdir).strip() == "":
        return 0.0, 0.0

    name = str(sim_subdir).replace("\\", "/").strip("/").split("/")[-1]
    depower_marker = "depower_"
    steer_marker = "_steer_"

    if depower_marker not in name or steer_marker not in name:
        logging.warning(
            "Could not infer source extensions from starting_from_sim_subdir='%s'; assuming 0.0m.",
            sim_subdir,
        )
        return 0.0, 0.0

    depower_start = name.index(depower_marker) + len(depower_marker)
    steer_start = name.index(steer_marker, depower_start)
    depower_tag = name[depower_start:steer_start]
    steering_tag = name[steer_start + len(steer_marker) :]

    try:
        return _parse_length_tag_mm(depower_tag), _parse_length_tag_mm(steering_tag)
    except Exception:
        logging.warning(
            "Failed parsing source extensions from starting folder '%s'; assuming 0.0m.",
            name,
        )
        return 0.0, 0.0


def _build_actuation_case_folder(config: dict) -> str:
    """Course-angle-aware variant of build_actuation_case_folder for 3-D sweeps."""
    depower_tag = format_length_tag(
        config.get(
            "power_tape_final_extension", CONFIG_DEFAULTS["power_tape_final_extension"]
        )
    )
    steering_tag = format_length_tag(config.get("steering_tape_final_extension", 0.0))
    course_tag = f"course_{int(config.get('angle_course_deg', CONFIG_DEFAULTS['angle_course_deg'])):03d}deg"
    return f"{course_tag}_depower_{depower_tag}_steer_{steering_tag}"


def _select_nearest_case(
    target_course: float,
    target_steering: float,
    target_depower: float,
    simulated_cases: list,
    course_scale: float,
    steering_scale: float,
    depower_scale: float,
):
    """Return nearest simulated case in normalized (course, steering, depower) space."""
    if not simulated_cases:
        return None

    def _dist2(case):
        dc = (float(target_course) - float(case["course"])) / course_scale
        ds = (float(target_steering) - float(case["steering"])) / steering_scale
        dd = (float(target_depower) - float(case["depower"])) / depower_scale
        return dc * dc + ds * ds + dd * dd

    return min(simulated_cases, key=_dist2)


# ── Main sweep ────────────────────────────────────────────────────────────────


def main() -> None:
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    kite_name = DEFAULT_KITE_NAME

    # Resolve standard kite paths (config, aero_geometry, struc_geometry)
    config_path, aero_geometry_path, struc_geometry_path = resolve_kite_paths(
        PROJECT_DIR, kite_name
    )

    # Build parameter ranges
    course_values = COURSE_ANGLES_DEG
    steering_values = steering_values_from_count_or_step(
        STEERING_START_M,
        STEERING_END_M,
        n_values=STEERING_N_VALUES,
        step_m=STEERING_STEP_M,
    )
    # Handle depower: if START == END, use single value; otherwise sweep
    if DEPOWER_START_M == DEPOWER_END_M:
        depower_values = np.array([DEPOWER_START_M])
    else:
        depower_values = np.arange(
            DEPOWER_START_M,
            DEPOWER_END_M + 0.5 * DEPOWER_STEP_M,
            DEPOWER_STEP_M,
        )

    results_root = aerostructural_results_root(PROJECT_DIR, kite_name)
    summary_csv_path = results_root / SUMMARY_CSV_NAME

    total_runs = len(course_values) * len(steering_values) * len(depower_values)

    # Log sweep configuration
    logging.info("\n" + "=" * 80)
    logging.info("SWEEP CONFIGURATION")
    logging.info("=" * 80)
    logging.info(f"Course angles:    {course_values} ({len(course_values)} values)")
    if STEERING_N_VALUES is not None:
        logging.info(
            f"Steering range:   {STEERING_START_M:.4f} to {STEERING_END_M:.4f} m "
            f"(count: {STEERING_N_VALUES}) -> {len(steering_values)} values"
        )
    else:
        logging.info(
            f"Steering range:   {STEERING_START_M:.4f} to {STEERING_END_M:.4f} m "
            f"(step: {STEERING_STEP_M:.4f} m) -> {len(steering_values)} values"
        )
    logging.info(
        f"Depower range:    {DEPOWER_START_M:.4f} to {DEPOWER_END_M:.4f} m (step: {DEPOWER_STEP_M:.4f} m) → {len(depower_values)} values"
    )
    logging.info(f"Total runs:       {total_runs}")
    logging.info(f"Output CSV:       {summary_csv_path}")
    logging.info("=" * 80 + "\n")

    run_idx = 0

    # ── One-time: load base config and geometry ───────────────────────────────
    system_config_path = Path(PROJECT_DIR) / "data" / kite_name / "system.yaml"
    import yaml as _yaml

    with system_config_path.open("r", encoding="utf-8") as _f:
        system_config = _yaml.safe_load(_f)
    awesio_validate(system_config, restrictive=False)

    base_config = load_yaml(config_path)
    struc_geometry = load_yaml(struc_geometry_path)

    # Structural arrays from geometry
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

    # Extract wind speed from config (single wind speed for entire sweep)
    wind_speed = base_config.get("wind_speed_wind_ref", 6.0)

    # Resolve optional fixed restart source from config (same behavior as single-run).
    start_subdir = str(base_config.get("starting_from_sim_subdir", "")).strip()
    if start_subdir == "":
        start_subdir = str(base_config.get("starting_from_sim_of_date", "")).strip()

    source_depower_from_start, source_steering_from_start = (
        _infer_source_extensions_from_subdir(start_subdir)
    )

    start_case_dir = None
    if start_subdir != "":
        candidates = candidate_case_dirs(PROJECT_DIR, kite_name, start_subdir)

        for cand in candidates:
            if cand.exists() and cand.is_dir():
                start_case_dir = cand
                break

        if start_case_dir is None:
            raise FileNotFoundError(
                "Configured starting simulation directory does not exist for sweep. "
                f"Tried: {', '.join(str(c) for c in candidates)}"
            )
        logging.info(
            "Sweep initialization mode: always start from configured simulation folder: %s",
            start_case_dir,
        )
        logging.info(
            "Source extensions from configured start folder: depower=%.4fm, steering=%.4fm",
            source_depower_from_start,
            source_steering_from_start,
        )
    else:
        logging.info(
            "Sweep initialization mode: always start from baseline initial state"
        )

    # ── Loop: THREE nested loops for course × steering × depower ──────────────
    for course_angle in course_values:
        for steering in steering_values:
            for depower in depower_values:
                run_idx += 1
                logging.info(
                    "\n=== Sweep run %d/%d: "
                    "course=%.1f deg  steering=%.4f m  depower=%.4f m  wind=%.2f m/s ===",
                    run_idx,
                    total_runs,
                    course_angle,
                    steering,
                    depower,
                    wind_speed,
                )

                # Build per-run config with current sweep parameters
                cfg = copy.deepcopy(base_config)
                cfg["wind_speed_wind_ref"] = wind_speed
                cfg["angle_course_deg"] = float(course_angle)
                cfg["steering_tape_final_extension"] = float(steering)
                cfg["power_tape_final_extension"] = float(depower)
                # Keep depower actuation step from config.
                # DEPOWER_STEP_M defines sweep grid spacing, not in-simulation actuation step.
                cfg["power_tape_extension_step"] = float(
                    base_config.get("power_tape_extension_step", 0.0)
                )

                # Determine output directory for this run
                case_folder = _build_actuation_case_folder(cfg)
                case_dir = results_root / case_folder
                case_dir.mkdir(parents=True, exist_ok=True)

                # Persist effective config, and input geometries when enabled in config.
                results_dir = save_input_snapshot(
                    config=cfg,
                    struc_geometry_path=struc_geometry_path,
                    aero_geometry_path=aero_geometry_path,
                    results_dir=case_dir,
                )

                # Always initialize from the fixed config-selected source.
                source_steering = float(source_steering_from_start)
                source_depower = float(source_depower_from_start)

                l0_arr_active = l0_arr
                if start_case_dir is not None:
                    struc_nodes_recovered = resolve_starting_struc_nodes(
                        start_case_dir, struc_nodes_base
                    )
                    l0_arr_active = resolve_starting_rest_lengths(
                        start_case_dir, l0_arr
                    )
                    if struc_nodes_recovered is None:
                        raise FileNotFoundError(
                            f"Configured start case has no valid sim_output positions: {start_case_dir}"
                        )
                    struc_nodes = struc_nodes_recovered.copy()
                    logging.info(
                        "Starting from configured simulation state: %s",
                        start_case_dir,
                    )
                else:
                    struc_nodes = struc_nodes_base.copy()
                    logging.info(
                        "Starting from baseline initial geometry and rest lengths"
                    )

                initial_length_power_tape = l0_arr_active[power_tape_index]

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
                steering_tape_extension_step = cfg.get(
                    "steering_tape_extension_step", 0.0
                )
                steering_tape_final_extension = cfg.get(
                    "steering_tape_final_extension", 0.0
                )

                steering_to_apply = (
                    float(steering_tape_final_extension) - source_steering
                )
                logging.info(
                    "Steering delta to apply: target %.4fm - source %.4fm = %.4fm",
                    float(steering_tape_final_extension),
                    source_steering,
                    float(steering_to_apply),
                )
                initial_length_steering_left = float(
                    l0_arr_active[steering_tape_indices[0]]
                )
                initial_length_steering_right = float(
                    l0_arr_active[steering_tape_indices[1]]
                )

                # Power-tape actuation
                power_tape_extension_step = cfg.get("power_tape_extension_step", 0.0)
                power_tape_final_extension = cfg.get("power_tape_final_extension", 0.0)

                depower_to_apply = float(power_tape_final_extension) - source_depower
                desired_length_power_tape = float(initial_length_power_tape) + float(
                    depower_to_apply
                )
                logging.info(
                    "Depower delta to apply: target %.4fm - source %.4fm = %.4fm | length %.3fm -> %.3fm",
                    float(power_tape_final_extension),
                    source_depower,
                    float(depower_to_apply),
                    float(initial_length_power_tape),
                    float(desired_length_power_tape),
                )

                # Two-level depower behavior:
                # 1) Sweep target (this file): DEPOWER_START/END/STEP define case grid.
                # 2) In-simulation progression (config): power_tape_extension_step ramps toward target.
                if (
                    abs(float(depower_to_apply)) > 1e-9
                    and abs(float(power_tape_extension_step)) <= 1e-12
                ):
                    logging.warning(
                        "Depower target is non-zero (%.4fm) but power_tape_extension_step in config is 0.0; "
                        "internal progressive depower will not move.",
                        float(depower_to_apply),
                    )

                n_power_tape_steps = (
                    int(np.ceil(abs(depower_to_apply) / abs(power_tape_extension_step)))
                    if power_tape_extension_step != 0
                    else 0
                )

                # ── SystemModel ───────────────────────────────────────────────────────

                tether_struct = system_config["components"]["tether"]["structure"]
                tether = RigidLumpedTether(
                    diameter=tether_struct["diameter"],
                    density=tether_struct.get("density", 970.0),
                )
                system_model = SystemModel(tether=tether)
                system_model.mass_wing = float(np.sum(m_arr))
                configure_system_model_from_config(system_model, cfg)

                # ── Aero–structure mapping (created per-run) ───────────────────────────
                aero2struc_mapping = (
                    BilinearAeroToStructuralLoadMapper()
                    .initialize(
                        body_aero_init.panels,
                        struc_nodes,
                        struc_node_le_indices,
                        struc_node_te_indices,
                    )
                    .panel_corner_map
                )

                # ── Run the coupled solver ─────────────────────────────────────────────
                converged = False
                tracking_data = None
                try:
                    tracking_data, meta = aerostructural_coupled_solver_qsm.main(
                        m_arr=m_arr,
                        struc_nodes=struc_nodes,
                        struc_nodes_initial=struc_nodes_initial,
                        system_model=system_model,
                        config=cfg,
                        # Actuation
                        initial_length_power_tape=initial_length_power_tape,
                        n_power_tape_steps=n_power_tape_steps,
                        power_tape_final_extension=depower_to_apply,
                        power_tape_extension_step=power_tape_extension_step,
                        initial_length_steering_left=initial_length_steering_left,
                        initial_length_steering_right=initial_length_steering_right,
                        steering_tape_indices=steering_tape_indices,
                        steering_tape_final_extension=steering_to_apply,
                        steering_tape_extension_step=steering_tape_extension_step,
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
                    converged = True
                except Exception as e:
                    logging.error(f"Solver failed: {e}")
                    meta = {}

                # Save run state for post-processing.
                if tracking_data is not None:
                    save_sim_output(tracking_data, meta, results_dir)

                # ── Append result to CSV ───────────────────────────────────────────────
                csv_row = build_sweep_csv_row(
                    wind_speed=wind_speed,
                    steering=steering,
                    config=cfg,
                    meta=meta,
                    case_folder=case_folder,
                    results_dir=results_dir,
                    config_defaults=CONFIG_DEFAULTS,
                    power_tape_index=power_tape_index,
                    steering_tape_indices=steering_tape_indices,
                )
                append_sweep_csv_row(summary_csv_path, csv_row)
                logging.info(f"Appended to CSV: {summary_csv_path}")

    logging.info(f"\n=== Sweep complete! Total runs: {total_runs} ===")
    logging.info(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

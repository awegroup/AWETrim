"""Result persistence helpers for aerostructural scripts."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from awetrim.aerostructural.utils import load_yaml, save_results

SAVE_GEOMETRY_SNAPSHOTS_KEY = "is_save_geometry_snapshots"

SWEEP_CSV_FIELDNAMES: list[str] = [
    "wind_speed_wind_ref_ms",
    "depower_tape_final_extension_m",
    "steering_tape_final_extension_m",
    "case_folder",
    "results_dir",
    "is_with_gravity",
    "is_with_aero_bridle",
    "angle_elevation_deg",
    "angle_azimuth_deg",
    "angle_course_deg",
    "speed_radial",
    "distance_radial",
    "timeder_speed_tangential",
    "timeder_speed_radial",
    "aoa_deg",
    "side_slip_deg",
    "aero_roll_deg",
    "cl",
    "cd",
    "converged",
    "depower_tape_final_length_m",
    "steering_tape_left_final_length_m",
    "steering_tape_right_final_length_m",
    "steering_tape_avg_length_m",
    "steering_tape_asymmetry_m",
    "opt_kite_speed",
    "opt_roll_deg",
    "opt_pitch_deg",
    "opt_yaw_deg",
    "opt_course_rate_body",
    "va",
    "tether_force",
]


def aerostructural_results_root(project_dir: Path, kite_name: str) -> Path:
    """Return the canonical aerostructural result directory for a kite."""
    return Path(project_dir) / "results" / kite_name / "aerostructural"


def legacy_results_root(project_dir: Path, kite_name: str) -> Path:
    """Return legacy result roots checked for warm-start recovery."""
    return Path(project_dir) / "results" / "aerostructural" / kite_name


def candidate_case_dirs(project_dir: Path, kite_name: str, case_subdir: str) -> list[Path]:
    """Return canonical and legacy candidate case directories for restart lookup."""
    normalized = str(case_subdir).strip()
    variants = [normalized]
    m_to_p = normalized.replace("m0000mm", "p0000mm")
    p_to_m = normalized.replace("p0000mm", "m0000mm")
    if m_to_p != normalized:
        variants.append(m_to_p)
    if p_to_m != normalized:
        variants.append(p_to_m)

    roots = [
        aerostructural_results_root(project_dir, kite_name),
        legacy_results_root(project_dir, kite_name),
    ]
    return [root / variant for root in roots for variant in variants]


def steering_values_from_count_or_step(
    start_m: float,
    end_m: float,
    *,
    n_values: int | None = None,
    step_m: float | None = None,
) -> np.ndarray:
    """Build an inclusive steering sweep from a point count or a step size."""
    if n_values is not None:
        if int(n_values) < 1:
            raise ValueError("n_values must be >= 1")
        if int(n_values) == 1:
            return np.asarray([float(start_m)], dtype=float)
        return np.linspace(float(start_m), float(end_m), int(n_values))

    if step_m is None or float(step_m) <= 0.0:
        raise ValueError("step_m must be > 0 when n_values is not provided")
    if float(end_m) < float(start_m):
        raise ValueError("end_m must be >= start_m")
    return np.arange(float(start_m), float(end_m) + 0.5 * float(step_m), float(step_m))


def save_input_snapshot(
    *,
    config: dict[str, Any],
    results_dir: Path,
) -> Path:
    """Create the results directory and save the effective run config."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    return results_dir


def build_deformed_struc_geometry(
    struc_geometry: dict[str, Any],
    struc_nodes: np.ndarray,
) -> dict[str, Any]:
    """Return a copy of struc_geometry with node positions replaced by deformed values.

    Node indices in the YAML are used directly as row indices into struc_nodes,
    matching the ordering established by structural_geometry_io.main().
    """
    sg = copy.deepcopy(struc_geometry)

    # KCU attachment point (node 0)
    if "bridle_point_node" in sg:
        sg["bridle_point_node"] = struc_nodes[0].tolist()

    # Wing particles
    for row in sg["wing_particles"]["data"]:
        nid = int(row[0])
        row[1], row[2], row[3] = float(struc_nodes[nid, 0]), float(struc_nodes[nid, 1]), float(struc_nodes[nid, 2])

    # Bridle particles (if present)
    if "bridle_particles" in sg:
        for row in sg["bridle_particles"]["data"]:
            nid = int(row[0])
            row[1], row[2], row[3] = float(struc_nodes[nid, 0]), float(struc_nodes[nid, 1]), float(struc_nodes[nid, 2])

    return sg


def build_deformed_aero_geometry(
    aero_geometry: dict[str, Any],
    struc_nodes: np.ndarray,
    le_indices: list[int],
    te_indices: list[int],
) -> dict[str, Any]:
    """Return a copy of aero_geometry with LE/TE positions replaced by deformed values.

    When the aero mesh is finer than the structural mesh, the same linear
    interpolation used by LinearStructuralToAeroMapper is applied to subdivide
    each structural section into the correct number of aero panels.  The
    airfoil_id column is preserved unchanged.
    """
    from awetrim.aerostructural.mapping import interpolate_points

    ag = copy.deepcopy(aero_geometry)
    sections = ag["wing_sections"]["data"]
    n_aero = len(sections)
    n_struc = len(le_indices)

    if n_aero == n_struc:
        n_panels_per_section = 1
    else:
        n_sections = n_struc - 1
        if n_sections == 0 or (n_aero - 1) % n_sections != 0:
            raise ValueError(
                f"Cannot infer panels-per-section from {n_aero} aero sections "
                f"and {n_struc} structural LE nodes."
            )
        n_panels_per_section = (n_aero - 1) // n_sections

    deformed_le = interpolate_points(struc_nodes[le_indices], n_panels_per_section)
    deformed_te = interpolate_points(struc_nodes[te_indices], n_panels_per_section)

    for i, row in enumerate(sections):
        le, te = deformed_le[i], deformed_te[i]
        row[1], row[2], row[3] = float(le[0]), float(le[1]), float(le[2])
        row[4], row[5], row[6] = float(te[0]), float(te[1]), float(te[2])
    return ag


def save_geometry_snapshot(
    config: dict[str, Any],
    struc_geometry_deformed: dict[str, Any],
    aero_geometry_deformed: dict[str, Any],
    results_dir: Path,
    system_yaml_path: Path | str | None = None,
) -> None:
    """Save deformed geometry YAMLs when is_save_geometry_snapshots is True.

    When ``system_yaml_path`` is given, also emit a ``system.yaml`` into
    ``results_dir`` whose mass, centre of gravity and inertia tensor are
    recomputed from the *deformed* struc geometry (the canonical source file is
    left untouched). This makes the case folder a self-consistent config folder:
    config + system + aero + struc all describe the same deformed shape.
    """
    if not bool(config.get(SAVE_GEOMETRY_SNAPSHOTS_KEY, False)):
        return
    results_dir = Path(results_dir)
    struc_path = results_dir / "struc_geometry.yaml"
    with struc_path.open("w", encoding="utf-8") as f:
        yaml.dump(struc_geometry_deformed, f, sort_keys=False)
    with (results_dir / "aero_geometry.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(aero_geometry_deformed, f, sort_keys=False)

    if system_yaml_path is not None:
        # Recompute the inertial properties (mass/CoG/inertia) from the deformed
        # struc geometry just written, writing the updated system.yaml alongside.
        from awetrim.utils.system_yml_sync import update_from_geometry

        update_from_geometry(
            system_yaml_path,
            struc_path,
            output_path=results_dir / "system.yaml",
        )


def save_sim_output(tracking_data: dict[str, Any], meta: dict[str, Any], results_dir: Path) -> Path:
    """Save the standard aerostructural HDF5 output and return its path."""
    h5_path = Path(results_dir) / "sim_output.h5"
    save_results(tracking_data, meta, h5_path)
    return h5_path


def append_sweep_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a sweep summary row using the shared aerostructural CSV schema."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_sweep_csv_row(
    *,
    wind_speed: float,
    steering: float,
    config: dict[str, Any],
    meta: dict[str, Any],
    case_folder: str,
    results_dir: Path,
    config_defaults: dict[str, Any],
    power_tape_index: int | None = None,
    steering_tape_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Flatten one QSM sweep result into the shared aerostructural CSV schema."""
    opt_x = np.asarray(meta.get("opt_x", []), dtype=float).reshape(-1)
    opt_names = ["kite_speed", "roll_deg", "pitch_deg", "yaw_deg", "course_rate_body"]

    row = {
        "wind_speed_wind_ref_ms": float(wind_speed),
        "depower_tape_final_extension_m": float(
            config.get("power_tape_final_extension", 0.0)
        ),
        "steering_tape_final_extension_m": float(steering),
        "case_folder": case_folder,
        "results_dir": str(results_dir),
        "is_with_gravity": bool(
            config.get("is_with_gravity", config_defaults["is_with_gravity"])
        ),
        "is_with_aero_bridle": bool(
            config.get("is_with_aero_bridle", config_defaults["is_with_aero_bridle"])
        ),
        "angle_elevation_deg": float(
            config.get("angle_elevation_deg", config_defaults["angle_elevation_deg"])
        ),
        "angle_azimuth_deg": float(
            config.get("angle_azimuth_deg", config_defaults["angle_azimuth_deg"])
        ),
        "angle_course_deg": float(
            config.get("angle_course_deg", config_defaults["angle_course_deg"])
        ),
        "speed_radial": float(config.get("speed_radial", config_defaults["speed_radial"])),
        "distance_radial": float(
            config.get("distance_radial", config_defaults["distance_radial"])
        ),
        "timeder_speed_tangential": float(
            config.get(
                "timeder_speed_tangential",
                config_defaults["timeder_speed_tangential"],
            )
        ),
        "timeder_speed_radial": float(
            config.get("timeder_speed_radial", config_defaults["timeder_speed_radial"])
        ),
        "aoa_deg": float(meta.get("aoa_deg", np.nan)),
        "side_slip_deg": float(meta.get("side_slip_deg", np.nan)),
        "aero_roll_deg": float(meta.get("aero_roll_deg", np.nan)),
        "cl": float(meta.get("cl", np.nan)),
        "cd": float(meta.get("cd", np.nan)),
        "converged": bool(meta.get("converged", False)),
    }

    rest_lengths = np.asarray(meta.get("rest_lengths", []), dtype=float)
    if power_tape_index is not None and rest_lengths.size > power_tape_index:
        row["depower_tape_final_length_m"] = float(rest_lengths[power_tape_index])
    else:
        row["depower_tape_final_length_m"] = np.nan

    if steering_tape_indices is not None and len(steering_tape_indices) >= 2:
        left_idx = int(steering_tape_indices[0])
        right_idx = int(steering_tape_indices[1])
        if rest_lengths.size > max(left_idx, right_idx):
            left_length = float(rest_lengths[left_idx])
            right_length = float(rest_lengths[right_idx])
            row["steering_tape_left_final_length_m"] = left_length
            row["steering_tape_right_final_length_m"] = right_length
            row["steering_tape_avg_length_m"] = (left_length + right_length) / 2.0
            row["steering_tape_asymmetry_m"] = (left_length - right_length) / 2.0
        else:
            row["steering_tape_left_final_length_m"] = np.nan
            row["steering_tape_right_final_length_m"] = np.nan
            row["steering_tape_avg_length_m"] = np.nan
            row["steering_tape_asymmetry_m"] = np.nan
    else:
        row["steering_tape_left_final_length_m"] = np.nan
        row["steering_tape_right_final_length_m"] = np.nan
        row["steering_tape_avg_length_m"] = np.nan
        row["steering_tape_asymmetry_m"] = np.nan

    for idx, name in enumerate(opt_names):
        row[f"opt_{name}"] = float(opt_x[idx]) if idx < opt_x.size else np.nan

    row["va"] = float(meta.get("va", np.nan))
    row["tether_force"] = float(meta.get("tether_force", np.nan))
    return row

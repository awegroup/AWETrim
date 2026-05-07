"""Shared helpers for all scripts/aerostructural/ entry points."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from awetrim.aerostructural.utils import load_sim_output
from awetrim.system.system_model import SystemModel

DEFAULT_KITE_NAME = "LEI-V3-KITE"

# Single source of truth for config defaults used across all aerostructural scripts.
CONFIG_DEFAULTS: dict = {
    "angle_elevation_deg": 0.0,
    "angle_azimuth_deg": 0.0,
    "angle_course_deg": 90.0,
    "speed_radial": 1.0,
    "distance_radial": 200.0,
    "wind_speed_wind_ref": 5.0,
    "timeder_speed_tangential": 0.0,
    "timeder_speed_radial": 0.0,
    "is_with_gravity": False,
    "is_with_aero_bridle": True,
    "power_tape_final_extension": 0.0,
    "initial_geometry_rotation_deg": 0.0,
    "initial_geometry_rotation_point": [0.0, 0.0, 0.0],
    "initial_geometry_rotation_axes": ["x", "y", "z"],
}


def resolve_kite_paths(
    project_dir: Path | str,
    kite_name: str = DEFAULT_KITE_NAME,
) -> tuple[Path, Path, Path | None]:
    """Resolve standard kite geometry and config paths.

    Returns:
        (config_path, aero_geometry_path, struc_geometry_path)
        where struc_geometry_path is None if no structural geometry file exists.
    """
    project_dir = Path(project_dir).resolve()

    config_path = project_dir / "data" / kite_name / "as_config.yaml"

    geometry_dir = project_dir / "data" / kite_name

    aero_geometry_path = geometry_dir / "aero_geometry.yaml"
    struc_geometry_path = geometry_dir / "struc_geometry.yaml"

    return config_path, aero_geometry_path, struc_geometry_path


def format_length_tag(value_m: float) -> str:
    """Format a signed length [m] as a filesystem-friendly tag, e.g. p0150mm."""
    sign = "p" if value_m >= 0 else "m"
    milli = int(round(abs(float(value_m)) * 1000.0))
    return f"{sign}{milli:04d}mm"


def build_actuation_case_folder(config: dict) -> str:
    """Build case folder name from final depower/steering actuation settings."""
    depower_tag = format_length_tag(
        config.get(
            "power_tape_final_extension", CONFIG_DEFAULTS["power_tape_final_extension"]
        )
    )
    steering_tag = format_length_tag(config.get("steering_tape_final_extension", 0.0))
    return f"depower_{depower_tag}_steer_{steering_tag}"


def resolve_initial_geometry_rotation_kwargs(config: dict) -> dict:
    """Build rotate_geometry kwargs from config, with legacy fallback."""
    angle_deg = config.get("initial_geometry_rotation_angles_deg")
    angle_rad = config.get("initial_geometry_rotation_angles_rad")
    if angle_deg is not None and angle_rad is not None:
        raise ValueError(
            "Provide only one of `initial_geometry_rotation_angles_deg` or "
            "`initial_geometry_rotation_angles_rad`."
        )
    if angle_deg is None and angle_rad is None:
        angle_deg = [
            0.0,
            float(
                config.get(
                    "initial_geometry_rotation_deg",
                    CONFIG_DEFAULTS["initial_geometry_rotation_deg"],
                )
            ),
            0.0,
        ]
    return {
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
        "point": config.get(
            "initial_geometry_rotation_point",
            CONFIG_DEFAULTS["initial_geometry_rotation_point"],
        ),
        "axes": config.get(
            "initial_geometry_rotation_axes",
            CONFIG_DEFAULTS["initial_geometry_rotation_axes"],
        ),
    }


def configure_system_model_from_config(system_model: SystemModel, config: dict) -> None:
    """Populate a SystemModel from config values using CONFIG_DEFAULTS for missing keys."""
    system_model.angle_elevation = np.deg2rad(
        float(config.get("angle_elevation_deg", CONFIG_DEFAULTS["angle_elevation_deg"]))
    )
    system_model.angle_azimuth = np.deg2rad(
        float(config.get("angle_azimuth_deg", CONFIG_DEFAULTS["angle_azimuth_deg"]))
    )
    system_model.angle_course = np.deg2rad(
        float(config.get("angle_course_deg", CONFIG_DEFAULTS["angle_course_deg"]))
    )
    system_model.speed_radial = float(
        config.get("speed_radial", CONFIG_DEFAULTS["speed_radial"])
    )
    system_model.distance_radial = float(
        config.get("distance_radial", CONFIG_DEFAULTS["distance_radial"])
    )
    system_model.wind.speed_wind_ref = float(
        config.get("wind_speed_wind_ref", CONFIG_DEFAULTS["wind_speed_wind_ref"])
    )
    system_model.timeder_speed_tangential = float(
        config.get(
            "timeder_speed_tangential", CONFIG_DEFAULTS["timeder_speed_tangential"]
        )
    )
    system_model.timeder_speed_radial = float(
        config.get("timeder_speed_radial", CONFIG_DEFAULTS["timeder_speed_radial"])
    )


def resolve_starting_struc_nodes(
    case_dir: Path,
    struc_nodes_default: np.ndarray,
) -> np.ndarray | None:
    """Load final struc_nodes from a previous simulation in case_dir, or return None."""
    h5_path = case_dir / "sim_output.h5"
    if not h5_path.exists():
        return None
    try:
        _, tracking_data = load_sim_output(h5_path)
        if "positions" not in tracking_data:
            return None
        positions = np.asarray(tracking_data["positions"])
        if positions.ndim != 3 or positions.shape[2] != 3:
            return None
        struc_nodes_loaded = np.array(positions[-1], dtype=float)
        if struc_nodes_loaded.shape != np.asarray(struc_nodes_default).shape:
            return None
        logging.info(f"Recovered struc_nodes from previous run: {h5_path.parent.name}")
        return struc_nodes_loaded
    except Exception as exc:
        logging.warning(f"Could not load struc_nodes from {h5_path}: {exc}")
        return None


def resolve_starting_rest_lengths(
    case_dir: Path,
    l0_arr_default: np.ndarray,
) -> np.ndarray:
    """Load final rest_lengths from a previous simulation in case_dir, or return defaults."""
    h5_path = case_dir / "sim_output.h5"
    if not h5_path.exists():
        return l0_arr_default
    try:
        metadata, _ = load_sim_output(h5_path)
        if "rest_lengths" in metadata:
            rest_lengths_loaded = np.asarray(metadata["rest_lengths"], dtype=float)
            if rest_lengths_loaded.shape == np.asarray(l0_arr_default).shape:
                logging.info(
                    f"Recovered rest_lengths from previous run: {h5_path.parent.name}"
                )
                return rest_lengths_loaded
        return l0_arr_default
    except Exception as exc:
        logging.warning(f"Could not load rest_lengths from {h5_path}: {exc}")
        return l0_arr_default

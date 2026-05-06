from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_VSM_SRC = PROJECT_DIR.parent / "Vortex-Step-Method" / "src"
DEFAULT_CONFIG_FOLDER = PROJECT_DIR / "data" / "LEI-V3-KITE"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "results" / "aerodynamics"


def csv_vector(value: str, *, length: int, name: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"{name} must contain {length} values.")
    return np.asarray(parts, dtype=float)


def add_vsm_path(path: str | None) -> None:
    if path:
        vsm_src = Path(path).expanduser().resolve()
        if str(vsm_src) not in sys.path:
            sys.path.insert(0, str(vsm_src))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vsm-src",
        default=str(DEFAULT_VSM_SRC) if DEFAULT_VSM_SRC.exists() else None,
        help="Optional path to VSM/src.",
    )
    parser.add_argument(
        "--config-folder",
        default=str(DEFAULT_CONFIG_FOLDER),
        help="Config folder containing system.yaml, aero_geometry.yaml, and optional struc_geometry.yaml (default: data/LEI-V3-KITE).",
    )
    parser.add_argument(
        "--no-struc-geometry",
        action="store_true",
        help="Ignore any structural geometry file and run aero-only.",
    )
    parser.add_argument("--n-panels", type=int, default=18)
    parser.add_argument("--spanwise-panel-distribution", default="uniform")
    parser.add_argument("--reference-point", default="0,0,0")
    parser.add_argument("--center-of-gravity", default="0.5,0,5")
    parser.add_argument(
        "--mass-wing",
        type=float,
        default=None,
        help="Wing mass in kg (default: from system.yaml)",
    )
    parser.add_argument("--tether-diameter", type=float, default=0.0)
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--elevation-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--course-deg", type=float, default=90.0)
    parser.add_argument("--radial-speed", type=float, default=0.0)
    parser.add_argument("--distance-radial", type=float, default=200.0)
    parser.add_argument("--x-guess", default="25,0,0,0,0")
    parser.add_argument("--bounds-lower", default="2,-15,-15,-15,-5")
    parser.add_argument("--bounds-upper", default="80,15,15,15,5")
    parser.add_argument(
        "--inertia-xx",
        type=float,
        default=None,
        help="Ixx in kg·m² (default: from system.yaml)",
    )
    parser.add_argument(
        "--inertia-yy",
        type=float,
        default=None,
        help="Iyy in kg·m² (default: from system.yaml)",
    )
    parser.add_argument(
        "--inertia-zz",
        type=float,
        default=None,
        help="Izz in kg·m² (default: from system.yaml)",
    )
    parser.add_argument("--moment-tolerance", type=float, default=1e-3)
    parser.add_argument("--include-gravity", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to results/aerodynamics/<script>.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening interactive windows.",
    )


def output_dir(args: argparse.Namespace, script_name: str) -> Path:
    path = (
        Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / script_name
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def save_figure(fig: Any, path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Wrote {path}")


def parsed_common(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reference_point": csv_vector(
            args.reference_point, length=3, name="reference-point"
        ),
        "center_of_gravity": csv_vector(
            args.center_of_gravity, length=3, name="center-of-gravity"
        ),
        "x_guess": csv_vector(args.x_guess, length=5, name="x-guess"),
        "bounds_lower": csv_vector(args.bounds_lower, length=5, name="bounds-lower"),
        "bounds_upper": csv_vector(args.bounds_upper, length=5, name="bounds-upper"),
    }


def _resolve_csv_paths(config: dict, config_folder: Path) -> dict:
    """Resolve relative CSV file paths in config to absolute paths.

    Updates wing_airfoils data in-place to use absolute paths so they work
    when the config is written to a temporary file elsewhere.
    """
    if "wing_airfoils" not in config or "data" not in config["wing_airfoils"]:
        return config

    config_folder = Path(config_folder).resolve()
    for row in config["wing_airfoils"]["data"]:
        if len(row) >= 3 and isinstance(row[2], dict) and "csv_file_path" in row[2]:
            csv_path = row[2]["csv_file_path"]
            # Only resolve if it's a relative path
            if not Path(csv_path).is_absolute():
                abs_csv_path = (config_folder / csv_path).resolve()
                row[2]["csv_file_path"] = str(abs_csv_path)

    return config


def _merge_aero_and_structural_geometry(
    aero_geometry_path: Path, struc_geometry_path: Path
) -> dict:
    """Load profiles from aero_geometry.yaml and geometry from struc_geometry.yaml.

    Returns a config dict suitable for VSM's BodyAerodynamics.instantiate with:
    - wing_airfoils from aero_geometry.yaml (profiles/polars)
    - wing_sections from aero_geometry.yaml (wing definition stays aerodynamic)

    The structural file is still used separately via `bridle_path` and for
    physical properties in `load_config_from_folder`.
    """
    import yaml

    # Load aero geometry (profiles)
    with open(aero_geometry_path, "r") as f:
        aero_config = yaml.safe_load(f)

    # Structural geometry is intentionally not merged into wing_sections here.
    # The VSM body already receives the structural file via `bridle_path`.
    merged_config = dict(aero_config)

    # Resolve CSV file paths to absolute paths
    _resolve_csv_paths(merged_config, aero_geometry_path.parent)

    return merged_config


def load_config_from_folder(
    config_folder: Path | str,
    *,
    use_struc_geometry: bool = True,
) -> dict[str, Any]:
    """Load aerodynamic and physical properties from a config folder.

    Expected files:
    - system.yaml        (or system.yml) — contains physical properties (mass, inertia, CoG)
    - aero_geometry.yaml — VSM aerodynamic geometry (profiles, wing sections)
    - struc_geometry.yaml (optional) — structural geometry (LE/TE nodes, bridles)

    Property source selection:
    - If struc_geometry.yaml EXISTS: extract from components.kite (aggregate: wing+bridle+KCU)
    - If struc_geometry.yaml NOT FOUND: extract from components.kite.structure (wing only)

    Returns a dict with:
    - 'body_config': dict ready for VSM BodyAerodynamics.instantiate
    - 'mass': mass from system.yaml (kite if struc exists, wing structure else)
    - 'inertia': inertia tensor from system.yaml (kite if struc exists, wing structure else)
    - 'center_of_mass': CoG from system.yaml (kite if struc exists, wing structure else)
    - 'aero_geometry_path': path to aero_geometry.yaml
    - 'struc_geometry_path': path to struc_geometry.yaml (or None if not found)
    """
    import yaml

    config_folder = Path(config_folder).expanduser().resolve()

    # Load system.yaml (physical properties)
    system_yml = config_folder / "system.yaml"
    if not system_yml.exists():
        system_yml = config_folder / "system.yml"
    if not system_yml.exists():
        raise FileNotFoundError(f"system.yaml not found in {config_folder}")

    with open(system_yml, "r") as f:
        system_config = yaml.safe_load(f)

    # Load aero_geometry.yaml
    aero_geometry_path = config_folder / "aero_geometry.yaml"
    if not aero_geometry_path.exists():
        raise FileNotFoundError(f"aero_geometry.yaml not found in {config_folder}")

    with open(aero_geometry_path, "r") as f:
        aero_config = yaml.safe_load(f)

    # Check for an optional structural geometry file.
    # Prefer the exact canonical name first, then any YAML containing "struc_geometry".
    struc_geometry_path = None
    if use_struc_geometry:
        candidate_paths = [
            config_folder / "struc_geometry.yaml",
            *sorted(config_folder.glob("*struc_geometry*.yaml")),
        ]
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                struc_geometry_path = candidate_path
                break

    # Select property source based on whether struc_geometry exists
    kite_node = system_config.get("components", {}).get("kite", {})

    if struc_geometry_path:
        # Using bridles/KCU: extract aggregate properties from kite root
        properties = {
            "mass": float(kite_node.get("mass", 0.0)),
            "inertia": kite_node.get(
                "inertia_tensor", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            ),
            "center_of_mass": kite_node.get("center_of_mass", [0.0, 0.0, 0.0]),
        }
    else:
        # No bridles/KCU: extract wing structure properties only
        wing_struct = kite_node.get("wing", {}).get("structure", {})
        properties = {
            "mass": float(wing_struct.get("mass", 0.0)),
            "inertia": wing_struct.get(
                "inertia_tensor", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            ),
            "center_of_mass": wing_struct.get("center_of_mass", [0.0, 0.0, 0.0]),
        }

    # Build body config: merge aero + struc if available, else use aero alone
    if struc_geometry_path:
        body_config = _merge_aero_and_structural_geometry(
            aero_geometry_path, struc_geometry_path
        )
    else:
        body_config = aero_config
        # Resolve CSV paths to absolute even when not merging
        _resolve_csv_paths(body_config, config_folder)

    return {
        "body_config": body_config,
        "mass": properties["mass"],
        "inertia": properties["inertia"],
        "center_of_mass": properties["center_of_mass"],
        "aero_geometry_path": str(aero_geometry_path),
        "struc_geometry_path": (
            str(struc_geometry_path) if struc_geometry_path else None
        ),
    }


def build_body(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Load VSM body and physical properties from config folder.

    Returns:
        (body, properties) where:
        - body: VSM BodyAerodynamics instance
        - properties: dict with mass, inertia, center_of_mass, geometry paths
    """
    add_vsm_path(args.vsm_src)
    from VSM.core.BodyAerodynamics import BodyAerodynamics
    import tempfile
    import yaml

    # Load config from folder
    config = load_config_from_folder(
        args.config_folder,
        use_struc_geometry=not getattr(args, "no_struc_geometry", False),
    )
    body_config = config["body_config"]

    # Write body config to temporary file for VSM to read
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(body_config, tmp)
        tmp_path = tmp.name

    try:
        body = BodyAerodynamics.instantiate(
            n_panels=args.n_panels,
            file_path=tmp_path,
            spanwise_panel_distribution=args.spanwise_panel_distribution,
            bridle_path=(
                config["struc_geometry_path"] if config["struc_geometry_path"] else None
            ),
        )
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink()

    # Return both body and properties for use in scripts
    properties = {
        "mass": config["mass"],
        "inertia": config["inertia"],
        "center_of_mass": config["center_of_mass"],
        "aero_geometry_path": config["aero_geometry_path"],
        "struc_geometry_path": config["struc_geometry_path"],
    }

    return body, properties


def build_system_model(args: argparse.Namespace, mass_wing: float | None = None):
    from awetrim.system.system_model import SystemModel
    from awetrim.system.tether import RigidLumpedTether

    # Use provided mass_wing or extract from args (fallback to 30.0)
    mass = (
        mass_wing
        if mass_wing is not None
        else (args.mass_wing if args.mass_wing is not None else 30.0)
    )

    system = SystemModel(tether=RigidLumpedTether(diameter=args.tether_diameter))
    system.mass_wing = mass
    system.kite.mass_wing = mass
    system.angle_elevation = np.deg2rad(args.elevation_deg)
    system.angle_azimuth = np.deg2rad(args.azimuth_deg)
    system.angle_course = np.deg2rad(args.course_deg)
    system.speed_radial = args.radial_speed
    system.distance_radial = args.distance_radial
    system.wind.speed_wind_ref = args.wind_speed
    system.wind.direction_wind = 0.0
    system.timeder_speed_tangential = 0.0
    system.timeder_speed_radial = 0.0
    system.timeder_angle_course = 0.0
    return system


def update_system_model(system: Any, case_values: dict[str, float]) -> None:
    if "wind_speed" in case_values:
        system.wind.speed_wind_ref = case_values["wind_speed"]
    if "elevation_deg" in case_values:
        system.angle_elevation = np.deg2rad(case_values["elevation_deg"])
    if "azimuth_deg" in case_values:
        system.angle_azimuth = np.deg2rad(case_values["azimuth_deg"])
    if "course_deg" in case_values:
        system.angle_course = np.deg2rad(case_values["course_deg"])
    if "radial_speed" in case_values:
        system.speed_radial = case_values["radial_speed"]
    if "distance_radial" in case_values:
        system.distance_radial = case_values["distance_radial"]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return to_jsonable(value.tolist())
        return value.tolist()
    if isinstance(value, np.generic):
        if np.iscomplexobj(value):
            return {"real": float(value.real), "imag": float(value.imag)}
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {
            key: to_jsonable(val) for key, val in value.items() if key != "optimizer"
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def print_trim_summary(result: dict[str, Any]) -> None:
    opt_x = np.asarray(result["opt_x"], dtype=float)
    cm = np.asarray(result["cm"], dtype=float)
    print("success_optimizer:", result["success"])
    print("success_physical:", result["success_physical"])
    print("speed_tangential [m/s]:", f"{opt_x[0]:.6g}")
    print("angle_roll_body_deg:", f"{opt_x[1]:.6g}")
    print("angle_pitch_body_deg:", f"{opt_x[2]:.6g}")
    print("angle_yaw_body_deg:", f"{opt_x[3]:.6g}")
    print("timeder_angle_course_body [rad/s]:", f"{opt_x[4]:.6g}")
    print("cm:", np.array2string(cm, precision=6))
    print("cfx/cfy:", f"{result['cfx']:.6g}", f"{result['cfy']:.6g}")
    print("aoa_center_deg:", f"{result['aoa_deg']:.6g}")
    print("side_slip_deg:", f"{result['side_slip_deg']:.6g}")
    print("cl/cd:", f"{float(result['cl']):.6g}", f"{float(result['cd']):.6g}")

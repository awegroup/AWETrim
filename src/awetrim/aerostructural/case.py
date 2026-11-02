"""Case setup shared by every aerostructural driver and script.

Kite paths, config defaults, the actuation case-folder name, the initial
geometry rotation and the SystemModel the coupled trim runs on. Lived in
``examples/2_aerostructural/common.py`` until 2026-10-07; that file now only
re-exports this module so existing scripts keep importing ``common``.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from awetrim.aerostructural.utils import load_sim_output
from awetrim.environment.profile_laws import air_density
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.system.system_model import SystemModel

DEFAULT_KITE_NAME = "LEI-V3-KITE"

# THE structural geometry of a kite: an awesIO structure document
# (structure_schema.yml, read with coupled.awesio_structure.load_structure)
# that Billow's full model reads and the wireframe is reduced from
# (wireframe.reduce_fem), so both fidelities start from one input file. A
# deformed snapshot in a results folder has the same name. See
# data/<kite>/README_struc_files.md.
DEFAULT_STRUC_GEOMETRY_FILENAME = "structure.yml"
SNAPSHOT_STRUC_GEOMETRY_FILENAME = DEFAULT_STRUC_GEOMETRY_FILENAME
#: The per-case wireframe reduction (wireframe.driver).
WIREFRAME_STRUC_GEOMETRY_FILENAME = "structure_wireframe.yml"

# Code defaults of MODEL keys a stored or hand-built config may lack. The
# operating point is not here: it is the Case's, and has no defaults.
CONFIG_DEFAULTS: dict = {
    "is_with_gravity": False,
    "is_with_aero_bridle": True,
    # KCU bluff-body drag in the trim (awetrim.aerodynamics.kcu_drag).
    # Added 2026-08-21; set False to reproduce a pre-2026-08-21 result.
    "is_with_kcu_drag": True,
    "initial_geometry_rotation_deg": 0.0,
    "initial_geometry_rotation_point": [0.0, 0.0, 0.0],
    "initial_geometry_rotation_axes": ["x", "y", "z"],
}

#: Output options: the script's, not as_config's. A run's ``output`` override
#: block sets them.
OUTPUT_DEFAULTS: dict = {
    "is_with_initial_structure_plot": False,
    "is_with_coupling_plot_per_iteration": False,
    "is_with_struc_plot_per_iteration": False,
    "is_with_aero_plot_per_iteration": False,
    "is_with_final_plot": False,
    "is_with_aero_frame_final_plot": False,
    "is_save_geometry_snapshots": True,
}


@dataclass(frozen=True)
class Case:
    """What an aerostructural run IS: flight state, wind, atmosphere, command.

    The script that runs a case owns it; ``as_config.yaml`` (how the problem
    is modelled and solved) carries none of it. The flight state, the wind and
    the command have no defaults -- every script states its full operating
    point -- only the atmosphere does (standard sea-level constants).

    ``wind`` is a wind spec as :func:`awetrim.system.factory.
    create_wind_model_from_config` reads it (``model`` and ``speed_wind_ref``
    required, ``height_ref`` and the law's parameters optional). The tape
    extensions are relative to the tape lengths the structure is built with
    [m]: + lengthens the depower tape; the steering one is the half-difference.
    """

    elevation_deg: float
    azimuth_deg: float
    course_deg: float
    distance_radial: float  # m
    speed_radial: float  # m/s, + reel-out
    timeder_speed_tangential: float  # m/s2
    timeder_speed_radial: float  # m/s2
    wind: Mapping[str, Any]
    power_tape_extension: float  # m
    steering_tape_extension: float  # m
    rho: float = 1.225  # kg/m3, sea level (air_density_law: exponential)
    air_density_law: str = "constant"  # constant | exponential
    height_gnd: float = 0.0  # m, ground height above sea level
    mu: float = 1.789e-05  # kg/(m s)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)  # m/s2

    def __post_init__(self):
        missing = {"model", "speed_wind_ref"} - set(self.wind)
        if missing:
            raise ValueError(f"Case.wind needs {sorted(missing)}, got {dict(self.wind)}")
        if self.air_density_law not in ("constant", "exponential"):
            raise ValueError(f"unknown air_density_law {self.air_density_law!r}")

    def to_config(self) -> dict:
        """The case as the runtime keys the coupled solvers read."""
        return {
            "angle_elevation_deg": float(self.elevation_deg),
            "angle_azimuth_deg": float(self.azimuth_deg),
            "angle_course_deg": float(self.course_deg),
            "distance_radial": float(self.distance_radial),
            "speed_radial": float(self.speed_radial),
            "timeder_speed_tangential": float(self.timeder_speed_tangential),
            "timeder_speed_radial": float(self.timeder_speed_radial),
            "wind": dict(self.wind),
            "wind_speed_wind_ref": float(self.wind["speed_wind_ref"]),
            "power_tape_final_extension": float(self.power_tape_extension),
            "steering_tape_final_extension": float(self.steering_tape_extension),
            "rho": float(self.rho),
            "air_density_law": self.air_density_law,
            "height_gnd": float(self.height_gnd),
            "mu": float(self.mu),
            "grav_constant": [float(g) for g in self.gravity],
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Case":
        """The case a runtime config (e.g. a stored result's) was run at.

        Results are data: a config stored before the Case existed has no
        ``wind`` spec and is read as the uniform wind it was flown in.
        """
        wind = config.get("wind") or {
            "model": "uniform",
            "speed_wind_ref": float(config["wind_speed_wind_ref"]),
        }
        if float(wind["speed_wind_ref"]) != float(config["wind_speed_wind_ref"]):
            raise ValueError(
                f"wind_speed_wind_ref {config['wind_speed_wind_ref']} disagrees with the "
                f"wind spec's speed_wind_ref {wind['speed_wind_ref']}: change the wind "
                "through the Case (dataclasses.replace(case, wind=...).apply(config)), "
                "not by editing one key"
            )
        return cls(
            elevation_deg=config["angle_elevation_deg"],
            azimuth_deg=config["angle_azimuth_deg"],
            course_deg=config["angle_course_deg"],
            distance_radial=config["distance_radial"],
            speed_radial=config["speed_radial"],
            timeder_speed_tangential=config.get("timeder_speed_tangential", 0.0),
            timeder_speed_radial=config.get("timeder_speed_radial", 0.0),
            wind=wind,
            power_tape_extension=config.get("power_tape_final_extension", 0.0),
            steering_tape_extension=config.get("steering_tape_final_extension", 0.0),
            rho=config.get("rho", 1.225),
            air_density_law=config.get("air_density_law", "constant"),
            height_gnd=config.get("height_gnd", 0.0),
            mu=config.get("mu", 1.789e-05),
            gravity=tuple(config.get("grav_constant", (0.0, 0.0, -9.81))),
        )

    def apply(self, config: Mapping[str, Any]) -> dict:
        """``config`` run at THIS case: its case keys replaced by this case's."""
        return {**config, **self.to_config()}

    def folder_name(self) -> str:
        """The results folder name of this case (:func:`build_flight_case_folder`)."""
        return build_flight_case_folder(self.to_config())


#: The runtime keys a Case owns. as_config and the overrides may not set them.
CASE_KEYS = frozenset(
    Case(0, 0, 0, 0, 0, 0, 0, {"model": "uniform", "speed_wind_ref": 0.0}, 0, 0).to_config()
)
#: The subsystem blocks of as_config.yaml (all required) and of an override
#: (any of them, plus ``output``).
AS_CONFIG_BLOCKS = ("aerodynamics", "structure", "tether", "coupling")
OVERRIDE_BLOCKS = AS_CONFIG_BLOCKS + ("output",)

#: Where a key of a block lands in the runtime config the solvers read:
#: ``(block, key) -> runtime key`` (``None``: the top level). A block's other
#: keys go to :data:`_BLOCK_DESTINATION` (``None``: the top level; a block
#: missing there takes only the keys named here).
_RUNTIME_NAMES = {
    ("aerodynamics", "bridle"): "aerodynamic_bridle",
    ("aerodynamics", "is_with_aero_bridle"): None,
    ("aerodynamics", "is_with_kcu_drag"): None,
    ("structure", "solver"): "structural_solver",
    ("structure", "wireframe"): "structural_pss",
    ("structure", "billow"): "structural_billow",
    ("coupling", "trim"): "quasi_steady_trim",
    ("coupling", "loop"): "aero_structural_solver",
    ("coupling", "load_transfer"): "aero2struc",
}
_BLOCK_DESTINATION = {"aerodynamics": "aerodynamic", "tether": "tether", "coupling": None, "output": None}

#: Wireframe keys of the elongation-bound stiffening and the modulus ramp,
#: removed 2026-11-02: the structure flies its file's datasheet moduli.
_REMOVED_WIREFRAME_KEYS = frozenset({
    "update_stiffness", "stiffness_update_scope", "max_elongation",
    "max_stiffness_factor", "stiffness_update_factor", "stiffness_settle_iters",
    "stiffness_trigger_factor", "modulus_ramp_target_factor",
    "modulus_ramp_step_factor", "modulus_ramp_scope", "modulus_stall_iters",
    "modulus_max_halvings", "modulus_divergence_factor",
})

#: Runtime entries the structure owns (set by :func:`apply_structure`), never
#: a config's: the KCU point is the moment reference and the fixed node.
_STRUCTURE_OWNED = (("aerodynamic", "reference_point"), ("structural_pss", "fixed_point_indices"))


def merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any] | None):
    """``base`` with ``overrides`` applied, nested dicts merged key by key."""
    merged = copy.deepcopy(dict(base))
    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def to_runtime(blocks: Mapping[str, Any], source: str) -> dict:
    """The runtime config (the keys the solvers read) of subsystem ``blocks``.

    Refuses an unknown block or key, a case key (the operating point is the
    Case's) and the KCU point (the structure's).
    """
    unknown = sorted(set(blocks) - set(OVERRIDE_BLOCKS))
    if unknown:
        raise ValueError(
            f"{source}: unknown blocks {unknown}; the blocks are {list(OVERRIDE_BLOCKS)} "
            "(the operating point is the Case's; the flat and the model/numerics "
            "formats were replaced on 2026-11-02)"
        )
    runtime: dict = {}
    for block, content in blocks.items():
        for key, value in (content or {}).items():
            if (block, key) in _RUNTIME_NAMES:
                target = _RUNTIME_NAMES[(block, key)]
                runtime = merge_config(runtime, {key if target is None else target: value})
            elif block in _BLOCK_DESTINATION:
                target = _BLOCK_DESTINATION[block]
                runtime = merge_config(runtime, {key: value} if target is None else {target: {key: value}})
            else:
                raise ValueError(f"{source}: unknown key {block}.{key}")
    found = sorted(CASE_KEYS & set(runtime))
    if found:
        raise ValueError(
            f"{source} sets {found}: the operating point (flight state, wind, "
            "atmosphere, tape command) is the Case's, not a config's"
        )
    removed = sorted(_REMOVED_WIREFRAME_KEYS & set(runtime.get("structural_pss") or {}))
    if removed:
        raise ValueError(
            f"{source} sets structure.wireframe {removed}: the elongation-bound "
            "stiffening and the modulus ramp were removed on 2026-11-02 (the "
            "structure flies its file's own moduli)"
        )
    for parent, key in _STRUCTURE_OWNED:
        if key in (runtime.get(parent) or {}):
            raise ValueError(
                f"{source} sets {key}: the KCU point is the structure's "
                "(structure.yml awetrim.kcu_point and its fixed point)"
            )
    return runtime


def flatten_as_config(document: Mapping[str, Any], source: str = "as_config") -> dict:
    """The runtime base of an ``as_config`` document (its four subsystem blocks)."""
    if set(document) != set(AS_CONFIG_BLOCKS):
        raise ValueError(
            f"{source} must have exactly the blocks {list(AS_CONFIG_BLOCKS)}, got "
            f"{sorted(document)}; the flat and the model/numerics formats were "
            "replaced on 2026-11-02"
        )
    return to_runtime(document, source)


def load_as_config(path: Path | str) -> dict:
    """The runtime base of the ``as_config.yaml`` at ``path`` (:func:`flatten_as_config`)."""
    from awetrim.aerostructural.utils import load_yaml

    return flatten_as_config(load_yaml(Path(path)), source=str(path))


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict:
    """``config`` with ``overrides`` (subsystem blocks and ``output``) merged
    over it, nested dicts key by key."""
    return merge_config(config, to_runtime(overrides or {}, "overrides"))


def apply_structure(config: Mapping[str, Any], structure) -> dict:
    """``config`` with what the structure owns: the KCU point as the moment
    reference and the structure's fixed nodes (``structure`` is a
    :class:`~awetrim.aerostructural.coupled.awesio_structure.Structure`)."""
    kcu = [float(v) for v in np.asarray(structure.positions[structure.kcu], dtype=float)]
    return merge_config(
        config,
        {
            "aerodynamic": {"reference_point": kcu},
            "structural_pss": {"fixed_point_indices": [int(i) for i in structure.fixed]},
        },
    )


def runtime_config(
    case: Case,
    as_config: Mapping[str, Any] | Path | str,
    overrides: Mapping[str, Any] | None = None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> dict:
    """THE config a coupled run reads: the output defaults, ``as_config``
    (a path or an already flattened dict), a driver's ``defaults`` and the
    caller's ``overrides`` (both in override-block form), then the case."""
    base = load_as_config(as_config) if isinstance(as_config, (Path, str)) else as_config
    config = merge_config(OUTPUT_DEFAULTS, base)
    config = apply_overrides(apply_overrides(config, defaults), overrides)
    return case.apply(config)


def air_density_at_case(config: dict) -> float:
    """Air density [kg/m3] at the case's kite position.

    The ONE value every aerostructural consumer (VSM, bridle drag, the system
    model's tether) uses for a case: ``rho`` itself, or with
    ``air_density_law: exponential`` the density at z = r sin(beta).
    """
    rho = float(config["rho"])
    law = config.get("air_density_law", "constant")
    if law == "constant":
        return rho
    if law != "exponential":
        raise ValueError(f"unknown air_density_law {law!r}")
    elevation = np.deg2rad(float(config["angle_elevation_deg"]))
    distance = float(config["distance_radial"])
    height_gnd = float(config.get("height_gnd", 0.0))
    return float(air_density(distance * np.sin(elevation), rho, height_gnd=height_gnd))


def resolve_kite_paths(
    project_dir: Path | str,
    kite_name: str = DEFAULT_KITE_NAME,
) -> tuple[Path, Path, Path | None]:
    """Resolve standard kite geometry and config paths.

    Returns:
        (config_path, aero_geometry_path, struc_geometry_path)
        where struc_geometry_path is the kite's awesIO structure document
        (DEFAULT_STRUC_GEOMETRY_FILENAME).
    """
    project_dir = Path(project_dir).resolve()

    config_path = project_dir / "data" / kite_name / "as_config.yaml"

    geometry_dir = project_dir / "data" / kite_name

    aero_geometry_path = geometry_dir / "aero_geometry.yaml"
    struc_geometry_path = geometry_dir / DEFAULT_STRUC_GEOMETRY_FILENAME

    return config_path, aero_geometry_path, struc_geometry_path


def resolve_config_paths(config: dict, project_dir: Path | str) -> dict:
    """``config`` with its relative file paths made absolute against the
    project root (``aero2struc.cp_distribution_path``), in place and returned:
    a path relative to the working directory silently changed the physics
    whenever a script ran from elsewhere."""
    aero2struc = config.get("aero2struc") or {}
    cp = aero2struc.get("cp_distribution_path")
    if cp and not Path(cp).is_absolute():
        aero2struc["cp_distribution_path"] = str(Path(project_dir) / cp)
    return config


def find_struc_geometry(folder: Path | str) -> Path | None:
    """The structural geometry in ``folder`` (awesIO ``structure.yml``), else None."""
    path = Path(folder) / DEFAULT_STRUC_GEOMETRY_FILENAME
    return path if path.exists() else None


def format_length_tag(value_m: float) -> str:
    """Format a signed length [m] as a filesystem-friendly tag, e.g. p0150mm."""
    sign = "p" if value_m >= 0 else "m"
    milli = int(round(abs(float(value_m)) * 1000.0))
    return f"{sign}{milli:04d}mm"


def build_actuation_case_folder(config: dict) -> str:
    """Build case folder name from final depower/steering actuation settings."""
    depower_tag = format_length_tag(config["power_tape_final_extension"])
    steering_tag = format_length_tag(config["steering_tape_final_extension"])
    return f"depower_{depower_tag}_steer_{steering_tag}"


def build_flight_case_folder(config: dict) -> str:
    """Case folder name from the flight state AND the actuation.

    ``el30_az0_chi90_r300_vr1.5_vw8_g0_depower_p0000mm_steer_p0000mm``: every
    input the demonstrator scripts expose is in the name, so changing one
    starts a new case instead of overwriting the last.
    """

    def value(key):
        return f"{float(config[key]):g}"

    gravity = int(bool(config.get("is_with_gravity", CONFIG_DEFAULTS["is_with_gravity"])))
    # A non-uniform wind names its law: the same reference speed is another case.
    law = (config.get("wind") or {}).get("model", "uniform")
    wind_law = "" if law == "uniform" else f"_{law}"
    return (
        f"el{value('angle_elevation_deg')}_az{value('angle_azimuth_deg')}"
        f"_chi{value('angle_course_deg')}_r{value('distance_radial')}"
        f"_vr{value('speed_radial')}_vw{value('wind_speed_wind_ref')}{wind_law}_g{gravity}_"
        + build_actuation_case_folder(config)
    )


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


def build_system_model(
    system_yaml_path: Path,
    tether,
    mass_structural,
    config: dict,
) -> SystemModel:
    """Create a SystemModel from system.yaml, override tether and masses, apply config.

    Using ``create_system_model_from_yaml`` ensures the Kite is fully populated
    from YAML (aero coefficients, etc.) rather than the bare default.

    ``mass_structural`` is the per-node structural mass array (``m_arr``) from the
    PSS/FEM geometry, with the KCU at index 0 by convention. It is authoritative
    for aerostructural simulations, so the SystemModel masses are derived from it:
    ``mass_kcu = m_arr[0]`` and ``mass_wing = sum(m_arr[1:])`` (wing + bridle). This
    keeps the QSM total (``mass_wing + mass_kcu``) equal to ``sum(m_arr)`` and avoids
    double-counting the KCU — which would happen if a full ``sum(m_arr)`` were stored
    in ``mass_wing`` while the YAML ``mass_kcu`` were left in place.
    """
    system_model = create_system_model_from_yaml(system_yaml_path)
    system_model.tether = tether
    m_arr = np.asarray(mass_structural, dtype=float)
    system_model.kite.mass_kcu = float(m_arr[0])  # KCU is structural node 0
    system_model.kite.mass_wing = float(m_arr[1:].sum())  # wing + bridle nodes
    configure_system_model_from_config(system_model, config)
    return system_model


def build_tether(config: dict, system_config: dict, diameter: float | None = None):
    """The tether the coupled trim runs with, of the class ``config["tether"]`` asks for.

    Diameter and density come from ``system_config`` (``diameter`` overrides
    the former). ``tether.model: williams`` gives a :class:`WilliamsTether`
    (``n_elements``, ``is_elastic``, ``cf``), anything else a
    :class:`RigidLumpedTether`. The class matters only when
    ``tether.include_in_trim`` is set, but then it must match: the Williams
    trim refuses any other tether on the system model.
    """
    from awetrim.system.tether import RigidLumpedTether
    from awetrim.utils.system_config import get_tether

    structure = get_tether(system_config)["structure"]
    tether_diameter = float(diameter) if diameter is not None else structure["diameter"]
    density = structure.get("density", 970.0)
    settings = config.get("tether", {}) or {}
    if str(settings.get("model", "rigid_lumped")).lower() == "williams":
        from awetrim.system.williams_tether import WilliamsTether

        return WilliamsTether(
            diameter=tether_diameter,
            density=density,
            n_elements=int(settings.get("n_elements", 10)),
            elastic=bool(settings.get("is_elastic", False)),
            cf=float(settings.get("cf", 0.01)),
        )
    return RigidLumpedTether(diameter=tether_diameter, density=density)


def configure_system_model_from_config(system_model: SystemModel, config: dict) -> None:
    """Put a SystemModel at the case a runtime config carries (:class:`Case`)."""
    from awetrim.system.factory import create_wind_model_from_config

    case = Case.from_config(config)
    system_model.angle_elevation = np.deg2rad(float(case.elevation_deg))
    system_model.angle_azimuth = np.deg2rad(float(case.azimuth_deg))
    system_model.angle_course = np.deg2rad(float(case.course_deg))
    system_model.speed_radial = float(case.speed_radial)
    system_model.distance_radial = float(case.distance_radial)
    system_model.wind = create_wind_model_from_config(dict(case.wind))
    system_model.timeder_speed_tangential = float(case.timeder_speed_tangential)
    system_model.timeder_speed_radial = float(case.timeder_speed_radial)
    # The trim's tether sees the same density as the VSM wing and bridle: the
    # position is fixed per case, so the kite-height value is a constant.
    system_model.rho = air_density_at_case(config)
    system_model.kite.air_density_model = None


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

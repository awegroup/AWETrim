"""The kite of the step-1 scripts, built as the coupled solver builds it.

Step 1 trims the kite on a FROZEN shape: no structural solve, but the same
wing, bridle, VSM settings and model choices a step-2 run starts from, so a
step-1 trim and a coupled solve differ only in the deformation. Everything
physical comes from the kite's data folder:

- ``structure.yml``: the wing (its FEM sections), the full bridle, the KCU
  point (the moment reference) and, with ``system.yaml``'s KCU mass, the mass,
  CG and inertia of the one body the trim balances;
- ``aero_geometry.yaml``: the airfoil polars;
- ``as_config.yaml``: how to model it -- the billowing and panel count of the
  wing (``aerodynamics``), the KCU drag, the tether in the trim (``tether``)
  and gravity (``coupling``);
- ``system.yaml``: the system model (wing and KCU mass, tether).

What a run IS -- flight state, wind, tape command -- is the script's
:class:`~awetrim.aerostructural.case.Case`.

One known difference from step 2: the VSM circulation loop. Step 1 trims with
the least_squares trims and VSM's base loop WITHOUT artificial viscosity
(:data:`STEP1_VSM_SETTINGS`), as it always has. as_config's loop (Anderson +
AV) leaves those trims at their seed -- their finite-difference Jacobian is
noise on the AV circulation solve -- and the trim that does converge with it
(``trim_casadi``) is still under review (2026-11-02).

Shapes (:func:`load_kite`): ``"built"`` is the structure's wing with
as_config's billowing (contraction 0 = the CAD wing,
``wireframe.driver.wing_edges``) and the full built bridle; ``"wireframe"`` /
``"billow"`` is the deformed shape the step-2 run of the SAME case left in
``results/<kite>/aerostructural/<fidelity>/<case.folder_name()>``. A deformed
snapshot stores its wing resampled onto ``aero_geometry.yaml``'s stations, so
the billowing arcs between them are approximated by straight panels.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "results" / "aerodynamics"
SHAPES = ("built", "wireframe", "billow")

#: Moment residual a step-1 trim converges to [-]. A trim inside a coupled
#: loop stops at the solvers' 1e-2: its geometry moves right after.
MOMENT_TOLERANCE = 1e-3

#: The VSM loop of the step-1 trims, over as_config's other VSM settings: the
#: base loop without artificial viscosity (see the module docstring).
STEP1_VSM_SETTINGS = {
    "gamma_loop_type": "base",
    "allowed_error": 1e-6,
    "is_with_artificial_viscosity": False,
}


@dataclass
class Kite:
    """Everything a step-1 trim needs, from one data folder and one case."""

    config: dict
    body: Any
    solver: Any
    system_model: Any
    kcu_drag: Any
    mass: float
    center_of_gravity: np.ndarray
    inertia_cg: np.ndarray
    reference_point: np.ndarray
    structure_path: Path
    aero_geometry_path: Path
    shape: str
    shape_folder: Path | None

    @property
    def include_gravity(self) -> bool:
        return bool(self.config["is_with_gravity"])

    @property
    def tether_in_trim(self) -> str | None:
        """The tether model the trim carries (as_config ``tether``), or None."""
        tether = self.config.get("tether") or {}
        if not bool(tether.get("include_in_trim", False)):
            return None
        return str(tether.get("model", "rigid_lumped")).lower()

    def properties(self) -> dict:
        """JSON-friendly provenance of the body."""
        return {
            "shape": self.shape,
            "shape_folder": None if self.shape_folder is None else str(self.shape_folder),
            "structure": str(self.structure_path),
            "aero_geometry": str(self.aero_geometry_path),
            "n_panels": len(self.body.panels),
            "mass": self.mass,
            "center_of_gravity": self.center_of_gravity,
            "inertia_cg": self.inertia_cg,
            "reference_point": self.reference_point,
            "tether_in_trim": self.tether_in_trim,
            "kcu_drag": self.kcu_drag is not None,
        }


def deformed_shape_folder(
    kite_name: str, case, shape: str, project_dir: Path = PROJECT_DIR
) -> Path:
    """Where the step-2 run of ``case`` on fidelity ``shape`` writes its result."""
    from awetrim.aerostructural.results import aerostructural_results_root

    return aerostructural_results_root(project_dir, kite_name) / shape / case.folder_name()


def kite_rigid_body(system_model, aggregate: Mapping[str, Any]) -> dict:
    """Mass, CG and CG inertia tensor of the ONE body the trim balances.

    The mass is the system model's wing + KCU (system.yaml), the CG and inertia
    the structure aggregate's (structure + the system.yaml KCU). They describe
    the same body only when the aggregate's mass is that mass, so a mismatch
    raises: system.yaml and structure.yml out of sync. Before v1.0.1 the
    stability script mixed three bodies (KCU counted twice, an arbitrary CG).
    """
    from awetrim.aerodynamics.vsm_quasi_steady import _system_model_mass_total

    mass = float(_system_model_mass_total(system_model))
    mass_aggregate = float(aggregate["mass"])
    if not np.isclose(mass, mass_aggregate, rtol=1e-3):
        raise ValueError(
            f"Trim mass {mass:.4f} kg (system model wing + KCU) differs from the "
            f"structure aggregate's {mass_aggregate:.4f} kg, whose CG and inertia "
            "would then belong to another body (system.yaml and structure.yml "
            "out of sync)."
        )
    return {
        "mass": mass,
        "center_of_gravity": np.asarray(aggregate["center_of_mass"], dtype=float),
        "inertia_cg": np.asarray(aggregate["inertia_tensor"], dtype=float).reshape(3, 3),
    }


def _with_absolute_polars(aero_geometry_path: Path, polar_root: Path) -> Path:
    """A temporary copy of ``aero_geometry_path`` whose polar CSV paths are
    absolute: a deformed snapshot lives in a results folder without them."""
    import yaml

    with open(aero_geometry_path, "r", encoding="utf-8") as handle:
        geometry = yaml.safe_load(handle)
    for row in (geometry.get("wing_airfoils") or {}).get("data", []):
        if len(row) >= 3 and isinstance(row[2], dict) and "csv_file_path" in row[2]:
            path = Path(row[2]["csv_file_path"])
            if not path.is_absolute():
                row[2]["csv_file_path"] = str((polar_root / path).resolve())
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.dump(geometry, handle, sort_keys=False)
    return Path(handle.name)


def load_kite(
    kite_name: str,
    case,
    *,
    shape: str = "built",
    deformed_from: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
    system_file: str = "system.yaml",
    project_dir: Path = PROJECT_DIR,
) -> Kite:
    """The kite ``data/<kite_name>`` describes, at ``case``, on ``shape``.

    ``overrides``: as_config blocks to replace for this run (e.g.
    ``{"coupling": {"is_with_gravity": True}}``), as for the step-2 drivers.
    ``deformed_from``: a results case folder to take the deformed shape from,
    instead of the one ``shape`` and ``case`` name.
    """
    import yaml

    from awetrim.aerodynamics.kcu_drag import KcuDragModel
    from awetrim.aerostructural.aerodynamic_vsm import initialize
    from awetrim.aerodynamics.vsm_settings import VSM_SOLVER_DEFAULTS, make_vsm_solver
    from awetrim.aerostructural.case import (
        DEFAULT_STRUC_GEOMETRY_FILENAME,
        air_density_at_case,
        apply_structure,
        configure_system_model_from_config,
        find_struc_geometry,
        runtime_config,
    )
    from awetrim.aerostructural.coupled.awesio_structure import load_structure
    from awetrim.aerostructural.utils import compute_kite_aggregate
    from awetrim.aerostructural.wireframe.driver import billowing_contraction, wing_edges
    from awetrim.system.factory import create_system_model_from_yaml

    data = Path(project_dir) / "data" / kite_name
    if deformed_from is not None:
        shape, folder = "deformed", Path(deformed_from)
    elif shape == "built":
        folder = None
    elif shape in SHAPES:
        folder = deformed_shape_folder(kite_name, case, shape, project_dir)
    else:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    if folder is None and (case.power_tape_extension or case.steering_tape_extension):
        raise ValueError(
            "the built shape has the built tapes: a tape command needs the "
            "deformed shape a step-2 run of the case solved (shape 'wireframe' "
            "or 'billow')"
        )
    if folder is not None and not (folder / "aero_geometry.yaml").is_file():
        script = {"billow": "run_simulation_BILLOW.py"}.get(shape, "run_simulation_PSM.py")
        raise FileNotFoundError(
            f"no deformed shape in {folder}: run examples/2_aerostructural/{script} "
            "with this Case first"
        )

    config = runtime_config(case, data / "as_config.yaml", overrides)
    if folder is None:
        structure_path = data / DEFAULT_STRUC_GEOMETRY_FILENAME
        aero_geometry_path = data / "aero_geometry.yaml"
    else:
        structure_path = find_struc_geometry(folder)
        aero_geometry_path = folder / "aero_geometry.yaml"
    structure = load_structure(structure_path)
    config = apply_structure(config, structure)

    per_section = int(config["aerodynamic"]["n_aero_panels_per_struc_section"])
    if folder is None:
        le, te = wing_edges(structure, billowing_contraction(config), per_section)
        n_panels = len(le) - 1
        body, _, _, polars = initialize(
            aero_geometry_path, config, n_panels, bridle_path=structure
        )
        body.update_from_points(
            le, te, aero_input_type="reuse_initial_polar_data", initial_polar_data=polars
        )
    else:
        # the full model's FEM sections set the mesh, as in the run that
        # produced the shape (a snapshot is a wireframe: no FEM sections)
        n_sections = len(load_structure(data / DEFAULT_STRUC_GEOMETRY_FILENAME).resolved()[2])
        resolved = _with_absolute_polars(aero_geometry_path, data)
        try:
            body, _, _, _ = initialize(
                resolved, config, (n_sections - 1) * per_section, bridle_path=structure
            )
        finally:
            resolved.unlink()

    aero = config["aerodynamic"]
    solver = make_vsm_solver(
        **{
            **{key: aero[key] for key in VSM_SOLVER_DEFAULTS if key in aero},
            **STEP1_VSM_SETTINGS,
            "reference_point": aero["reference_point"],
            "mu": config["mu"],
            "rho": air_density_at_case(config),
        }
    )

    with open(data / system_file, "r", encoding="utf-8") as handle:
        system_config = yaml.safe_load(handle)
    system_model = create_system_model_from_yaml(
        data / system_file, tether_config=config.get("tether")
    )
    configure_system_model_from_config(system_model, config)
    rigid = kite_rigid_body(system_model, compute_kite_aggregate(structure, system_config))

    return Kite(
        config=config,
        body=body,
        solver=solver,
        system_model=system_model,
        kcu_drag=(
            KcuDragModel.from_system_model(system_model)
            if bool(config.get("is_with_kcu_drag", True))
            else None
        ),
        mass=rigid["mass"],
        center_of_gravity=rigid["center_of_gravity"],
        inertia_cg=rigid["inertia_cg"],
        reference_point=np.asarray(config["aerodynamic"]["reference_point"], dtype=float),
        structure_path=Path(structure_path),
        aero_geometry_path=Path(aero_geometry_path),
        shape=shape,
        shape_folder=folder,
    )


def trim_config(kite: Kite) -> dict:
    """as_config with ``quasi_steady_trim.pre_loop`` applied: a step-1 trim
    sets the WHOLE attitude, like the coupled solver's first trim, while the
    plain bounds limit a coupled iteration's increment (+-4 deg on the LEI-V3,
    which pins the yaw of a 30 deg elevation state)."""
    from awetrim.aerostructural.aerodynamic_vsm import pre_loop_trim_config

    return pre_loop_trim_config(kite.config)


def initial_guess(kite: Kite) -> np.ndarray:
    """The coupled solver's cold trim seed (``feasible_initial_guess``)."""
    from awetrim.aerostructural.aerodynamic_vsm import (
        DEFAULT_INITIAL_AOA_BAND_DEG,
        feasible_initial_guess,
        trim_bounds,
        trim_settings,
    )

    config = trim_config(kite)
    lower, upper = trim_bounds(config)
    return feasible_initial_guess(
        kite.body,
        kite.system_model,
        (lower[0], upper[0]),
        trim_settings(config).get("initial_aoa_band_deg", DEFAULT_INITIAL_AOA_BAND_DEG),
    )


def trim(kite: Kite, *, x_guess=None):
    """Trim ``kite``: ``(result, solved_body)``.

    The least_squares trims, tetherless or with the tether as_config's
    ``tether`` block puts in the trim (``include_in_trim``, ``model``), on the
    kite's step-1 VSM solver (:data:`STEP1_VSM_SETTINGS`) and KCU drag. Search
    bounds and seed from as_config's ``quasi_steady_trim`` block with its
    ``pre_loop`` bounds (:func:`trim_config`); gravity from
    ``coupling.is_with_gravity``.
    """
    from awetrim.aerodynamics.vsm_quasi_steady import (
        solve_vsm_qs_trim_with_williams_tether,
        solve_vsm_quasi_steady_trim,
    )
    from awetrim.aerostructural.aerodynamic_vsm import trim_bounds

    lower, upper = trim_bounds(trim_config(kite))
    kwargs = dict(
        body_aero=kite.body,
        center_of_gravity=kite.center_of_gravity,
        reference_point=kite.reference_point,
        system_model=kite.system_model,
        x_guess=initial_guess(kite) if x_guess is None else np.asarray(x_guess, dtype=float),
        solver=kite.solver,
        bounds_lower=lower,
        bounds_upper=upper,
        include_gravity=kite.include_gravity,
        kcu_drag=kite.kcu_drag,
        moment_tolerance=MOMENT_TOLERANCE,
    )
    if kite.tether_in_trim is not None:
        return solve_vsm_qs_trim_with_williams_tether(**kwargs, tether_model=kite.tether_in_trim)
    return solve_vsm_quasi_steady_trim(**kwargs, return_timing_breakdown=True)


def output_dir(script_name: str, root: Path | str | None = None) -> Path:
    path = Path(root) if root else DEFAULT_OUTPUT_ROOT / script_name
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
    if isinstance(value, Path):
        return str(value)
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


def print_kite(kite: Kite) -> None:
    """One line on what is being trimmed."""
    print(
        f"Kite: shape {kite.shape}"
        + (f" ({kite.shape_folder})" if kite.shape_folder else "")
        + f", {len(kite.body.panels)} panels, mass {kite.mass:.3f} kg, "
        f"tether in trim: {kite.tether_in_trim}, KCU drag: {kite.kcu_drag is not None}, "
        f"gravity: {kite.include_gravity}"
    )


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

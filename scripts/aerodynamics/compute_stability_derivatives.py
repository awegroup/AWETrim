"""Solve a VSM quasi-steady trim and report STATIC stability.

Computes the finite-difference trim linearisation
(``compute_vsm_trim_stability_derivatives``) about a VSM quasi-steady trim
point — the raw ``J_full`` columns ARE the static force/moment derivatives —
and turns it into the six-channel static-stability verdict of
``awetrim.aerodynamics.cg_eom.static_slopes_summary``:

    roll / pitch / yaw   attitude stiffness   [N m / rad]   restoring iff < 0
    v_tau                speed stability      [N / (m/s)]   restoring iff < 0
    radial               tether stiffness     [N / m]       restoring iff < 0
    chi_dot              turn-rate damping    [N m / (rad/s)]  damping iff > 0

The ``chi_dot`` slope is in the trim vector's ``timeder_angle_course`` sense:
a positive turn rate is a rotation about ``-e_radial``, so a positive
``dM_radial/dchi_dot`` opposes the physical rotation (damping iff slope > 0).

History note: this script previously performed dynamic/modal analysis —
eigenvalue extraction, complex-plane plots, and per-eigenmode animations.
The project moved to static stability; the full modal version of this file
is recoverable from git history
(``git log -- scripts/aerodynamics/compute_stability_derivatives.py``).
The old ``--stability-frame body`` behaviour, which passed the identified
principal body axes INTO the linearisation, was removed as well: the
Omega_C transport construction is only frame-consistent in the course frame,
so the linearisation now always runs in ``DEFAULT_AXES``. With
``--stability-frame body`` the principal body axes (rotated to the trim
attitude) are instead passed as ``attitude_axes`` to a second
``static_slopes_summary`` call, so both course-frame and body-frame
attitude slopes are reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common import (
    add_common_arguments,
    build_body,
    build_system_model,
    output_dir,
    parsed_common,
    print_trim_summary,
    save_figure,
    write_json,
    DEFAULT_OUTPUT_ROOT,
)

from awetrim.aerodynamics.vsm_quasi_steady import (
    ALL_STATE_NAMES,
    DEFAULT_AXES,
    _compose_attitude_rotation,
    compute_vsm_trim_stability_derivatives,
    solve_vsm_qs_trim_with_williams_tether,
    solve_vsm_quasi_steady_trim,
)
from awetrim.aerodynamics.cg_eom import static_slopes_summary
from awetrim.aerodynamics.protocols import AxisDefinition
from awetrim.system.williams_tether import WilliamsTether

from awetrim.aerostructural.utils import load_sim_output
from awetrim.identification.rigid_body_axes import (
    compute_rigid_body_axes,
    load_psm_nodes_and_masses,
)

# ---------------------------------------------------------------------------
# Operating condition (edit here)
# ---------------------------------------------------------------------------
# These set the trim/stability operating point. They become the argument
# defaults, so the matching CLI flags (--elevation-deg, --azimuth-deg,
# --course-deg, --wind-speed, --radial-speed, --distance-radial) still
# override them when provided.
OPERATING_CONDITION = {
    "elevation_deg": 30.0,
    "azimuth_deg": 10.0,
    "course_deg": 90.0,
    "wind_speed": 8.0,
    "radial_speed": 1.5,
    "distance_radial": 250.0,
}

#: Verdict-table channel order and SI units.
_STATIC_CHANNELS = (
    ("roll", "N m / rad"),
    ("pitch", "N m / rad"),
    ("yaw", "N m / rad"),
    ("v_tau", "N / (m/s)"),
    ("radial", "N / m"),
    ("chi_dot", "N m / (rad/s)"),
)


def _load_rigid_body_axes_from_result(result_path: Path, struc_override: Path | None):
    """Load RigidBodyAxes from a structural result directory or struc_geometry YAML.

    Priority (same as plot_body_axes.py):
      1. struc_override (explicit --rigid-body-struc)
      2. {result_path}/struc_geometry.yaml  (deformed, saved by save_geometry_snapshot)
      3. HDF5 positions + data/{kite}/struc_geometry.yaml  (fallback)
    """
    result_path = result_path.resolve()
    case_dir = result_path if result_path.is_dir() else result_path.parent

    if struc_override is not None:
        struc_path = struc_override.resolve()
        with struc_path.open("r", encoding="utf-8") as f:
            sg = yaml.safe_load(f)
        nodes, m_arr = load_psm_nodes_and_masses(sg)
    else:
        saved = case_dir / "struc_geometry.yaml"
        if saved.exists():
            with saved.open("r", encoding="utf-8") as f:
                sg = yaml.safe_load(f)
            nodes, m_arr = load_psm_nodes_and_masses(sg)
        else:
            h5 = case_dir / "sim_output.h5"
            if not h5.exists():
                raise FileNotFoundError(
                    f"No sim_output.h5 or struc_geometry.yaml in {case_dir}"
                )
            _, tracking = load_sim_output(h5)
            nodes = np.asarray(tracking["positions"][-1], dtype=float)
            # infer struc_geometry from path layout
            parts = case_dir.parts
            try:
                ri = next(i for i, p in enumerate(parts) if p == "results")
                kite_name = (
                    parts[ri + 2]
                    if parts[ri + 1] == "aerostructural"
                    else parts[ri + 1]
                )
                project_root = Path(*parts[:ri])
                kite_data = project_root / "data" / kite_name
                fallback = kite_data / "struc_geometry.yaml"
                if not fallback.exists():
                    fallback = (
                        kite_data
                        / "deformed_results"
                        / "powered_2019"
                        / "struc_geometry.yaml"
                    )
            except (StopIteration, IndexError):
                raise FileNotFoundError(
                    "Could not infer struc_geometry path from result layout."
                )
            if not fallback.exists():
                raise FileNotFoundError(
                    f"Fallback struc_geometry not found: {fallback}"
                )
            with fallback.open("r", encoding="utf-8") as f:
                sg = yaml.safe_load(f)
            _, m_arr = load_psm_nodes_and_masses(sg)

    return compute_rigid_body_axes(nodes, m_arr)


def _axes_to_dict(axes: AxisDefinition) -> dict[str, list[float]]:
    """JSON-friendly axis definition."""
    return {
        "course": np.asarray(axes.course, dtype=float).tolist(),
        "normal": np.asarray(axes.normal, dtype=float).tolist(),
        "radial": np.asarray(axes.radial, dtype=float).tolist(),
    }


def _print_static_verdict(summary: dict, *, title: str) -> None:
    """Print the six-channel static-stability verdict table."""
    slopes = summary["slopes_SI"]
    print(f"\n=== static stability verdict — {title} ===")
    print(f"{'channel':<10} {'slope [SI]':>16}  {'unit':<16} verdict")
    for channel, unit in _STATIC_CHANNELS:
        slope = float(slopes[channel])
        if channel == "chi_dot":
            verdict = "damping" if summary["chi_dot_damping"] else "NOT damping"
        else:
            verdict = "restoring" if summary["restoring"][channel] else "DIVERGENT"
        print(f"{channel:<10} {slope:>+16.6e}  {unit:<16} {verdict}")
    parts = summary["chi_dot_parts"]
    print(
        "  chi_dot decomposition: kinematic "
        f"{parts['kinematic']:+.6e} + body-rate {parts['body_rate']:+.6e} "
        "N m / (rad/s)"
    )
    print(
        "  sign conventions: restoring iff slope < 0 (roll/pitch/yaw/v_tau/"
        "radial); chi_dot damping iff slope > 0 — the slope is in the"
    )
    print(
        "  `timeder_angle_course` sense, where a positive turn rate is a "
        "rotation about -e_radial, so a positive dM_radial/dchi_dot opposes it."
    )
    res_f = np.asarray(summary["trim_residual_force"], dtype=float)
    res_m = np.asarray(summary["trim_residual_moment"], dtype=float)
    print(
        f"  trim residuals: |F| = {np.linalg.norm(res_f):.3e} N, "
        f"|M_cg| = {np.linalg.norm(res_m):.3e} N m"
    )
    print(f"  eps (CG-form linearisation): {summary['eps']}")


def _static_summary_jsonable(summary: dict) -> dict:
    """Static summary with array fields converted for the JSON payload."""
    return {
        "slopes_SI": {k: float(v) for k, v in summary["slopes_SI"].items()},
        "restoring": dict(summary["restoring"]),
        "chi_dot_damping": bool(summary["chi_dot_damping"]),
        "chi_dot_parts": {
            k: float(v) for k, v in summary["chi_dot_parts"].items()
        },
        "trim_residual_force": np.asarray(
            summary["trim_residual_force"], dtype=float
        ).tolist(),
        "trim_residual_moment": np.asarray(
            summary["trim_residual_moment"], dtype=float
        ).tolist(),
        "eps": dict(summary["eps"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve VSM aerodynamic trim, compute the FD stability derivatives, "
            "and report the six-channel static-stability verdict."
        )
    )
    add_common_arguments(parser)
    # Apply the script-level operating condition as defaults (CLI still overrides).
    parser.set_defaults(**OPERATING_CONDITION)
    parser.add_argument(
        "--deformed-case",
        default=None,
        help=(
            "Name of a result case folder under --deformed-root "
            "(e.g. depower_p0000mm_steer_p0200mm). Uses that case's deformed "
            "aero_geometry.yaml/struc_geometry.yaml for the trim (shortcut for "
            "--deformed-from). Mass/inertia/CoG/tether still come from "
            "system.yaml in --config-folder."
        ),
    )
    parser.add_argument(
        "--deformed-root",
        default=None,
        help=(
            "Directory holding the deformed result cases "
            "(default: data/<kite>/deformed_results when present; otherwise "
            "results/<kite>/aerostructural)."
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available deformed cases under --deformed-root and exit.",
    )
    parser.add_argument("--eps-vel", type=float, default=0.1)
    parser.add_argument("--eps-angle-deg", type=float, default=0.5)
    parser.add_argument("--eps-rate", type=float, default=0.01)
    parser.add_argument(
        "--eps-position",
        type=float,
        default=0.5,
        help="Finite-difference step [m] for radial position state `z`.",
    )
    parser.add_argument(
        "--eps-course-rate",
        type=float,
        default=0.02,
        help=(
            "Finite-difference step [rad/s] for the course-rate column "
            "J_course_rate (feeds the chi_dot kinematic slope)."
        ),
    )
    parser.add_argument(
        "--stability-frame",
        choices=["course", "body"],
        default="course",
        help=(
            "Attitude-slope reporting frame. The trim linearisation ALWAYS "
            "runs in the course frame (the Omega_C transport construction is "
            "only frame-consistent there). `body` additionally reports the "
            "roll/pitch/yaw static slopes about the identified principal body "
            "axes at the trim attitude; requires --rigid-body-result."
        ),
    )
    parser.add_argument(
        "--rigid-body-result",
        type=Path,
        default=None,
        help=(
            "Path to a structural result directory (or sim_output.h5). "
            "Loads identified body axes, CG, and inertia from the PSM model, "
            "overriding --center-of-gravity and --inertia-xx/yy/zz."
        ),
    )
    parser.add_argument(
        "--rigid-body-struc",
        type=Path,
        default=None,
        help="struc_geometry.yaml to use with --rigid-body-result (auto-detected if omitted).",
    )
    args = parser.parse_args()

    # Resolve the deformed-results case selection (--deformed-case / --list-cases).
    kite_name = Path(args.config_folder).name
    data_deformed_root = Path(args.config_folder) / "deformed_results"
    deformed_root = (
        Path(args.deformed_root)
        if args.deformed_root
        else (
            data_deformed_root
            if data_deformed_root.is_dir()
            else DEFAULT_OUTPUT_ROOT.parent / kite_name / "aerostructural"
        )
    )
    if args.list_cases:
        print(f"Deformed-result cases under {deformed_root}:")
        if deformed_root.is_dir():
            cases = sorted(
                d.name
                for d in deformed_root.iterdir()
                if d.is_dir() and (d / "aero_geometry.yaml").exists()
            )
            for name in cases:
                print(f"  {name}")
            if not cases:
                print("  (none found)")
        else:
            print("  (directory does not exist)")
        return
    if args.deformed_case:
        case_dir = deformed_root / args.deformed_case
        if not case_dir.is_dir():
            parser.error(
                f"--deformed-case '{args.deformed_case}' not found under {deformed_root}. "
                "Use --list-cases to see available cases."
            )
        # build_body reads args.deformed_from to use the frozen deformed geometry.
        args.deformed_from = str(case_dir)
        print(f"Using deformed geometry from: {case_dir}")
    elif args.deformed_from is None:
        powered_case = deformed_root / "powered_2019"
        if powered_case.is_dir():
            args.deformed_from = str(powered_case)
            print(f"Using default powered deformed geometry from: {powered_case}")

    values = parsed_common(args)
    out_dir = output_dir(args, "stability_derivatives")

    stability_frame = args.stability_frame
    if stability_frame == "body" and args.rigid_body_result is None:
        parser.error("--stability-frame body requires --rigid-body-result.")

    print(f"Output directory: {out_dir.resolve()}")

    # Load body and properties from config folder
    body, props = build_body(args)

    mass_wing = (
        args.mass_wing if args.mass_wing is not None else props.get("mass", 30.0)
    )

    # Inertia: full CG tensor from system.yaml unless explicitly overridden.
    # --inertia-xx/yy/zz replace individual diagonal entries; the products of
    # inertia (off-diagonals, geometry basis) are kept and passed through.
    try:
        inertia_cg = np.asarray(
            props.get("inertia", [[100, 0, 0], [0, 20, 0], [0, 0, 100]]),
            dtype=float,
        ).reshape(3, 3)
    except (ValueError, TypeError):
        inertia_cg = np.diag([100.0, 20.0, 100.0])
    for _i, _diag_override in enumerate(
        (args.inertia_xx, args.inertia_yy, args.inertia_zz)
    ):
        if _diag_override is not None:
            inertia_cg[_i, _i] = float(_diag_override)
    inertia_xx, inertia_yy, inertia_zz = (float(v) for v in np.diag(inertia_cg))

    center_of_gravity = values["center_of_gravity"]
    trim_axes = DEFAULT_AXES
    rigid_body = None  # RigidBodyAxes result when --rigid-body-result is given
    deformed_aero_path: Path | None = None  # set below when deformed geometry is found

    # Optionally load identified rigid-body properties and deformed geometry.
    # NOTE: the linearisation ALWAYS runs in the course frame — the identified
    # principal body axes are used only for the body-frame attitude slopes of
    # the second static_slopes_summary call (--stability-frame body).
    if args.rigid_body_result is not None:
        from common import _resolve_csv_paths, add_vsm_path as _add_vsm_path
        import tempfile

        rigid_body = _load_rigid_body_axes_from_result(
            args.rigid_body_result, args.rigid_body_struc
        )
        center_of_gravity = rigid_body.cg
        inertia_xx, inertia_yy, inertia_zz = (
            float(v) for v in rigid_body.principal_moments
        )
        # Course stability axes: pass the full structural-frame tensor so the
        # trim-attitude rotation needs no principal-axis approximation — the
        # products of inertia are kept.
        inertia_cg = np.asarray(rigid_body.inertia_cg, dtype=float)

        case_dir = args.rigid_body_result.resolve()
        if case_dir.is_file():
            case_dir = case_dir.parent

        _aero_candidate = case_dir / "aero_geometry.yaml"

        if _aero_candidate.exists():
            deformed_aero_path = _aero_candidate
            _add_vsm_path(args.vsm_src)
            from VSM.core.BodyAerodynamics import BodyAerodynamics as _BA

            with deformed_aero_path.open("r", encoding="utf-8") as _f:
                aero_cfg = yaml.safe_load(_f)
            _resolve_csv_paths(aero_cfg, Path(args.config_folder))

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as _tmp:
                yaml.dump(aero_cfg, _tmp)
                _tmp_path = _tmp.name
            try:
                body = _BA.instantiate(
                    n_panels=args.n_panels,
                    file_path=_tmp_path,
                    spanwise_panel_distribution=args.spanwise_panel_distribution,
                    bridle_path=props.get("struc_geometry_path"),
                )
            finally:
                Path(_tmp_path).unlink()
            print(f"VSM body rebuilt from deformed aero_geometry: {deformed_aero_path}")
        else:
            print(
                f"Warning: no deformed aero_geometry.yaml in {case_dir}. "
                "Run with is_save_geometry_snapshots: true to save it. "
                "Falling back to data/ geometry."
            )

        print(f"\nRigid-body axes loaded from: {args.rigid_body_result}")
        print(f"  CG (structural frame):  {rigid_body.cg}")
        print(f"  Inertia [Ix, Iy, Iz]:   {rigid_body.principal_moments}")
        print(f"  x_body (roll):   {rigid_body.body_axes[0]}")
        print(f"  y_body (pitch):  {rigid_body.body_axes[1]}")
        print(f"  z_body (yaw):    {rigid_body.body_axes[2]}\n")

    print(f"Attitude-slope reporting frame: {stability_frame}")
    print("  trim axes:          course")
    print("  linearisation axes: course (always)")

    system_model = build_system_model(args, mass_wing=mass_wing)
    # Robust Williams detection: ``isinstance`` can miss it when ``awetrim`` is
    # importable via two paths (the src path injected by common.py and an
    # installed copy), giving two distinct class objects.
    _tether = getattr(system_model, "tether", None)
    use_williams = (
        isinstance(_tether, WilliamsTether)
        or type(_tether).__name__ == "WilliamsTether"
    )
    if use_williams:
        print("Tether model: WilliamsTether -> running joint trim+tether solve.")
        result, solved_body = solve_vsm_qs_trim_with_williams_tether(
            body_aero=body,
            center_of_gravity=center_of_gravity,
            reference_point=values["reference_point"],
            system_model=system_model,
            x_guess=values["x_guess"],
            bounds_lower=values["bounds_lower"],
            bounds_upper=values["bounds_upper"],
            include_gravity=args.include_gravity,
            moment_tolerance=args.moment_tolerance,
            max_nfev=args.max_nfev,
            axes=trim_axes,
        )
    else:
        result, solved_body = solve_vsm_quasi_steady_trim(
            body_aero=body,
            center_of_gravity=center_of_gravity,
            reference_point=values["reference_point"],
            system_model=system_model,
            x_guess=values["x_guess"],
            bounds_lower=values["bounds_lower"],
            bounds_upper=values["bounds_upper"],
            include_gravity=args.include_gravity,
            moment_tolerance=args.moment_tolerance,
            return_timing_breakdown=True,
            max_nfev=args.max_nfev,
            axes=trim_axes,
        )
    print_trim_summary(result)

    # The linearisation always runs in the course frame (DEFAULT_AXES): the
    # Omega_C transport construction is frame-consistent only there.
    # course_rate_state=True supplies J_course_rate for the chi_dot slope.
    stability = compute_vsm_trim_stability_derivatives(
        body_aero=solved_body,
        center_of_gravity=center_of_gravity,
        reference_point=values["reference_point"],
        x_trim=np.asarray(result["opt_x"], dtype=float),
        trim_result=result,
        system_model=system_model,
        mass=mass_wing,
        # Full CG tensor (zero-attitude geometry basis).
        inertia_cg=inertia_cg,
        axes=DEFAULT_AXES,
        distance_radial=args.distance_radial,
        eps_vel=args.eps_vel,
        eps_angle_deg=args.eps_angle_deg,
        eps_rate=args.eps_rate,
        eps_position=args.eps_position,
        eps_course_rate=args.eps_course_rate,
        course_rate_state=True,
    )

    # --- Diagnostic: course-frame transport rate used for the gyroscopic term --
    _omega_c_axes = np.asarray(stability.get("omega_c_axes", np.zeros(3)), dtype=float)
    print("\n--- course-frame transport rate (Omega_C) ---")
    print(f"  omega_c_model:   {stability.get('omega_c_model')}")
    print(
        "  omega_c_axes:    "
        f"[chi={_omega_c_axes[0]:+.4f}, n={_omega_c_axes[1]:+.4f}, "
        f"r={_omega_c_axes[2]:+.4f}] rad/s"
    )

    # --- Diagnostic: inertia tensor used for the rotational dynamics ----------
    _I_stab = np.asarray(
        stability.get("inertia_stability", np.zeros((3, 3))), dtype=float
    )
    _off_diag = _I_stab - np.diag(np.diag(_I_stab))
    print("\n--- inertia tensor in stability frame ---")
    print(f"  rotated_by_trim: {stability.get('inertia_rotated_by_trim')}")
    print(f"  diag [Ixx,Iyy,Izz]: {np.diag(_I_stab)}")
    print(f"  max |product of inertia|: {np.max(np.abs(_off_diag)):.3g} kg m^2")

    # --- Diagnostic: is the Williams radial-position dependency captured? ----
    print("\n--- radial-position (z) dependency diagnostic ---")
    print(
        "tether_radial_position_model:",
        stability.get("tether_radial_position_model"),
    )
    print(f"  use_williams (this run): {use_williams}")
    print(
        "  actual tether class:    "
        f"{type(_tether).__name__}  (module={type(_tether).__module__})"
    )
    print(f"  config_folder:          {args.config_folder}")
    print(f"  eps_position requested:  {args.eps_position:g} m")
    print(f"  eps_position used:       {stability.get('eps_position_used'):g} m")
    J_full = np.asarray(stability["J_full"], dtype=float)
    out_names = list(stability.get("output_names", []))
    z_col = J_full[:, list(ALL_STATE_NAMES).index("z")]
    print("  J[:, z]  (force/moment sensitivity to radial distance):")
    for name, val in zip(out_names, z_col):
        print(f"    d{name}/dz = {val:+.6e}")
    print(f"  ||J[:, z]|| = {np.linalg.norm(z_col):.6e}")
    print("  (near-zero ||J[:, z]|| => radial dependency NOT captured)")
    print("--- end diagnostic ---\n")

    # --- Raw FD Jacobian: these columns ARE the static derivatives -----------
    state_names_full = list(stability["state_names_full"])
    print("--- raw FD Jacobian J_full (rows = force/moment outputs, SI) ---")
    header = " " * 10 + "".join(f"{s:>14}" for s in state_names_full)
    print(header)
    for i, name in enumerate(out_names):
        row = "".join(f"{J_full[i, j]:+14.4e}" for j in range(J_full.shape[1]))
        print(f"{name:<10}{row}")
    j_cr = stability.get("J_course_rate")
    if j_cr is not None:
        j_cr = np.asarray(j_cr, dtype=float)
        print("J_course_rate  (d(F, M)/d(chi_dot_turn)):")
        for name, val in zip(out_names, j_cr):
            print(f"  d{name}/dchi_dot = {val:+.6e}")

    # --- Static-stability verdicts -------------------------------------------
    base = stability["cg_eom_eval"]()
    static_course = static_slopes_summary(stability, base=base)
    _print_static_verdict(
        static_course, title="course-frame attitude axes (default)"
    )

    static_body = None
    body_axes_world_at_trim = None
    if stability_frame == "body":
        # Principal body axes at trim: rows of the identified structural-frame
        # axes, rotated by the trim attitude rotation (same composition as the
        # trim solver: yaw @ pitch @ roll about the course-frame axes).
        opt_x = np.asarray(result["opt_x"], dtype=float)
        rotation_trim = _compose_attitude_rotation(
            roll_deg=float(opt_x[1]),
            pitch_deg=float(opt_x[2]),
            yaw_deg=float(opt_x[3]),
            axes=DEFAULT_AXES,
        )
        body_axes_world_at_trim = {
            channel: rotation_trim @ np.asarray(axis, dtype=float)
            for channel, axis in zip(
                ("roll", "pitch", "yaw"), rigid_body.body_axes
            )
        }
        print("\nPrincipal body axes at trim attitude (world components):")
        for channel, axis in body_axes_world_at_trim.items():
            print(f"  {channel:<6} {axis}")
        static_body = static_slopes_summary(
            stability, base=base, attitude_axes=body_axes_world_at_trim
        )
        _print_static_verdict(
            static_body, title="principal body attitude axes"
        )
        print(
            "  (v_tau / radial / chi_dot are attitude-axes-independent and "
            "match the course-frame table.)"
        )

    # --- JSON output ----------------------------------------------------------
    static_payload: dict = {
        "conventions": {
            "restoring_iff": "slope < 0 (roll, pitch, yaw, v_tau, radial)",
            "chi_dot_damping_iff": (
                "slope > 0 (timeder_angle_course sense: positive turn rate "
                "rotates about -e_radial)"
            ),
            "units": {channel: unit for channel, unit in _STATIC_CHANNELS},
        },
        "course_frame": _static_summary_jsonable(static_course),
    }
    if static_body is not None:
        static_payload["body_frame"] = _static_summary_jsonable(static_body)
        static_payload["body_axes_world_at_trim"] = {
            k: v.tolist() for k, v in body_axes_world_at_trim.items()
        }

    write_json(
        out_dir / "stability_results.json",
        {
            "trim_result": result,
            "static_stability": static_payload,
            "stability_derivatives": {
                "J_full": J_full.tolist(),
                "J_course_rate": (
                    np.asarray(j_cr, dtype=float).tolist()
                    if j_cr is not None
                    else None
                ),
                "state_names_full": state_names_full,
                "output_names": out_names,
                "eps": {
                    "vel": args.eps_vel,
                    "angle_deg": args.eps_angle_deg,
                    "rate": args.eps_rate,
                    "position_requested": args.eps_position,
                    "position_used": stability.get("eps_position_used"),
                    "course_rate": stability.get("eps_course_rate"),
                },
                "omega_c_model": stability.get("omega_c_model"),
                "omega_c_axes": _omega_c_axes.tolist(),
                "inertia_stability": _I_stab.tolist(),
                "inertia_rotated_by_trim": stability.get(
                    "inertia_rotated_by_trim"
                ),
                "tether_radial_position_model": stability.get(
                    "tether_radial_position_model"
                ),
                "n_unconverged_perturbation_solves": stability.get(
                    "n_unconverged_perturbation_solves"
                ),
                "perturbation_solves_converged": stability.get(
                    "perturbation_solves_converged"
                ),
                "stall_margin_min_deg_at_trim": stability.get(
                    "stall_margin_min_deg_at_trim"
                ),
                "n_stalled_panels_at_trim": stability.get(
                    "n_stalled_panels_at_trim"
                ),
            },
            "inertia": {
                "mass": mass_wing,
                "inertia_xx": inertia_xx,
                "inertia_yy": inertia_yy,
                "inertia_zz": inertia_zz,
                "inertia_cg": np.asarray(inertia_cg, dtype=float).tolist(),
            },
            "frame": {
                "trim_frame": "course",
                "linearisation_frame": "course",
                "attitude_report_frame": stability_frame,
                "trim_axes": _axes_to_dict(trim_axes),
            },
            "run_settings": {
                "output_dir": str(out_dir.resolve()),
                "deformed_from": args.deformed_from,
            },
            "properties": props,
        },
    )

    # --- FD Jacobian heatmap --------------------------------------------------
    fig_mat, ax_mat = plt.subplots(figsize=(8, 4))
    im = ax_mat.imshow(J_full, aspect="auto", cmap="coolwarm")
    ax_mat.set_title("J_full — raw FD static derivatives")
    ax_mat.set_xticks(range(len(state_names_full)))
    ax_mat.set_xticklabels(state_names_full)
    ax_mat.set_yticks(range(len(out_names)))
    ax_mat.set_yticklabels(out_names)
    ax_mat.set_xlabel("state")
    ax_mat.set_ylabel("output")
    fig_mat.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
    fig_mat.tight_layout()
    save_figure(fig_mat, out_dir / "stability_derivative_matrices.pdf")

    if args.no_show:
        plt.close(fig_mat)
    else:
        plt.show()


if __name__ == "__main__":
    main()

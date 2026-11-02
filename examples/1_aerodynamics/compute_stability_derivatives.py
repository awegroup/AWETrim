"""Solve a VSM quasi-steady trim and report STATIC stability.

Edit the Inputs block and run from the project root::

    python examples/1_aerodynamics/compute_stability_derivatives.py

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

The kite, its shape and the trim are ``common.load_kite`` / ``common.trim``'s
(see ``solve_single_state.py``). The linearisation always runs in the course
frame (``DEFAULT_AXES``): the Omega_C transport construction is only
frame-consistent there. ``STABILITY_FRAME = "body"`` additionally reports the
roll/pitch/yaw slopes about the centre-panel body axes of the kite's own
structure, rotated to the trim attitude.

History note: this script previously performed dynamic/modal analysis; the
full modal version is recoverable from git history
(``git log -- examples/1_aerodynamics/compute_stability_derivatives.py``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import (
    load_kite,
    output_dir,
    print_kite,
    print_trim_summary,
    save_figure,
    trim,
    write_json,
)

from awetrim.aerodynamics.vsm_quasi_steady import (
    ALL_STATE_NAMES,
    DEFAULT_AXES,
    _compose_attitude_rotation,
    compute_vsm_trim_stability_derivatives,
)
from awetrim.aerodynamics.cg_eom import static_slopes_summary
from awetrim.aerodynamics.protocols import AxisDefinition
from awetrim.aerostructural.case import Case
from awetrim.identification.rigid_body_axes import (
    compute_rigid_body_axes,
    load_psm_geometry,
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
KITE = "LEI-V3-KITE"  # a folder under data/

#: The step-2 PSM example's case (examples/2_aerostructural/run_simulation_PSM.py),
#: so SHAPE = "wireframe" finds that example's result.
CASE = Case(
    elevation_deg=0.0,  # beta
    azimuth_deg=0.0,  # phi
    course_deg=90.0,  # chi
    distance_radial=300.0,  # r [m]
    speed_radial=1.5,  # v_r [m/s], positive = reeling out
    timeder_speed_tangential=0.0,  # steady flight
    timeder_speed_radial=0.0,
    wind={"model": "uniform", "speed_wind_ref": 6.0},
    # [m]; the built shape is at 0: a tape command needs a deformed SHAPE
    power_tape_extension=0.0,
    steering_tape_extension=0.0,
)
SHAPE = "built"  # "built" | "wireframe" | "billow"
DEFORMED_FROM = None  # a results case folder; overrides SHAPE
# A gravity-free "trim" leaves the weight unbalanced and every static verdict
# off-equilibrium.
WITH_GRAVITY = True

#: Finite-difference steps of the linearisation.
EPS = {
    "vel": 0.1,  # [m/s]
    "angle_deg": 0.5,  # [deg]
    "rate": 0.01,  # [rad/s]
    "position": 0.5,  # radial position z [m]
    "course_rate": 0.02,  # [rad/s], the J_course_rate column (chi_dot slope)
}
STABILITY_FRAME = "course"  # "course" | "body" (adds the body-axes slopes)

OUTPUT_DIR = None  # default results/aerodynamics/stability_derivatives
SHOW = True

#: Verdict-table channel order and SI units.
_STATIC_CHANNELS = (
    ("roll", "N m / rad"),
    ("pitch", "N m / rad"),
    ("yaw", "N m / rad"),
    ("v_tau", "N / (m/s)"),
    ("radial", "N / m"),
    ("chi_dot", "N m / (rad/s)"),
)


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
    if STABILITY_FRAME not in ("course", "body"):
        raise ValueError(f"STABILITY_FRAME must be 'course' or 'body', got {STABILITY_FRAME!r}")
    out_dir = output_dir("stability_derivatives", OUTPUT_DIR)
    print(f"Output directory: {out_dir.resolve()}")

    kite = load_kite(
        KITE,
        CASE,
        shape=SHAPE,
        deformed_from=DEFORMED_FROM,
        overrides={"coupling": {"is_with_gravity": WITH_GRAVITY}},
    )
    print_kite(kite)
    system_model = kite.system_model
    mass_kite = kite.mass
    mass_wing = float(system_model.kite.mass_wing)
    inertia_cg = kite.inertia_cg
    inertia_xx, inertia_yy, inertia_zz = (float(v) for v in np.diag(inertia_cg))
    center_of_gravity = kite.center_of_gravity
    trim_axes = DEFAULT_AXES

    rigid_body = None
    if STABILITY_FRAME == "body":
        rigid_body = compute_rigid_body_axes(*load_psm_geometry(kite.structure_path))
        print(f"\nCentre-panel body axes of {kite.structure_path}:")
        print(f"  x_body (roll):   {rigid_body.body_axes[0]}")
        print(f"  y_body (pitch):  {rigid_body.body_axes[1]}")
        print(f"  z_body (yaw):    {rigid_body.body_axes[2]}\n")

    print(f"Attitude-slope reporting frame: {STABILITY_FRAME}")
    print("  trim axes:          course")
    print("  linearisation axes: course (always)")

    result, solved_body = trim(kite)
    print_trim_summary(result)

    # The linearisation always runs in the course frame (DEFAULT_AXES): the
    # Omega_C transport construction is frame-consistent only there.
    # course_rate_state=True supplies J_course_rate for the chi_dot slope.
    stability = compute_vsm_trim_stability_derivatives(
        body_aero=solved_body,
        center_of_gravity=center_of_gravity,
        reference_point=kite.reference_point,
        x_trim=np.asarray(result["opt_x"], dtype=float),
        trim_result=result,
        system_model=system_model,
        mass=mass_kite,
        # Full CG tensor (zero-attitude geometry basis).
        inertia_cg=inertia_cg,
        axes=DEFAULT_AXES,
        distance_radial=CASE.distance_radial,
        eps_vel=EPS["vel"],
        eps_angle_deg=EPS["angle_deg"],
        eps_rate=EPS["rate"],
        eps_position=EPS["position"],
        eps_course_rate=EPS["course_rate"],
        course_rate_state=True,
        # Must match the trim's gravity setting: the library defaults to
        # gravity ON, which puts the linearisation base off-equilibrium
        # (net force = the unbalanced weight) after a gravity-off trim.
        include_gravity=kite.include_gravity,
        solver=kite.solver,
        kcu_drag=kite.kcu_drag,
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
    print(f"  tether in trim (this run): {kite.tether_in_trim}")
    print(
        "  actual tether class:    "
        f"{type(system_model.tether).__name__}  (module={type(system_model.tether).__module__})"
    )
    print(f"  structure:              {kite.structure_path}")
    print(f"  eps_position requested:  {EPS['position']:g} m")
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
    if STABILITY_FRAME == "body":
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
            static_body, title="centre-panel body attitude axes"
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
                    "vel": EPS["vel"],
                    "angle_deg": EPS["angle_deg"],
                    "rate": EPS["rate"],
                    "position_requested": EPS["position"],
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
                "mass": mass_kite,
                "mass_wing": mass_wing,
                "inertia_xx": inertia_xx,
                "inertia_yy": inertia_yy,
                "inertia_zz": inertia_zz,
                "inertia_cg": np.asarray(inertia_cg, dtype=float).tolist(),
            },
            "frame": {
                "trim_frame": "course",
                "linearisation_frame": "course",
                "attitude_report_frame": STABILITY_FRAME,
                "trim_axes": _axes_to_dict(trim_axes),
            },
            "run_settings": {
                "output_dir": str(out_dir.resolve()),
                "case": CASE.to_config(),
            },
            "properties": kite.properties(),
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

    if SHOW:
        plt.show()
    else:
        plt.close(fig_mat)


if __name__ == "__main__":
    main()

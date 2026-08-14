"""Rigid-kite equations of motion about the centre of gravity (CG form).

The primary linearisation (:mod:`~awetrim.aerodynamics.vsm_quasi_steady`)
writes the equations at the tether attachment **B** and rotates attitude
perturbations about B, so the tether is untouched by an attitude column and
the restoring moment comes from gravity at the offset CG. This module is the
complementary formulation the physics intuition usually starts from: the
equations at the **CG**, attitude perturbations rotating **about the CG** —
so the attachment B swings (pitch-down moves B forward along the course
axis by ``|c| * dtheta``, ~0.10 m/deg for the LEI V3), the tether is
re-solved at the displaced attachment, and the tether supplies the restoring
moment through its arm.

Equations of motion (world/VSM components; ``c_att`` is the attitude-rotated
CG offset from B, ``r_B = r_cg - c_att``):

    m v_dot_cg = F_a + F_t + m g - m Omega_C x v_cg
    I_cg omega_dot + omega x I_cg omega = M_a,cg + (r_B - r_cg) x F_t

with

* mass matrix block-diagonal ``diag(m 1, I_cg)`` — no ``m [c]_x`` coupling,
  no parallel-axis shift, and no separate centripetal offset force: the
  B-form's ``-m omega x (omega x c)`` is contained in the transport term
  because ``v_cg = v_B + omega x c_att``;
* gravity and the transport inertial force act at the CG — zero moment;
* the tether acts at B — moment ``-c_att x F_t`` (absent in the B form);
* the aero moment transported to the CG with the same arm,
  ``M_a,cg = M_a,B - c_att x F_a``.

**Trim carryover.** A trim of the B form satisfies both rows of the coupled
B-point system at zero rates: ``sum F = 0`` and ``sum M_B = 0``. Moments
transport as ``M_cg = M_B + (r_B - r_cg) x sum F``, so ``sum F = 0`` makes
``sum M_cg = sum M_B`` — the trim carries over identically, with no new
solve. (For a steered, pinned-roll trim the residual roll moment is the
steering-line reaction; it appears in BOTH forms with the same value, which
is itself a check.) :func:`verify_cg_trim` measures this numerically.

The evaluator itself lives inside
``compute_vsm_trim_stability_derivatives`` (result key ``"cg_eom_eval"``) so
it shares the trim frame chain, the VSM warm start, and the Williams
fixed-length tether closure with the B form — a parallel reimplementation
could drift from the model it is meant to check. The tether length is
measured to the attachment point B; a rotation about the CG changes the
anchor-to-B distance only at second order (B swings tangentially), which
:func:`pitch_sweep` also reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "verify_cg_trim",
    "pitch_sweep",
    "plot_cg_pitch_forces",
    "plot_pitch_moment_sweep",
    "linearise_cg_eom",
    "attitude_moment_matrix",
    "attitude_slope_from_lin",
    "static_slopes_summary",
    "CG_STATE_NAMES",
]

#: States of the CG-form fast subsystem: the canonical set minus the lateral
#: velocity (the course frame follows the velocity direction; the normal
#: momentum equation is the algebraic chi_dot_turn closure, not a state row)
#: and minus the tangential positions (pinning the CG tangentially is the
#: model's constraint — a REAL constraint, not a coordinate choice).
CG_STATE_NAMES: tuple[str, ...] = (
    "u", "w", "z", "phi", "theta", "psi", "p", "q", "r",
)


def linearise_cg_eom(
    stability: dict[str, Any],
    eps_vel: float = 0.1,
    eps_angle_deg: float = 0.5,
    eps_rate: float = 0.01,
    eps_z: float | None = None,
) -> dict[str, Any]:
    """Linearise the CG-form equations of motion about the trim.

    Nine states (:data:`CG_STATE_NAMES`), central differences through the
    ``cg_eom_eval`` closure — so the attitude columns rotate about the CG,
    the attachment B swings, and the tether (re-solved at the displaced B)
    supplies the restoring stiffness the B-form attitude columns do not have.
    The mass matrix is block-diagonal ``diag(m 1, I_cg)``: rows are simply
    ``F/m`` and ``I_cg^-1 M_cg`` — no coupled 6x6 solve.

    The NORMAL translational component is not integrated: it is the
    chi_dot_turn closure (``delta_chi_dot_turn = -a_n_row / u0``), returned
    as ``chi_turn_gain_row_cg``. With the block-diagonal mass matrix the
    turn-rate column is trivially force-only, so ``F_Omega = 0`` holds by
    construction and no rank-1 update is needed.

    Velocity/rate convention: the state is ``v_cg``, so rate columns carry
    the induced attachment-velocity change ``delta_v_B = -delta_omega x c``
    into the apparent wind (handled inside ``cg_eom_eval``).
    """
    ev = stability["cg_eom_eval"]
    base = ev()
    e_c = np.asarray(base["axes_course"], dtype=float)
    e_n = np.asarray(base["axes_normal"], dtype=float)
    e_r = np.asarray(base["axes_radial"], dtype=float)
    rows_axes = np.vstack([e_c, e_n, e_r])
    mass = float(stability["mass"])
    inertia_cg = np.asarray(base["inertia_cg_att"], dtype=float)
    u0 = float(np.asarray(base["v_cg_world"], dtype=float) @ e_c)
    if eps_z is None:
        eps_z = float(stability.get("eps_position_used", 0.5))

    def column(**kw) -> np.ndarray:
        """Central-difference (accel_cg, omega_dot) column, axes components."""
        neg_kw = {
            k: (-v if np.isscalar(v) else -np.asarray(v)) for k, v in kw.items()
        }
        step = kw.pop("_step")
        neg_kw.pop("_step")
        out_p, out_m = ev(**kw), ev(**neg_kw)
        d_force = (out_p["force_net"] - out_m["force_net"]) / (2.0 * step)
        d_moment = (out_p["moment_cg_net"] - out_m["moment_cg_net"]) / (2.0 * step)
        acc = rows_axes @ (d_force / mass)
        omd = rows_axes @ np.linalg.solve(inertia_cg, d_moment)
        return np.concatenate([acc, omd])  # (a_c, a_n, a_r, om_c, om_n, om_r)

    eps_ang = float(eps_angle_deg)
    cols = {
        "u": column(delta_v_cg=eps_vel * e_c, _step=eps_vel),
        "v": column(delta_v_cg=eps_vel * e_n, _step=eps_vel),
        "w": column(delta_v_cg=eps_vel * e_r, _step=eps_vel),
        "z": column(radial_position_offset=eps_z, _step=eps_z),
        "phi": column(delta_roll_deg=eps_ang, _step=np.deg2rad(eps_ang)),
        "theta": column(delta_pitch_deg=eps_ang, _step=np.deg2rad(eps_ang)),
        "psi": column(delta_yaw_deg=eps_ang, _step=np.deg2rad(eps_ang)),
        "p": column(omega_perturb=eps_rate * e_c, _step=eps_rate),
        "q": column(omega_perturb=eps_rate * e_n, _step=eps_rate),
        "r": column(omega_perturb=eps_rate * e_r, _step=eps_rate),
    }
    J = np.column_stack([cols[s] for s in CG_STATE_NAMES])  # 6 x 9

    names = list(CG_STATE_NAMES)
    n = len(names)
    A = np.zeros((n, n))
    A[names.index("u"), :] = J[0, :]   # course acceleration
    A[names.index("w"), :] = J[2, :]   # radial acceleration
    A[names.index("z"), names.index("w")] = 1.0
    K = np.asarray(stability["kinematic_map"], dtype=float)
    for i, att in enumerate(("phi", "theta", "psi")):
        for j, rate in enumerate(("p", "q", "r")):
            A[names.index(att), names.index(rate)] = K[i, j]
    for i, rate in enumerate(("p", "q", "r")):
        A[names.index(rate), :] = J[3 + i, :]

    eig, vec = np.linalg.eig(A)

    # v-FREE variant: same columns plus the lateral velocity, the normal
    # momentum equation integrated as the v_dot row instead of eliminated as
    # the chi_dot_turn closure. Here delta_v is a lateral CG velocity — a
    # genuine sideslip DOF whose aero response comes through the apparent
    # wind, exactly as in the B form; the CG/B difference stays confined to
    # the attitude columns (swinging attachment).
    names10 = ["u", "v", "w", "z", "phi", "theta", "psi", "p", "q", "r"]
    J10 = np.column_stack([cols[s] for s in names10])  # 6 x 10
    n10 = len(names10)
    A10 = np.zeros((n10, n10))
    A10[names10.index("u"), :] = J10[0, :]
    A10[names10.index("v"), :] = J10[1, :]
    A10[names10.index("w"), :] = J10[2, :]
    A10[names10.index("z"), names10.index("w")] = 1.0
    for i, att in enumerate(("phi", "theta", "psi")):
        for j, rate in enumerate(("p", "q", "r")):
            A10[names10.index(att), names10.index(rate)] = K[i, j]
    for i, rate in enumerate(("p", "q", "r")):
        A10[names10.index(rate), :] = J10[3 + i, :]
    eig10, vec10 = np.linalg.eig(A10)

    return {
        "A_cg9": A,
        "eig_cg9": eig,
        "vec_cg9": vec,
        "state_names_cg9": names,
        "J_cg9": J,
        "A_cg10": A10,
        "eig_cg10": eig10,
        "vec_cg10": vec10,
        "state_names_cg10": names10,
        # Normal-row closure: the turn rate as the algebraic output.
        "chi_turn_gain_row_cg": -J[1, :] / u0,
        "u0_cg": u0,
        "trim_residual_force": np.asarray(base["force_net"], dtype=float),
        "trim_residual_moment": np.asarray(base["moment_cg_net"], dtype=float),
        "eps": {"vel": eps_vel, "angle_deg": eps_ang, "rate": eps_rate, "z": eps_z},
    }


def attitude_moment_matrix(lin: dict[str, Any], base: dict[str, Any]) -> np.ndarray:
    """K [3x3]: world moment response per rad about each frame rotation axis.

    Column j is the ``dM_world`` 3-vector per rad of rotation about
    ``(e_course, e_normal, e_radial)[j]``, reconstructed from the
    :func:`linearise_cg_eom` attitude columns (which store accelerations —
    the inertia at trim attitude undoes the ``I_cg^-1`` solve). The tangent
    of a rotation about ANY axis ``a`` follows to first order as
    ``K @ (components of a)``, which is what makes it usable as the reference
    for body-axes attitude sweeps too.
    """
    rows_axes = np.vstack(
        [base["axes_course"], base["axes_normal"], base["axes_radial"]]
    ).astype(float)
    inertia = np.asarray(base["inertia_cg_att"], dtype=float)
    names = lin["state_names_cg9"]
    J9 = np.asarray(lin["J_cg9"], dtype=float)
    cols = []
    for state_name in ("phi", "theta", "psi"):
        omd_axes = J9[3:6, names.index(state_name)]
        cols.append(inertia @ (rows_axes.T @ omd_axes))
    return np.column_stack(cols)


def attitude_slope_from_lin(
    lin: dict[str, Any],
    base: dict[str, Any],
    state_name: str,
    e_axis: np.ndarray,
) -> float:
    """dM_axis per unit of one linearisation column, in moment units.

    Works for the attitude columns (Nm/rad) and, because the rate columns
    share the same moment-row structure, for ``p``/``q``/``r`` as well
    (Nm per rad/s) — e.g. the yaw-damping reference of the chi_dot channel.
    """
    names = lin["state_names_cg9"]
    j = names.index(state_name)
    omd_axes = np.asarray(lin["J_cg9"], dtype=float)[3:6, j]
    rows_axes = np.vstack(
        [base["axes_course"], base["axes_normal"], base["axes_radial"]]
    ).astype(float)
    inertia = np.asarray(base["inertia_cg_att"], dtype=float)
    d_moment_world = inertia @ (rows_axes.T @ omd_axes)
    return float(d_moment_world @ np.asarray(e_axis, dtype=float))


def static_slopes_summary(
    stability: dict[str, Any],
    base: dict[str, Any] | None = None,
    attitude_axes: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Six static-stability tangents from the linearisation alone (no sweeps).

    The cheap path for batch studies (actuation grids, flight states, design
    scripts): :func:`linearise_cg_eom` (~20 warm VSM solves) supplies the
    attitude, v_tau, and radial tangents; the course-rate column (present
    when the stability dict was built with ``course_rate_state=True``) and
    the yaw-rate column give the chi_dot decomposition. Attitude axes
    default to the course frame — pass body axes (rows, world components)
    for the body-axes trio.

    Sign conventions: restoring iff slope < 0 for the five stiffness
    channels; chi_dot damping iff the coordinated slope > 0 — the slope is
    in the ``timeder_angle_course`` sense, where a positive turn rate is a
    rotation about ``-e_radial``, so a positive ``dM_radial/dchi_dot``
    opposes the physical rotation.
    """
    if base is None:
        base = stability["cg_eom_eval"]()
    lin = linearise_cg_eom(stability)
    K = attitude_moment_matrix(lin, base)
    rows_axes = np.vstack(
        [base["axes_course"], base["axes_normal"], base["axes_radial"]]
    ).astype(float)
    if attitude_axes is None:
        attitude_axes = {
            "roll": rows_axes[0], "pitch": rows_axes[1], "yaw": rows_axes[2],
        }
    mass = float(stability["mass"])
    names9 = lin["state_names_cg9"]
    J9 = np.asarray(lin["J_cg9"], dtype=float)
    slopes = {
        channel: float(axis @ (K @ (rows_axes @ np.asarray(axis, dtype=float))))
        for channel, axis in attitude_axes.items()
    }
    slopes["v_tau"] = mass * J9[0, names9.index("u")]
    slopes["radial"] = mass * J9[2, names9.index("z")]
    # chi_dot: body-rate leg from the yaw-rate column (perturbation -d*e_r),
    # kinematic leg from the B-form course-rate column transported to the CG
    # (zero-arm at trim makes the transport exact: M_cg = M_B - c x dF).
    body_rate = -attitude_slope_from_lin(lin, base, "r", rows_axes[2])
    j_cr = stability.get("J_course_rate")
    if j_cr is not None:
        j_cr = np.asarray(j_cr, dtype=float)
        c_axes = np.asarray(stability["cg_offset_trim_axes"], dtype=float)
        kinematic = float(j_cr[5] - np.cross(c_axes, j_cr[:3])[2])
    else:
        kinematic = 0.0
    slopes["chi_dot"] = kinematic + body_rate
    return {
        "slopes_SI": slopes,
        "restoring": {
            ch: bool(slopes[ch] < 0.0)
            for ch in ("roll", "pitch", "yaw", "v_tau", "radial")
        },
        "chi_dot_damping": bool(slopes["chi_dot"] > 0.0),
        "chi_dot_parts": {"kinematic": kinematic, "body_rate": body_rate},
        "trim_residual_force": np.asarray(
            lin["trim_residual_force"], dtype=float
        ),
        "trim_residual_moment": np.asarray(
            lin["trim_residual_moment"], dtype=float
        ),
        "eps": lin["eps"],
    }


def verify_cg_trim(stability: dict[str, Any]) -> dict[str, Any]:
    """Check that the B-form trim is an equilibrium of the CG form.

    Evaluates the CG form at zero attitude perturbation and compares against
    the B form's own trim right-hand side (``nonlinear_rhs_full`` at zero):
    the force rows must agree term-for-term, and the moment rows must satisfy
    the transport identity ``M_cg = M_B - c x sum F`` exactly. Returns the
    residuals; prints nothing.
    """
    cg = stability["cg_eom_eval"]()
    rhs_b = stability["nonlinear_rhs_full"](np.zeros(10))
    f_b = np.asarray(rhs_b["rhs_world"][:3], dtype=float)
    m_b = np.asarray(rhs_b["rhs_world"][3:], dtype=float)
    c = np.asarray(cg["c_att_world"], dtype=float)

    f_cg = np.asarray(cg["force_net"], dtype=float)
    m_cg = np.asarray(cg["moment_cg_net"], dtype=float)
    # Transport identity, evaluated with the CG form's own net force.
    m_cg_expected = m_b - np.cross(c, f_cg)

    f_scale = max(float(np.linalg.norm(cg["F_tether"])), 1.0)
    m_scale = max(float(np.linalg.norm(cg["M_aero_cg"])), 1.0)
    return {
        "force_net_cg": f_cg,
        "force_net_b": f_b,
        "force_match": float(np.linalg.norm(f_cg - f_b)),
        "force_residual_rel": float(np.linalg.norm(f_cg)) / f_scale,
        "moment_net_cg": m_cg,
        "moment_transport_identity_residual": float(
            np.linalg.norm(m_cg - m_cg_expected)
        ),
        "moment_residual_rel": float(np.linalg.norm(m_cg)) / m_scale,
        "moment_net_b": m_b,
        "force_scale": f_scale,
        "moment_scale": m_scale,
    }


#: Attitude channels of the sweep/plots. ``moment_axis`` is the rotation axis
#: the moment component is read along (roll about e_course, pitch about
#: e_normal — the composition convention of the linearisation); ``plane`` is
#: the projection plane of the geometry figure (horizontal, vertical), and
#: ``swing_axis`` the direction B swings along at first order.
_CHANNELS: dict[str, dict[str, Any]] = {
    "pitch": {
        "kwarg": "delta_pitch_deg",
        "moment_axis": "axes_normal",
        "plane": ("axes_course", "axes_radial"),
        "swing_axis": "axes_course",
        "view": "side",
        "plane_label": "course axis [m] (flight direction is $-e_c$)",
        "ylabel": "radial axis [m] (outward)",
    },
    "roll": {
        "kwarg": "delta_roll_deg",
        "moment_axis": "axes_course",
        "plane": ("axes_normal", "axes_radial"),
        "swing_axis": "axes_normal",
        "view": "front",
        "plane_label": "normal axis [m] (turn-centre side depends on trim)",
        "ylabel": "radial axis [m] (outward)",
    },
    # Yaw rotates about e_radial — and the CG offset is almost purely radial,
    # so B barely swings (|delta_B| ~ |c_course| * dpsi, centimetres/deg): the
    # tether contribution should come out near zero and the yaw stiffness is
    # essentially pure weathercock aero. The sweep MEASURES that instead of
    # assuming it.
    "yaw": {
        "kwarg": "delta_yaw_deg",
        "moment_axis": "axes_radial",
        "plane": ("axes_course", "axes_normal"),
        "swing_axis": "axes_normal",
        "view": "top",
        "plane_label": "course axis [m] (flight direction is $-e_c$)",
        "ylabel": "normal axis [m]",
    },
}


def pitch_sweep(
    stability: dict[str, Any],
    angles_deg: np.ndarray,
    channel: str = "pitch",
) -> dict[str, np.ndarray]:
    """CG-form attitude sweep: per-contributor moment about the CG.

    ``channel`` selects the rotation ("pitch" or "roll"); the moment
    component is read along that rotation's axis. One VSM solve per angle.
    Returns each moment contributor, the net, the anchor-to-B tether length,
    and B's in-plane swing — the ingredients of the believe-it plots.

    Note for steered, pinned-roll trims: the net ROLL moment at zero
    perturbation is the steering-line reaction, not zero — the trim pins roll
    and reports the constraint moment. The pitch and yaw components vanish.
    """
    spec = _CHANNELS[channel]
    rows: dict[str, list[float]] = {
        k: []
        for k in (
            "M_aero",
            "M_tether",
            "M_gyro",
            "M_net",
            "tether_length_to_B",
            "delta_B_inplane",
            "F_tether_mag",
            "gamma_converged",
        )
    }
    evals = []
    for a in np.asarray(angles_deg, dtype=float):
        out = stability["cg_eom_eval"](**{spec["kwarg"]: float(a)})
        e_m = np.asarray(out[spec["moment_axis"]], dtype=float)
        e_s = np.asarray(out[spec["swing_axis"]], dtype=float)
        rows["M_aero"].append(float(out["M_aero_cg"] @ e_m))
        rows["M_tether"].append(float(out["M_tether_cg"] @ e_m))
        rows["M_gyro"].append(float(out["M_gyro_cg"] @ e_m))
        rows["M_net"].append(float(out["moment_cg_net"] @ e_m))
        rows["tether_length_to_B"].append(float(out["tether_length_to_B"]))
        rows["delta_B_inplane"].append(float(out["delta_B_world"] @ e_s))
        rows["F_tether_mag"].append(float(np.linalg.norm(out["F_tether"])))
        rows["gamma_converged"].append(float(out["gamma_converged"]))
        evals.append(out)
    result = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    result["angles_deg"] = np.asarray(angles_deg, dtype=float)
    result["channel"] = channel
    result["tether_model"] = [str(e.get("tether_model_used", "?")) for e in evals]
    result["evals"] = evals  # full breakdowns, for the geometry figure
    return result


def _project_side(points: np.ndarray, origin: np.ndarray, e_c, e_r) -> np.ndarray:
    """Project world points onto the (course, radial) side-view plane."""
    rel = np.atleast_2d(np.asarray(points, dtype=float)) - origin
    return np.column_stack([rel @ e_c, rel @ e_r])


def plot_cg_pitch_forces(
    baseline: dict[str, Any],
    pitched: dict[str, Any],
    pitch_deg: float,
    path,
    force_scale_m_per_kN: float = 0.15,
    channel: str = "pitch",
) -> None:
    """Geometry view of the CG-form force system for one attitude channel.

    Pitch: side view (course-radial plane). Roll: front view (normal-radial
    plane). Baseline trim in grey, the rotated kite in colour: outline
    rotated about the (fixed) CG, the attachment B swinging with it, the
    tether drawn toward the anchor with the anchor-to-B length annotated, and
    the force vectors at their application points — tether at B, gravity and
    transport at the CG, aero at the CG with its couple reported separately.
    """
    import matplotlib.pyplot as plt

    spec = _CHANNELS[channel]
    origin = np.asarray(baseline["r_cg_world"], dtype=float)
    e_c = np.asarray(baseline[spec["plane"][0]], dtype=float)
    e_r = np.asarray(baseline[spec["plane"][1]], dtype=float)
    e_n = np.asarray(baseline[spec["moment_axis"]], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    scale = force_scale_m_per_kN / 1e3  # m per N of arrow length

    for out, color, tag in ((baseline, "0.55", "trim"),
                            (pitched, "#D55E00", f"{channel} {pitch_deg:+.1f} deg")):
        le = _project_side(out["kite_LE_world"], origin, e_c, e_r)
        te = _project_side(out["kite_TE_world"], origin, e_c, e_r)
        ax.plot(le[:, 0], le[:, 1], color=color, lw=1.8)
        ax.plot(te[:, 0], te[:, 1], color=color, lw=1.0, ls="--")
        for i in range(0, len(le), max(1, len(le) // 9)):
            ax.plot([le[i, 0], te[i, 0]], [le[i, 1], te[i, 1]], color=color, lw=0.6)
        b = _project_side(out["r_B_world"], origin, e_c, e_r)[0]
        cg = _project_side(out["r_cg_world"], origin, e_c, e_r)[0]
        ax.plot(*b, "s", color=color, ms=7)
        ax.plot(*cg, "o", color="k", ms=8, mfc="none" if tag != "trim" else "k")
        # Tether: direction from B toward the anchor, drawn a few metres.
        if out["r_anchor_world"] is not None:
            d = np.asarray(out["r_anchor_world"], dtype=float) - np.asarray(
                out["r_B_world"], dtype=float
            )
            d2 = np.array([d @ e_c, d @ e_r])
            d2 /= np.linalg.norm(d2)
            seg = np.vstack([b, b + 3.5 * d2])
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2.2, alpha=0.8)
            ax.annotate(
                f"{tag}: L(anchor$\\to$B) = {out['tether_length_to_B']:.4f} m",
                xy=b + 3.5 * d2, fontsize=9, color=color,
                xytext=(8, -14 if tag == "trim" else -30),
                textcoords="offset points",
            )
        # Forces (world -> side plane), at their application points.
        for key, at, fcolor, label in (
            ("F_tether", b, "#0072B2", "tether (at B)"),
            ("F_aero", cg, "#009E73", "aero (at CG)"),
            ("F_gravity", cg, "#666666", "gravity"),
            ("F_transport", cg, "#CC79A7", "transport"),
        ):
            f = np.asarray(out[key], dtype=float)
            f2 = np.array([f @ e_c, f @ e_r]) * scale
            if np.linalg.norm(f2) < 1e-6:
                continue
            # annotate() does not expand the data limits — register the tip.
            ax.plot(*(at + f2), ".", ms=0.1, alpha=0.0)
            ax.annotate(
                "", xy=at + f2, xytext=at,
                arrowprops=dict(arrowstyle="-|>", color=fcolor,
                                lw=2.2 if tag != "trim" else 1.1,
                                alpha=1.0 if tag != "trim" else 0.45),
            )
            if tag != "trim":
                ax.annotate(label, xy=at + f2, fontsize=9, color=fcolor,
                            xytext=(4, 4), textcoords="offset points")

    m_net_b = float(np.asarray(baseline["moment_cg_net"]) @ e_n)
    m_net_p = float(np.asarray(pitched["moment_cg_net"]) @ e_n)
    m_t_p = float(np.asarray(pitched["M_tether_cg"]) @ e_n)
    ax.set_title(
        f"CG-form force system, {spec['view']} view; "
        f"arrows: {force_scale_m_per_kN:.2f} m per kN\n"
        f"{channel} moment about CG: trim {m_net_b:+.1f} Nm, "
        f"perturbed {m_net_p:+.1f} Nm (tether contribution {m_t_p:+.1f} Nm)"
    )
    ax.set_xlabel(spec["plane_label"])
    ax.set_ylabel(spec["ylabel"])
    ax.axis("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pitch_moment_sweep(sweep: dict[str, np.ndarray], path) -> None:
    """Per-contributor moment about the CG vs the swept attitude channel."""
    import matplotlib.pyplot as plt

    channel = str(sweep.get("channel", "pitch"))
    a = sweep["angles_deg"]
    i0 = int(np.argmin(np.abs(a)))
    a_rad = np.deg2rad(a)
    fig, (ax, axnl, axl) = plt.subplots(
        3, 1, figsize=(8.0, 10.5), height_ratios=[2.2, 1.2, 0.9], sharex=True
    )
    for key, color, label in (
        ("M_aero", "#009E73", "aero"),
        ("M_tether", "#0072B2", "tether (arm $-c \\times F_t$)"),
        ("M_gyro", "#CC79A7", "gyroscopic"),
        ("M_net", "k", "net"),
    ):
        ax.plot(a, sweep[key] / 1e3, color=color, lw=2.0 if key == "M_net" else 1.5,
                label=label)
    # Tangent lines at trim: the linearisation IS these lines, so the gap to
    # the curves is exactly what the modal analysis cannot see.
    for key, color in (("M_tether", "#0072B2"), ("M_aero", "#009E73")):
        slope0 = float(np.gradient(sweep[key], a_rad)[i0])
        ax.plot(
            a,
            (sweep[key][i0] + slope0 * (a_rad - a_rad[i0])) / 1e3,
            color=color, lw=0.9, ls="--", alpha=0.7,
        )
    # Points where the Williams solve fell back to a coarser tether model.
    models = sweep.get("tether_model")
    if models is not None:
        bad = np.array(["williams" not in m for m in models], dtype=bool)
        if np.any(bad):
            ax.scatter(a[bad], sweep["M_tether"][bad] / 1e3, s=36, marker="x",
                       color="#0072B2", label="tether-model fallback", zorder=5)
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.axvline(0.0, color="0.6", lw=0.7)
    m_trim = float(sweep["M_net"][i0])
    extra = ""
    if abs(m_trim) > 10.0:
        # Pinned-roll steered trim: the nonzero trim value is the
        # steering-line reaction, not an equilibrium violation.
        ax.axhline(m_trim / 1e3, color="0.4", lw=0.8, ls=":")
        extra = (
            f"\ntrim offset {m_trim / 1e3:+.2f} kNm = steering-line reaction "
            "(pinned-roll trim); read the SLOPE"
        )
    ax.set_ylabel(f"{channel} moment about CG [kNm]")
    ax.legend(frameon=False)
    slope = np.gradient(sweep["M_net"], a_rad)
    ax.set_title(
        f"CG-form {channel} moment breakdown "
        "(gravity and transport act at the CG: zero)\n"
        f"net stiffness at trim dM/dangle = {slope[i0] / 1e3:+.1f} kNm/rad "
        f"({'restoring' if slope[i0] < 0 else 'DIVERGENT'})" + extra
    )
    # Nonlinearity panel: departure of each contributor from its tangent at
    # trim, in kNm. Zero everywhere would mean the linearisation is exact.
    for key, color, label in (
        ("M_tether", "#0072B2", "tether"),
        ("M_aero", "#009E73", "aero"),
        ("M_net", "k", "net"),
    ):
        slope0 = float(np.gradient(sweep[key], a_rad)[i0])
        tangent = sweep[key][i0] + slope0 * (a_rad - a_rad[i0])
        axnl.plot(a, (sweep[key] - tangent) / 1e3, color=color,
                  lw=2.0 if key == "M_net" else 1.5, label=label)
    axnl.axhline(0.0, color="0.6", lw=0.7)
    axnl.axvline(0.0, color="0.6", lw=0.7)
    axnl.set_ylabel("departure from tangent [kNm]")
    axnl.legend(frameon=False, fontsize=9)
    axnl.grid(alpha=0.25)

    axl.plot(a, sweep["tether_length_to_B"] - sweep["tether_length_to_B"][i0],
             color="#0072B2", lw=1.5)
    axl.set_ylabel("$\\Delta$L(anchor$\\to$B) [m]")
    axl.set_xlabel(f"{channel} perturbation about the CG [deg]")
    axl.grid(alpha=0.25)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

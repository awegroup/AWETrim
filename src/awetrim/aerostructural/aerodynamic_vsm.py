# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are adapted from ASKITE
# (https://github.com/awegroup/ASKITE), licensed under the MIT License,
# Copyright (c) 2024 jellepoland (Jelle Poland, Patrick Roeleveld, TU Delft).
# See the NOTICE file at the repository root for the full MIT licence text.

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.WingGeometry import Wing
from VSM.core.Solver import Solver
from VSM.plot_geometry_matplotlib import plot_geometry
from awetrim.aerodynamics.vsm_quasi_steady import (
    solve_quasi_steady_state,
    DEFAULT_TRANSFORMATION_C_FROM_VSM,
)

# Bounds and defaults (aoa, sideslip, course_rate_body)
kite_speed_bounds = (1.0, 50.0)  # m/s
pitch_bounds = (-10, 10)  # deg
yaw_bounds = (-10, 10)  # deg
course_rate_bounds = (
    -3,
    3,
)  # rad/s, small course rate allowed for numerical reasons; not a physical course rate
roll_bounds = (
    -10,
    10,
)  # deg, small roll allowed for numerical reasons; not a physical roll

DEFAULT_GUESS_QS = np.array(
    [30.0, 0.0, 0.0, 0.0, -0.0]
)  # [kite_speed, roll, pitch, yaw, course_rate_body]


def _alpha_stall_from_polar(polar_data: np.ndarray) -> float:
    """Return the stall angle of attack [rad] from a 2-D polar table.

    Stall is defined as the **first local Cl maximum** in the positive-Cl
    region of the polar (the onset of stall, not any post-stall Cl recovery).
    The polar table must have columns [alpha_rad, Cl, Cd, Cm].

    Returns ``math.inf`` if no peak is found within the table range, meaning
    stall is not detectable from this polar and the panel is never flagged.
    """
    import math
    polar = np.asarray(polar_data)
    cl = polar[:, 1]
    alpha = polar[:, 0]

    # Only search in the positive-Cl region to ignore noise at negative alpha.
    pos = np.where(cl > 0)[0]
    if len(pos) == 0:
        return math.inf

    # Find the first local maximum within the positive-Cl rows.
    for k in pos[1:-1]:
        if cl[k] > cl[k - 1] and cl[k] > cl[k + 1]:
            return float(alpha[k])

    # No interior peak found — polar does not cover stall.
    return math.inf


def check_panel_stall(
    alpha_at_ac: np.ndarray,
    panel_polar_data: list,
) -> np.ndarray:
    """Return a boolean mask flagging panels whose local AoA exceeds stall.

    Args:
        alpha_at_ac: Local angle of attack per panel [rad], shape (n_panels,).
        panel_polar_data: List of polar arrays (one per panel), each shaped
            (N, 4) with columns [alpha_rad, Cl, Cd, Cm].

    Returns:
        stall_mask: Boolean array of shape (n_panels,); True where panel stalls.
    """
    alpha = np.ravel(alpha_at_ac)
    n = len(alpha)
    stall_mask = np.zeros(n, dtype=bool)
    for i, polar in enumerate(panel_polar_data):
        if i >= n:
            break
        alpha_stall = _alpha_stall_from_polar(polar)
        stall_mask[i] = alpha[i] > alpha_stall
    return stall_mask


def _run_vsm_direct_fallback(body_aero, solver, system_model, current_guess):
    """
    Fallback used when quasi-steady trim fails.

    This path bypasses trim optimization and runs a direct aerodynamic solve on the
    current body geometry. The returned dictionary mirrors the keys expected by the
    coupled solver pipeline.
    """
    res = solver.solve(body_aero)

    # Convert AWETrim vectors into the same course-frame convention used by QSM outputs.
    trans = np.asarray(DEFAULT_TRANSFORMATION_C_FROM_VSM, dtype=float)
    # Total kite mass (wing + KCU); mass_wing/mass_kcu live on system_model.kite.
    mass_total = float(system_model.kite.mass_wing) + float(
        getattr(system_model.kite, "mass_kcu", 0.0)
    )
    inertial_force = -mass_total * np.asarray(
        trans @ np.asarray(system_model.acceleration_course_body, dtype=float),
        dtype=float,
    ).reshape(3)
    gravity_force = np.asarray(
        trans @ np.asarray(system_model.force_gravity, dtype=float),
        dtype=float,
    ).reshape(3)

    guess = np.asarray(current_guess, dtype=float).reshape(5)
    fallback_opt_x = np.array(DEFAULT_GUESS_QS, dtype=float)
    print(
        f"Falling back to direct VSM solve with guess {guess} (quasi-steady optimization failed)."
    )
    print(
        f"Direct VSM results: {res.get('F_distribution')}, cmx: {res.get('cmx')}, cmy: {res.get('cmy')}, cmz: {res.get('cmz')}, side_slip_deg: {res.get('side_slip_deg')}, side_slip_course_deg: {res.get('side_slip_course_deg')}"
    )
    return {
        "opt_x": fallback_opt_x,
        "success": False,
        "inertial_force": inertial_force,
        "gravity_force": gravity_force,
        "panel_cp_locations": res.get("panel_cp_locations"),
        "F_distribution": res.get("F_distribution"),
        "alpha_at_ac": res.get("alpha_at_ac"),
        "stall_mask": None,
    }


def initialize(
    aero_geometry_path,
    config,
    n_panels_aero: int,
    bridle_path=None,
) -> BodyAerodynamics:
    """
    Initialize aerodynamic model and VSM solver.

    Args:
        aero_geometry_path: Path to aerodynamic geometry file.
        config (dict): Main ASKITE configuration dictionary.
        n_panels_aero (int): Number of aerodynamic panels.
        bridle_path: Optional structural geometry path used by VSM to build bridle lines.

    Returns:
        tuple: (body_aero, vsm_solver, vel_app, initial_polar_data)
    """
    body_aero = BodyAerodynamics.instantiate(
        n_panels=int(n_panels_aero),
        file_path=aero_geometry_path,
        spanwise_panel_distribution=config["aerodynamic"][
            "spanwise_panel_distribution"
        ],
        bridle_path=bridle_path,
    )

    aero_cfg = config["aerodynamic"]
    vsm_solver = Solver(
        max_iterations=aero_cfg["max_iterations"],
        allowed_error=aero_cfg["allowed_error"],
        relaxation_factor=aero_cfg["relaxation_factor"],
        reference_point=aero_cfg["reference_point"],
        mu=config["mu"],
        rho=config["rho"],
        # Optional post-stall stabilization: the parameter-free Li/Gaunaa
        # spanwise artificial viscosity (TORQUE 2026), applied in the base
        # gamma loop. Off by default to preserve historical behaviour.
        is_with_artificial_viscosity=aero_cfg.get(
            "is_with_artificial_viscosity", False
        ),
        artificial_viscosity_factor=aero_cfg.get(
            "artificial_viscosity_factor", 0.035
        ),
    )

    # For QSM, wind speed comes from system model configuration (wind_speed_wind_ref).
    # Kite velocity is computed by the optimizer, so we initialize with wind direction.
    wind_speed_ref = float(config.get("wind_speed_wind_ref", 6.0))
    vel_app = np.array([wind_speed_ref, 0.0, 0.0])
    body_aero.va = vel_app
    wing = body_aero.wings[0]
    new_sections = wing.refine_aerodynamic_mesh()
    initial_polar_data = []
    for new_section in new_sections:
        initial_polar_data.append(new_section.polar_data)

    return body_aero, vsm_solver, vel_app, initial_polar_data


def plot_vsm_geometry(body_aero):
    """
    Plot the VSM geometry using the provided aerodynamic body.

    Args:
        body_aero (BodyAerodynamics): Aerodynamic body object.

    Returns:
        None. Displays a 3D plot.
    """
    plot_geometry(
        body_aero,
        title="VSM Geometry",
        data_type=None,
        save_path=None,
        is_save=False,
        is_show=True,
        view_elevation=15,
        view_azimuth=-120,
    )


def run_vsm_package(
    body_aero,
    solver,
    system_model,
    center_of_gravity,
    le_arr,
    te_arr,
    # va_vector,
    aero_input_type="reuse_initial_polar_data",
    initial_polar_data=None,
    reference_point=[0.0, 0.0, 0.0],
    include_gravity=False,
    is_with_plot=False,
    current_guess=None,
):
    """
    Run quasi-steady aerodynamic solve for the current structural geometry.

    Args:
        body_aero (BodyAerodynamics): Aerodynamic body object.
        solver (Solver): VSM solver object.
        system_model: AWETrim system model used by quasi-steady trim.
        center_of_gravity (np.ndarray): Current center of gravity in solver frame.
        le_arr (np.ndarray): Leading edge points (n,3).
        te_arr (np.ndarray): Trailing edge points (n,3).
        aero_input_type (str): Type of aerodynamic input.
        initial_polar_data (list or None): Initial polar data for panels.
        reference_point (list[float]): Reference point for moments and rotations.
        include_gravity (bool): Include gravity in quasi-steady force/moment balance.
        is_with_plot (bool): If True, plot the geometry.
        current_guess (np.ndarray or None): Initial guess for quasi-steady optimizer.

    Returns:
        tuple: (F_distribution, body_aero, results)
    """
    # Update aerodynamic mesh from the latest structural leading/trailing-edge points.
    body_aero.update_from_points(
        le_arr,
        te_arr,
        aero_input_type=aero_input_type,
        initial_polar_data=initial_polar_data,
    )
    # set again where velocity vector is coming from
    # The VSM va setter accepts keyword arguments but properties don't support that in Python
    # So we call the underlying setter method directly using the descriptor protocol
    # type(body_aero).va.fset(body_aero, va_vector)

    bounds_lower = np.array(
        [
            kite_speed_bounds[0],
            roll_bounds[0],
            pitch_bounds[0],
            yaw_bounds[0],
            course_rate_bounds[0],
        ]
    )
    bounds_upper = np.array(
        [
            kite_speed_bounds[1],
            roll_bounds[1],
            pitch_bounds[1],
            yaw_bounds[1],
            course_rate_bounds[1],
        ]
    )
    if current_guess is None:
        current_guess = DEFAULT_GUESS_QS

    # Primary path: quasi-steady trim solve.
    try:
        # Preserve the pre-trim body state in case we need direct-solve fallback.
        body_fallback = copy.deepcopy(body_aero)
        results, body_aero = solve_quasi_steady_state(
            body_aero=body_aero,
            center_of_gravity=center_of_gravity,
            reference_point=reference_point,
            system_model=system_model,
            x_guess=current_guess,
            solver=solver,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            include_gravity=include_gravity,
        )
        if not results.get("success", False):
            print(
                "Quasi-steady optimization did not converge to a valid trim state. "
                "Falling back to direct VSM solver.solve(body_aero)."
            )
            results = _run_vsm_direct_fallback(
                body_aero=body_fallback,
                solver=solver,
                system_model=system_model,
                current_guess=current_guess,
            )
    except ValueError as exc:
        # Typical case: non-finite residual in initial optimizer point.
        print(
            f"QSM failed ({type(exc).__name__}: {exc}). "
            "Falling back to direct VSM solver.solve(body_aero)."
        )
        results = _run_vsm_direct_fallback(
            body_aero=body_aero,
            solver=solver,
            system_model=system_model,
            current_guess=current_guess,
        )
    if is_with_plot:
        plot_vsm_geometry(body_aero)

    # Stall detection: compare local AoA against each panel's polar Cl-peak angle.
    alpha_at_ac = results.get("alpha_at_ac")
    if alpha_at_ac is not None and initial_polar_data:
        stall_mask = check_panel_stall(alpha_at_ac, initial_polar_data)
        results["stall_mask"] = stall_mask
        n_stalled = int(stall_mask.sum())
        if n_stalled > 0:
            stalled_indices = np.where(stall_mask)[0].tolist()
            logging.warning(
                "STALL detected: %d/%d panels stalled (indices: %s)",
                n_stalled,
                len(stall_mask),
                stalled_indices,
            )
            print(
                f"  *** STALL: {n_stalled}/{len(stall_mask)} panels stalled "
                f"(indices: {stalled_indices})"
            )
    else:
        results["stall_mask"] = None

    return np.array(results["F_distribution"]), body_aero, results


def run_frozen_geometry_alpha_sweep(
    body_aero,
    solver,
    *,
    va_magnitude: float,
    alpha_values_deg,
    side_slip_deg: float = 0.0,
    body_rates: float | np.ndarray = 0.0,
    body_axis: np.ndarray | None = None,
    reference_point: np.ndarray | None = None,
) -> list[dict]:
    """Direct VSM alpha sweep at a fixed (frozen) deformed geometry.

    For each requested angle of attack the apparent wind is imposed on the frozen
    mesh *exactly as the quasi-steady trim does it* — through
    ``body_aero.va_initialize`` with the SAME sideslip, body-rate, body axis and
    reference point — and a single ``solver.solve(body_aero)`` is run on the
    unchanged geometry (no re-trim).

    The freestream apparent wind is
    ``va = |va| * (cos a cos b, sin b, sin a)`` so that ``atan2(va_z, va_x) == a``
    and the sideslip is ``b``; the rigid-body rate is added as rotational inflow
    about ``body_axis`` (default ``-radial = (0, 0, -1)``) by the VSM ``va``
    setter, and ``phi_a`` is the tilt of the total aero force about that
    freestream apparent wind — matching
    ``vsm_quasi_steady.solve_vsm_quasi_steady_trim``.

    With the defaults (``side_slip_deg=0``, ``body_rates=0``) the sweep recovers
    the pure symmetric ``C_L(alpha)`` / ``C_D(alpha)`` / ``phi_a(alpha)`` response.
    To sweep *around* a turning anchor — so the swept row at the anchor's angle
    of attack reproduces the anchor state, including its sideslip-driven
    aerodynamic roll — pass the anchor's ``side_slip_deg`` and ``body_rates``.

    Args:
        body_aero: VSM ``BodyAerodynamics`` (already updated to the deformed
            leading/trailing-edge points); its inflow is overwritten per sample.
        solver: VSM ``Solver`` whose ``solve(body_aero)`` returns a results dict
            with wing ``"cl"``/``"cd"`` and a per-panel ``"F_distribution"``.
        va_magnitude: apparent-wind speed magnitude [m/s] held constant over the
            sweep.
        alpha_values_deg: iterable of angles of attack [deg].
        side_slip_deg: sideslip angle [deg] held constant over the sweep.
        body_rates: rigid-body rate(s) [rad/s] held constant over the sweep
            (scalar or per-axis), added as rotational inflow about ``body_axis``.
        body_axis: rotation axis (or axes) for ``body_rates``; defaults to
            ``-radial = (0, 0, -1)`` to match the trim.
        reference_point: reference point r0 for the rotational inflow
            ``v_rot(r) = omega x (r - r0)``; defaults to the origin.

    Returns:
        One dict per requested alpha with keys ``alpha`` [rad], ``cl``, ``cd``,
        ``phi_a`` [rad], ``v_a`` [m/s] and ``success``.  A failed solve yields a
        row with ``success=False`` and ``nan`` coefficients rather than raising,
        so one bad angle never aborts the sweep.
    """
    # phi_a uses the same force-vector definition as the identification dataset;
    # imported locally to avoid a module-load cycle (mirrors the local import in
    # ``plot_aero_forces_with_frames``).
    from awetrim.identification.aero_dataset import aerodynamic_roll

    # Lift/side reference axis for the aerodynamic-roll decomposition (the radial
    # axis, +z): a force purely along +z has zero aerodynamic roll. This is the
    # lift reference and is distinct from ``body_axis`` (the rigid-body rate axis).
    radial_axis = np.array([0.0, 0.0, 1.0])
    if body_axis is None:
        body_axis = -radial_axis

    beta_rad = float(np.radians(side_slip_deg))
    rows: list[dict] = []
    for alpha_deg in np.asarray(alpha_values_deg, dtype=float).ravel():
        alpha_rad = float(np.radians(alpha_deg))
        # Freestream apparent wind (before the per-panel rotational inflow), used
        # as the aerodynamic-roll reference exactly as the trim does.
        va_free = float(va_magnitude) * np.array(
            [
                np.cos(alpha_rad) * np.cos(beta_rad),
                np.sin(beta_rad),
                np.sin(alpha_rad),
            ]
        )
        try:
            body_aero.va_initialize(
                Umag=float(va_magnitude),
                angle_of_attack=float(alpha_deg),
                side_slip=float(side_slip_deg),
                body_rates=body_rates,
                body_axis=body_axis,
                reference_point=reference_point,
                rates_in_body_frame=False,
            )
            results = solver.solve(body_aero)
            cl = float(np.mean(np.asarray(results["cl"], dtype=float)))
            cd = float(np.mean(np.asarray(results["cd"], dtype=float)))
            total_force = np.asarray(results["F_distribution"], dtype=float).sum(axis=0)
            phi_a = float(aerodynamic_roll(total_force, va_free, radial_axis))
            success = True
        except Exception:
            logging.exception(
                "Frozen alpha sweep: VSM solve failed at alpha=%.2f deg.", alpha_deg
            )
            cl = cd = phi_a = float("nan")
            success = False
        rows.append(
            {
                "alpha": alpha_rad,
                "cl": cl,
                "cd": cd,
                "phi_a": phi_a,
                "v_a": float(va_magnitude),
                "success": success,
            }
        )
    return rows


def plot_aero_forces_with_frames(
    struc_nodes: np.ndarray,
    kite_connectivity_arr,
    m_arr: np.ndarray,
    panel_cp_locations: np.ndarray,
    f_aero_panel: np.ndarray,
    title: str = "Aero forces, body frame and course frame",
) -> plt.Figure:
    """3-D plot of the deformed kite structure with:

    - structural connectivity (thin grey lines)
    - total aerodynamic force arrow at each panel aerodynamic centre
    - course frame triad at the origin (dashed)
    - body frame triad (principal inertia axes) at the CG (solid)

    All coordinates are in the structural/VSM frame.  The course-frame unit
    vectors in that frame are X_C=[-1,0,0], Y_C=[0,-1,0], Z_C=[0,0,1].
    """
    from awetrim.identification.rigid_body_axes import compute_rigid_body_axes

    # ── body axes ────────────────────────────────────────────────────────────
    rba = compute_rigid_body_axes(struc_nodes, m_arr)
    cg = rba.cg
    body_axes = rba.body_axes  # rows: x_K, y_K, z_K in structural frame

    # ── arrow scale: 20 % of the bounding-box diagonal ───────────────────────
    bbox_diag = np.linalg.norm(struc_nodes.max(axis=0) - struc_nodes.min(axis=0))
    frame_len = 0.20 * bbox_diag

    # Scale force arrows so the largest force == frame_len
    f_mags = np.linalg.norm(f_aero_panel, axis=1)
    f_max = f_mags.max() if f_mags.max() > 0 else 1.0
    force_scale = frame_len / f_max

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ── structural connectivity ───────────────────────────────────────────────
    for conn in kite_connectivity_arr:
        i, j = int(conn[0]), int(conn[1])
        xs = [struc_nodes[i, 0], struc_nodes[j, 0]]
        ys = [struc_nodes[i, 1], struc_nodes[j, 1]]
        zs = [struc_nodes[i, 2], struc_nodes[j, 2]]
        ax.plot(xs, ys, zs, color="dimgrey", linewidth=0.8, alpha=0.6)

    ax.scatter(
        struc_nodes[:, 0], struc_nodes[:, 1], struc_nodes[:, 2],
        s=10, c="dimgrey", alpha=0.5, zorder=2,
    )

    # ── aerodynamic force arrows at panel ACs ─────────────────────────────────
    cp = np.asarray(panel_cp_locations)
    for k in range(len(cp)):
        fvec = f_aero_panel[k] * force_scale
        ax.quiver(
            cp[k, 0], cp[k, 1], cp[k, 2],
            fvec[0], fvec[1], fvec[2],
            color="tab:orange", linewidth=1.2, arrow_length_ratio=0.2,
        )
    # single legend proxy for forces
    ax.quiver([], [], [], [], [], [], color="tab:orange", linewidth=1.2,
              label=f"Aero force (max={f_max:.1f} N)")

    # ── frame triads: shared colours (x=red, y=green, z=blue) ────────────────
    frame_colors = ["tab:red", "tab:green", "tab:blue"]

    # course frame at origin (dashed, thinner)
    course_axes = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    course_labels = ["$X_C$", "$Y_C$", "$Z_C$"]
    for k, (col, lbl) in enumerate(zip(frame_colors, course_labels)):
        tip = course_axes[k] * frame_len * 0.6
        ax.quiver(0, 0, 0, tip[0], tip[1], tip[2],
                  color=col, linewidth=1.0, arrow_length_ratio=0.15,
                  linestyle="dashed", alpha=0.55)
        ax.text(*(tip * 1.08), lbl, color=col, fontsize=7)
    ax.plot([], [], color="grey", linestyle="dashed", linewidth=1.0,
            label="Course frame $C$")

    # body frame at CG (solid, thicker)
    body_labels = ["$x_K$", "$y_K$", "$z_K$"]
    for k, (col, lbl) in enumerate(zip(frame_colors, body_labels)):
        tip = body_axes[k] * frame_len
        ax.quiver(*cg, tip[0], tip[1], tip[2],
                  color=col, linewidth=2.2, arrow_length_ratio=0.15)
        ax.text(*(cg + tip * 1.08), lbl, color=col, fontsize=7)
    ax.plot([], [], color="grey", linestyle="solid", linewidth=2.0,
            label="Body frame $K$ (inertia)")

    ax.scatter(*cg, s=80, c="black", marker="*", zorder=5, label="CG")

    # ── equal aspect ratio ────────────────────────────────────────────────────
    all_pts = np.vstack([struc_nodes, cp])
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.15
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    ax.set_xlabel("$x_{struc}$ (m)")
    ax.set_ylabel("$y_{struc}$ (m)")
    ax.set_zlabel("$z_{struc}$ (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    return fig

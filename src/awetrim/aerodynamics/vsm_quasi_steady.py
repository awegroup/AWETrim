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

from __future__ import annotations

import copy
import inspect
import logging
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import casadi as ca
import numpy as np
from scipy.optimize import least_squares

from awetrim.aerodynamics.protocols import (
    AWETrimSystemModel,
    AxisDefinition,
    VsmBodyAerodynamics,
    VsmSolver,
)

DEFAULT_AXES = AxisDefinition(
    course=np.array([1.0, 0.0, 0.0], dtype=float),
    normal=np.array([0.0, 1.0, 0.0], dtype=float),
    radial=np.array([0.0, 0.0, 1.0], dtype=float),
)

DEFAULT_TRANSFORMATION_C_FROM_VSM = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# x = [speed_tangential, angle_roll_body_deg, angle_pitch_body_deg,
#      angle_yaw_body_deg, timeder_angle_course_body]
DEFAULT_BOUNDS_LOWER = np.array([-2.0, -15.0, -15.0, -15.0, -5.0], dtype=float)
DEFAULT_BOUNDS_UPPER = np.array([200.0, 15.0, 15.0, 15.0, 5.0], dtype=float)


def _default_vsm_solver(
    reference_point: np.ndarray,
    allowed_error: float = 1e-6,
    gamma_loop_type: str = "base",
) -> VsmSolver:
    try:
        from VSM.core.Solver import Solver
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "VSM is required when no solver is supplied. Install or expose "
            "`VSM.core.Solver.Solver`, or pass a solver implementing VsmSolver."
        ) from exc

    # ``allowed_error`` is the inner VSM circulation-loop convergence tolerance.
    # The base gamma loop converges linearly under relaxation, so loosening this
    # from the 1e-6 default to e.g. 1e-4 cuts the iteration count several-fold
    # with a negligible change in the trimmed state (see the wind-window sweep,
    # which exposes it as ``--gamma-tolerance``).
    #
    # ``gamma_loop_type`` selects the inner circulation solver:
    #   * "base" (default) -- relaxed-Picard fixed-point iteration.
    #   * "anderson" -- Anderson-accelerated: ~25x fewer inner iterations in
    #     attached flow, targeting the same fixed point. It is a clear win for
    #     *direct* single-state solves, BUT the outer trim solvers here use a
    #     finite-difference Jacobian that needs the inner residual to be a smooth
    #     function of x. Anderson's superlinear (jumpy) convergence makes its
    #     tolerance-terminated gamma slightly non-smooth, which corrupts the FD
    #     Jacobian at loose ``allowed_error`` (>=1e-6) and produces WRONG trim /
    #     trim-existence results. Only use "anderson" here with a tight
    #     ``gamma_tolerance`` (~1e-8), where it is correct and ~1.5-2x faster.
    #     See the VSM ``Solver.gamma_loop_anderson`` bake-off.
    return Solver(
        reference_point=reference_point,
        gamma_initial_distribution_type="zero",
        allowed_error=allowed_error,
        gamma_loop_type=gamma_loop_type,
    )


def _as_3vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Expected a 3-vector, got shape {np.asarray(value).shape}")
    return vector


def _numeric_value_for_symbol(system_model: AWETrimSystemModel, name: str) -> Any:
    if name == "speed_wind_ref" and hasattr(system_model.wind, "speed_wind_ref_value"):
        value = system_model.wind.speed_wind_ref_value
        if value is not None:
            return value
    for owner in (
        system_model,
        getattr(system_model, "wind", None),
        getattr(system_model, "kite", None),
        getattr(system_model, "tether", None),
    ):
        if owner is not None and hasattr(owner, name):
            value = getattr(owner, name)
            if not isinstance(value, (ca.MX, ca.SX)):
                return value
    raise ValueError(f"No numeric value available for symbolic variable '{name}'.")


def _as_numeric_3vector(system_model: AWETrimSystemModel, value: Any) -> np.ndarray:
    try:
        return _as_3vector(value)
    except Exception as first_error:
        if not isinstance(value, (ca.MX, ca.SX, ca.DM)):
            raise first_error

    symbols = ca.symvar(value)
    if not symbols:
        return _as_3vector(ca.DM(value).full())
    inputs = [
        _numeric_value_for_symbol(system_model, symbol.name()) for symbol in symbols
    ]
    func = ca.Function("awetrim_vsm_numeric_eval", symbols, [value])
    return _as_3vector(func(*inputs).full())


def _as_5vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (5,):
        raise ValueError(
            f"{name} must be shape (5,) for "
            "[speed_tangential, roll, pitch, yaw, timeder_angle_course_body]."
        )
    return vector


def _system_model_mass_wing(system_model: AWETrimSystemModel) -> float:
    if hasattr(system_model, "mass_wing"):
        return float(getattr(system_model, "mass_wing"))
    if hasattr(system_model, "kite") and hasattr(system_model.kite, "mass_wing"):
        return float(system_model.kite.mass_wing)
    raise AttributeError("system_model must expose mass_wing or kite.mass_wing.")


def _system_model_mass_total(system_model: AWETrimSystemModel) -> float:
    """Total kite mass (wing + KCU) for the d'Alembert inertial force.

    The inertial reaction acts on the whole kite, so it must include the KCU
    mass — matching the gravity force, which already covers wing + KCU. Using
    only ``mass_wing`` here (which once held the full structural sum but now is
    wing+bridle only) leaves the KCU out and unbalances the gravity-on trim.
    """
    mass_total = _system_model_mass_wing(system_model)
    if hasattr(system_model, "mass_kcu"):
        mass_total += float(getattr(system_model, "mass_kcu"))
    elif hasattr(system_model, "kite") and hasattr(system_model.kite, "mass_kcu"):
        mass_total += float(system_model.kite.mass_kcu)
    return mass_total


def _set_course_rate_body(
    system_model: AWETrimSystemModel, course_rate_body: float
) -> None:
    if hasattr(system_model, "timeder_angle_course_body"):
        system_model.timeder_angle_course_body = course_rate_body
    else:
        system_model.timeder_angle_course = course_rate_body


def _acceleration_course_body(system_model: AWETrimSystemModel) -> np.ndarray:
    if hasattr(system_model, "acceleration_course_body"):
        return _as_numeric_3vector(system_model, system_model.acceleration_course_body)
    return _as_numeric_3vector(system_model, system_model.acceleration)


def _force_gravity(system_model: AWETrimSystemModel) -> np.ndarray:
    if hasattr(system_model, "force_gravity"):
        return _as_numeric_3vector(system_model, system_model.force_gravity)
    if hasattr(system_model, "expression"):
        return _as_numeric_3vector(
            system_model, system_model.expression("force_gravity")
        )
    if hasattr(system_model, "kite"):
        return _as_numeric_3vector(
            system_model, system_model.kite.force_gravity_for(system_model)
        )
    raise AttributeError(
        "system_model must expose force_gravity, expression('force_gravity'), "
        "or kite.force_gravity_for(system_model)."
    )


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    axis_vec = _as_3vector(axis)
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm == 0.0:
        raise ValueError("Rotation axis must be non-zero.")
    axis_unit = axis_vec / axis_norm
    kx, ky, kz = axis_unit
    skew = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=float,
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _compose_attitude_rotation(
    *,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
) -> np.ndarray:
    roll_matrix = _rotation_matrix(axes.course, roll_deg)
    pitch_matrix = _rotation_matrix(axes.normal, pitch_deg)
    yaw_matrix = _rotation_matrix(axes.radial, yaw_deg)
    return yaw_matrix @ pitch_matrix @ roll_matrix


def _set_body_attitude_from_baseline(
    body: VsmBodyAerodynamics,
    *,
    baseline_sections: list[list[tuple[np.ndarray, np.ndarray]]],
    baseline_spanwise: list[np.ndarray],
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
    reference_point: np.ndarray,
) -> None:
    combined_rotation = _compose_attitude_rotation(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        axes=axes,
    )
    origin = _as_3vector(reference_point)

    def rotate_point(point: np.ndarray) -> np.ndarray:
        return origin + combined_rotation @ (_as_3vector(point) - origin)

    for wing, wing_sections, spanwise_base in zip(
        body.wings, baseline_sections, baseline_spanwise
    ):
        for section, (le_base, te_base) in zip(wing.sections, wing_sections):
            section.LE_point = rotate_point(le_base)
            section.TE_point = rotate_point(te_base)

        rotated_span = combined_rotation @ spanwise_base
        span_norm = np.linalg.norm(rotated_span)
        if span_norm == 0.0:
            raise ValueError(
                "Combined attitude produced zero spanwise direction vector."
            )
        wing.spanwise_direction = rotated_span / span_norm

    body.geometry_rotation = combined_rotation
    body._build_panels()


def _baseline_geometry(
    body: VsmBodyAerodynamics,
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], list[np.ndarray]]:
    baseline_sections: list[list[tuple[np.ndarray, np.ndarray]]] = []
    baseline_spanwise: list[np.ndarray] = []
    for wing in body.wings:
        baseline_sections.append(
            [
                (
                    np.asarray(section.LE_point, dtype=float).copy(),
                    np.asarray(section.TE_point, dtype=float).copy(),
                )
                for section in wing.sections
            ]
        )
        baseline_spanwise.append(
            np.asarray(wing.spanwise_direction, dtype=float).copy()
        )
    return baseline_sections, baseline_spanwise


def solve_vsm_quasi_steady_trim(
    body_aero: VsmBodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: AWETrimSystemModel,
    x_guess: np.ndarray,
    *,
    solver: VsmSolver | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    transformation_c_from_vsm: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    applied_moment_nm: np.ndarray | None = None,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-2,
    return_timing_breakdown: bool = False,
    max_nfev: int | None = None,
    prescribed_roll_deg: float | None = None,
    gamma_tolerance: float = 1e-6,
    gamma_loop: str = "base",
) -> tuple[dict[str, Any], VsmBodyAerodynamics]:
    """Solve one aerodynamic VSM quasi-steady trim state.

    The optimized state is ordered as
    `[speed_tangential, angle_roll_body_deg, angle_pitch_body_deg,
    angle_yaw_body_deg, timeder_angle_course_body]`.

    ``applied_moment_nm`` is an optional external moment (3-vector, N·m, in the
    ``axes`` = [course, normal, radial] basis) added to the CG moment balance —
    e.g. a KCU steering roll moment about the course axis. The wing then banks
    and turns to react it, so the resulting roll (``opt_x[1]`` /
    ``aero_roll_deg``) and course rate (``opt_x[4]``) are *outputs*. ``None``
    (default) leaves the free-trim behaviour unchanged.

    ``prescribed_roll_deg`` pins the roll DOF to a kinematically prescribed
    value (geometric bridle steering: the inextensible steering lines dictate
    the roll, and transmit whatever roll moment is needed as a constraint
    reaction). The roll-moment residual ``cmx`` is then dropped from the
    objective and from the ``success_physical`` test — the solve is 4 unknowns
    ``[speed, pitch, yaw, course_rate]`` against ``[cmy, cmz, cfx, cfy]`` —
    and reported as the bridle reaction ``reaction_roll_moment_nm``. ``None``
    (default) keeps roll free.
    """

    bounds_lower = _as_5vector(bounds_lower, "bounds_lower")
    bounds_upper = _as_5vector(bounds_upper, "bounds_upper")
    x_guess = _as_5vector(x_guess, "x_guess")
    center_of_gravity = _as_3vector(center_of_gravity)
    reference_point = _as_3vector(reference_point)
    transformation_c_from_vsm = np.asarray(transformation_c_from_vsm, dtype=float)
    applied_moment_nm = (
        np.zeros(3, dtype=float)
        if applied_moment_nm is None
        else _as_3vector(applied_moment_nm)
    )

    # Seed kinematics so omega can be evaluated before the solver starts.
    system_model.speed_tangential = float(x_guess[0])
    _set_course_rate_body(system_model, float(x_guess[4]))

    if transformation_c_from_vsm.shape != (3, 3):
        raise ValueError("transformation_c_from_vsm must be shape (3, 3).")
    if np.any(bounds_lower >= bounds_upper):
        raise ValueError("Each lower bound must be smaller than its upper bound.")

    if solver is None:
        solver = _default_vsm_solver(reference_point, gamma_tolerance, gamma_loop)

    def evaluate_kinematics(x: np.ndarray) -> dict[str, np.ndarray]:
        speed_tangential, _roll, _pitch, _yaw, course_rate_body = x
        _set_course_rate_body(system_model, course_rate_body)
        system_model.speed_tangential = speed_tangential

        inertial_force = -_system_model_mass_total(system_model) * _as_3vector(
            transformation_c_from_vsm @ _acceleration_course_body(system_model)
        )
        gravity_force = _as_3vector(
            transformation_c_from_vsm @ _force_gravity(system_model)
        )
        wind_velocity = _as_numeric_3vector(
            system_model,
            transformation_c_from_vsm @ system_model.wind.velocity_wind(system_model),
        )
        kite_velocity = _as_numeric_3vector(
            system_model, transformation_c_from_vsm @ system_model.velocity_kite
        )
        apparent_velocity = _as_numeric_3vector(
            system_model,
            transformation_c_from_vsm @ system_model.velocity_apparent_wind,
        )
        return {
            "va": apparent_velocity,
            "inertial_force": inertial_force,
            "gravity_force": gravity_force,
            "wind_velocity": wind_velocity,
            "kite_velocity": kite_velocity,
            "apparent_velocity": apparent_velocity,
        }

    timing_counters = {
        "residual_evaluations": 0,
        "residual_total_s": 0.0,
        "body_rotate_s": 0.0,
        "kinematics_s": 0.0,
        "solver_s": 0.0,
        "postprocess_s": 0.0,
    }
    cached_eval: dict[str, Any] = {"x": None, "payload": None}
    working_body = copy.deepcopy(body_aero)
    baseline_sections, baseline_spanwise = _baseline_geometry(working_body)

    def moment_residual(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        cached_x = cached_eval["x"]
        if cached_x is not None and np.array_equal(x, cached_x):
            return np.asarray(cached_eval["payload"]["residual"], dtype=float)

        eval_t0 = perf_counter()
        _speed_tangential, roll_deg, pitch_deg, yaw_deg, course_rate_body = x

        t0 = perf_counter()
        _set_body_attitude_from_baseline(
            working_body,
            baseline_sections=baseline_sections,
            baseline_spanwise=baseline_spanwise,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            axes=axes,
            reference_point=reference_point,
        )
        timing_counters["body_rotate_s"] += perf_counter() - t0

        t0 = perf_counter()
        kin = evaluate_kinematics(x)
        va = _as_3vector(kin["va"])
        inertial_force = _as_3vector(kin["inertial_force"])
        gravity_force = (
            _as_3vector(kin.get("gravity_force", np.zeros(3, dtype=float)))
            if include_gravity
            else np.zeros(3, dtype=float)
        )
        timing_counters["kinematics_s"] += perf_counter() - t0

        aoa_course_deg = np.rad2deg(np.arctan2(va[2], va[0]))
        beta_course_deg = np.rad2deg(np.arctan2(va[1], np.hypot(va[0], va[2])))
        umag = np.linalg.norm(va)
        if umag <= 0.0:
            raise ValueError("Apparent wind magnitude must be positive.")

        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_course_deg,
            side_slip=beta_course_deg,
            body_rates=course_rate_body,
            body_axis=-axes.radial,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )

        t0 = perf_counter()
        res = solver.solve(working_body)
        timing_counters["solver_s"] += perf_counter() - t0

        cmx = float(res.get("cmx", np.nan))
        cmy = float(res.get("cmy", np.nan))
        cmz = float(res.get("cmz", np.nan))
        total_aero_force = np.array(
            [
                float(res.get("Fx", np.nan)),
                float(res.get("Fy", np.nan)),
                float(res.get("Fz", np.nan)),
            ],
            dtype=float,
        )

        projected_area = float(working_body.wings[0].compute_projected_area())
        if projected_area <= 0.0:
            raise ValueError("VSM body projected area must be positive.")
        max_chord = max(float(panel.chord) for panel in working_body.panels)
        q_inf = 0.5 * float(solver.rho) * umag**2
        denom = q_inf * projected_area * max_chord if max_chord > 0.0 else 1.0

        moment_vec = np.cross(center_of_gravity - reference_point, inertial_force)
        if include_gravity:
            moment_vec += np.cross(center_of_gravity - reference_point, gravity_force)
        delta_cm = moment_vec / denom

        cmx += delta_cm[0]
        cmy += delta_cm[1]
        cmz += delta_cm[2]

        # External applied moment (e.g. KCU steering roll moment). Converted to
        # a coefficient with the same denominator the inertial delta uses, so
        # the balance is aero + inertial + applied = 0.
        if np.any(applied_moment_nm):
            applied_cm = applied_moment_nm / denom
            cmx += applied_cm[0]
            cmy += applied_cm[1]
            cmz += applied_cm[2]

        net_force = total_aero_force + inertial_force + gravity_force
        force_denom = q_inf * projected_area
        cfx = np.dot(net_force, axes.course) / force_denom
        cfy = np.dot(net_force, axes.normal) / force_denom

        t0 = perf_counter()
        residual = np.array([cmx, cmy, cmz, cfx, cfy], dtype=float)
        timing_counters["postprocess_s"] += perf_counter() - t0
        timing_counters["residual_evaluations"] += 1
        timing_counters["residual_total_s"] += perf_counter() - eval_t0
        cached_eval["x"] = x.copy()
        cached_eval["payload"] = {
            "residual": residual,
            "kin": kin,
            "va": va,
            "umag": umag,
            "res": res,
            "gravity_force": gravity_force,
            "inertial_force": inertial_force,
            "denom": denom,
        }
        return residual

    if prescribed_roll_deg is None:
        opt = least_squares(
            lambda x: moment_residual(x),
            np.clip(x_guess, bounds_lower, bounds_upper),
            bounds=(bounds_lower, bounds_upper),
            max_nfev=max_nfev,
        )
        opt_x = np.asarray(opt.x, dtype=float)
    else:
        # Kinematic bridle steering: roll is a constraint, not a DOF. Optimize
        # the 4 remaining unknowns against [cmy, cmz, cfx, cfy]; cmx becomes
        # the roll reaction the steering lines must carry.
        free = np.array([0, 2, 3, 4], dtype=int)
        roll_fixed = float(prescribed_roll_deg)

        def _to_full(x4: np.ndarray) -> np.ndarray:
            x = np.empty(5, dtype=float)
            x[free] = np.asarray(x4, dtype=float)
            x[1] = roll_fixed
            return x

        opt = least_squares(
            lambda x4: moment_residual(_to_full(x4))[1:],
            np.clip(x_guess[free], bounds_lower[free], bounds_upper[free]),
            bounds=(bounds_lower[free], bounds_upper[free]),
            max_nfev=max_nfev,
        )
        opt_x = _to_full(opt.x)

    cm_best = moment_residual(opt_x)
    cmx, cmy, cmz, cfx, cfy = cm_best
    if prescribed_roll_deg is None:
        physical_success = bool(
            np.abs(cmx) < moment_tolerance
            and np.abs(cmy) < moment_tolerance
            and np.abs(cmz) < moment_tolerance
        )
    else:
        # cmx is the bridle-line reaction, not a residual to drive to zero.
        physical_success = bool(
            np.abs(cmy) < moment_tolerance and np.abs(cmz) < moment_tolerance
        )

    payload = (
        cached_eval["payload"] if np.array_equal(opt_x, cached_eval["x"]) else None
    )
    if payload is None:
        _ = moment_residual(opt_x)
        payload = cached_eval["payload"]

    kin = payload["kin"]
    va = _as_3vector(payload["va"])
    umag = float(payload["umag"])
    res = payload["res"]
    aoa_course_deg = float(np.rad2deg(np.arctan2(va[2], va[0])))
    beta_course_deg = float(np.rad2deg(np.arctan2(va[1], np.hypot(va[0], va[2]))))
    aoa_center_chord_deg = float(res.get("alpha_center_chord_deg", aoa_course_deg))
    beta_center_chord_deg = float(res.get("beta_center_chord_deg", beta_course_deg))

    total_aero_force = np.array(
        [
            float(res.get("Fx", np.nan)),
            float(res.get("Fy", np.nan)),
            float(res.get("Fz", np.nan)),
        ],
        dtype=float,
    )
    va_unit = va / np.linalg.norm(va)
    lift_dir = axes.radial - np.dot(axes.radial, va_unit) * va_unit
    side_dir = np.cross(lift_dir, va_unit)
    aero_roll_deg = float(
        np.rad2deg(
            np.arctan2(
                np.dot(total_aero_force, side_dir),
                np.dot(total_aero_force, lift_dir),
            )
        )
    )

    inertial_force = _as_3vector(payload["inertial_force"])
    gravity_force = _as_3vector(payload["gravity_force"])
    x_cp = res.get("center_of_pressure", np.nan)
    x_cp_arr = np.asarray(x_cp, dtype=float)
    x_cp_point = (
        x_cp_arr.reshape(3)
        if x_cp_arr.size == 3
        else np.array([float(x_cp_arr), 0.0, 0.0])
    )
    tether_force = float(total_aero_force[2] + gravity_force[2] + inertial_force[2])

    result: dict[str, Any] = {
        "opt_x": opt_x,
        "cm": np.array([cmx, cmy, cmz], dtype=float),
        "cfx": float(cfx),
        "cfy": float(cfy),
        "side_slip_deg": beta_center_chord_deg,
        "side_slip_course_deg": beta_course_deg,
        "aero_roll_deg": aero_roll_deg,
        "aoa_deg": aoa_center_chord_deg,
        "aoa_course_deg": aoa_course_deg,
        "success": bool(opt.success),
        "success_physical": physical_success,
        "gravity_force": gravity_force,
        "inertial_force": inertial_force,
        "cl": res.get("cl"),
        "cd": res.get("cd"),
        "total_aero_force_vec": total_aero_force,
        "x_cp_point": x_cp_point,
        "wind_vel_world": _as_3vector(kin.get("wind_velocity", np.zeros(3))),
        "kite_vel_world": _as_3vector(kin.get("kite_velocity", np.zeros(3))),
        "va_vel_world": _as_3vector(kin.get("apparent_velocity", va)),
        "Umag": umag,
        "course_axis": axes.course,
        "radial_axis": axes.radial,
        "normal_axis": axes.normal,
        "F_distribution": res.get("F_distribution"),
        "panel_cp_locations": res.get("panel_cp_locations"),
        "alpha_at_ac": res.get("alpha_at_ac"),
        "gamma_distribution": res.get("gamma_distribution"),
        "tether_force": tether_force,
        "optimizer": opt,
    }
    if prescribed_roll_deg is not None:
        result["prescribed_roll_deg"] = float(prescribed_roll_deg)
        result["reaction_roll_moment_nm"] = float(cmx * payload["denom"])

    if return_timing_breakdown:
        residual_total = float(timing_counters["residual_total_s"])
        if residual_total > 0.0:
            timing_counters["solver_share"] = (
                timing_counters["solver_s"] / residual_total
            )
            timing_counters["body_rotate_share"] = (
                timing_counters["body_rotate_s"] / residual_total
            )
            timing_counters["kinematics_share"] = (
                timing_counters["kinematics_s"] / residual_total
            )
            timing_counters["postprocess_share"] = (
                timing_counters["postprocess_s"] / residual_total
            )
        result["timing_breakdown"] = timing_counters

    return result, working_body


def turn_radius_vs_steer_moment(
    body_aero: VsmBodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: AWETrimSystemModel,
    steer_moments_nm: Sequence[float],
    x_guess: np.ndarray,
    *,
    solver: VsmSolver | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    steer_gain_nm_per_us: float | None = None,
    moment_tolerance: float = 1e-2,
    max_nfev: int | None = None,
) -> dict[str, Any]:
    """Map an applied KCU steering roll moment to the trimmed turn.

    For each roll moment in ``steer_moments_nm`` this solves the gravity-free
    (by default) quasi-steady trim with that moment applied about the course
    axis (see :func:`solve_vsm_quasi_steady_trim`'s ``applied_moment_nm``) and
    records the resulting bank, aerodynamic roll ``phi_a``, turn rate and turn
    radius ``R = speed_tangential / |course_rate|``, plus L/D, tether force,
    speed and side-slip. This is the design-tool analogue of the point-mass
    steering law ``angle_roll_aerodynamic = k_steering * u_s`` in
    ``awetrim.system.kite`` — here the roll↔steering relation is *computed* by
    VSM rather than identified.

    When ``steer_gain_nm_per_us`` is given, each moment is also reported as a
    steering deflection ``u_s = M / steer_gain_nm_per_us`` (a first-order
    bridle-lever model), and an effective ``k_steering = d(phi_a)/d(u_s)``
    [rad per unit ``u_s``] is fitted across the sweep for comparison with the
    identified gain.

    Points are solved outward from zero moment in each sign direction, each
    direction warm-started from the straight-flight solution, so the sweep is
    robust across the sign flip.

    Returns
    -------
    dict
        Arrays (one entry per input moment, in the input order) under keys
        ``steer_moment_nm``, ``input_steering``, ``roll_body_deg``,
        ``aero_roll_deg``, ``course_rate``, ``turn_radius``,
        ``speed_tangential``, ``cl``, ``cd``, ``lift_over_drag``,
        ``tether_force``, ``side_slip_deg``, ``success_physical``; plus scalar
        ``k_steering_effective`` (``nan`` if no gain given) and the raw
        ``records`` list.
    """
    moments = np.asarray(list(steer_moments_nm), dtype=float).reshape(-1)
    bounds_lower = _as_5vector(bounds_lower, "bounds_lower")
    bounds_upper = _as_5vector(bounds_upper, "bounds_upper")
    x_guess = np.clip(_as_5vector(x_guess, "x_guess"), bounds_lower, bounds_upper)
    roll_axis = _as_3vector(axes.course)
    if solver is None:
        solver = _default_vsm_solver(_as_3vector(reference_point))

    def _solve_one(
        moment: float, warm: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray]:
        res, _ = solve_vsm_quasi_steady_trim(
            body_aero=body_aero,
            center_of_gravity=center_of_gravity,
            reference_point=reference_point,
            system_model=system_model,
            x_guess=warm,
            solver=solver,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            include_gravity=include_gravity,
            applied_moment_nm=moment * roll_axis,
            axes=axes,
            moment_tolerance=moment_tolerance,
            max_nfev=max_nfev,
        )
        return res, np.clip(
            np.asarray(res["opt_x"], dtype=float), bounds_lower, bounds_upper
        )

    # Solve outward from zero in each direction, warm-started from straight flight.
    order = np.argsort(np.abs(moments))
    results: list[dict[str, Any] | None] = [None] * moments.size
    warm_pos = x_guess.copy()
    warm_neg = x_guess.copy()
    for idx in order:
        moment = float(moments[idx])
        if moment >= 0.0:
            res, warm_pos = _solve_one(moment, warm_pos)
        else:
            res, warm_neg = _solve_one(moment, warm_neg)
        results[idx] = res

    def _turn_radius(res: dict[str, Any]) -> float:
        opt_x = np.asarray(res["opt_x"], dtype=float)
        v_tau, course_rate = float(opt_x[0]), float(opt_x[4])
        return float(v_tau / abs(course_rate)) if abs(course_rate) > 1e-9 else np.inf

    def _lift_over_drag(res: dict[str, Any]) -> float:
        cl, cd = res.get("cl"), res.get("cd")
        if cl is None or cd is None or float(cd) == 0.0:
            return float("nan")
        return float(cl) / float(cd)

    out: dict[str, Any] = {
        "steer_moment_nm": moments,
        "input_steering": (
            moments / float(steer_gain_nm_per_us)
            if steer_gain_nm_per_us
            else np.full(moments.size, np.nan)
        ),
        "roll_body_deg": np.array([float(r["opt_x"][1]) for r in results]),
        "aero_roll_deg": np.array([float(r["aero_roll_deg"]) for r in results]),
        "course_rate": np.array([float(r["opt_x"][4]) for r in results]),
        "turn_radius": np.array([_turn_radius(r) for r in results]),
        "speed_tangential": np.array([float(r["opt_x"][0]) for r in results]),
        "cl": np.array([float(r["cl"]) for r in results]),
        "cd": np.array([float(r["cd"]) for r in results]),
        "lift_over_drag": np.array([_lift_over_drag(r) for r in results]),
        "tether_force": np.array([float(r["tether_force"]) for r in results]),
        "side_slip_deg": np.array([float(r["side_slip_deg"]) for r in results]),
        "success_physical": np.array([bool(r["success_physical"]) for r in results]),
        "records": results,
    }

    # Effective steering gain d(phi_a)/d(u_s) [rad per u_s], fitted across the
    # sweep when a deflection gain is provided and the points span a range.
    k_steering_effective = float("nan")
    if steer_gain_nm_per_us:
        u_s = out["input_steering"]
        phi_a_rad = np.deg2rad(out["aero_roll_deg"])
        good = np.isfinite(u_s) & np.isfinite(phi_a_rad) & out["success_physical"]
        if np.count_nonzero(good) >= 2 and np.ptp(u_s[good]) > 1e-9:
            k_steering_effective = float(np.polyfit(u_s[good], phi_a_rad[good], 1)[0])
    out["k_steering_effective"] = k_steering_effective
    return out


def steering_delta_limit(steering_h: float, steering_b: float) -> float:
    """Maximum |delta| [m] of the rigid-segment bridle-steering triangle."""
    h, b = float(steering_h), float(steering_b)
    c = h**2 + (b / 2.0) ** 2
    return 0.5 * (np.sqrt(c + h * b) - np.sqrt(max(c - h * b, 0.0)))


def roll_angle_from_steering_delta(
    steering_h: float, steering_b: float, delta_m: float
) -> float:
    """Rigid-body roll angle [deg] from a steering line-length difference [m].

    Purely geometric bridle-steering model (the triangle from
    ``scripts/personal/aerodynamics/calculate_max_roll.py``): the kite hangs
    from the KCU apex by two lines attached at the ends of a rigid segment of
    span ``steering_b`` whose midpoint sits ``steering_h`` above the KCU (both
    in the roll plane, i.e. y-z distances). A steering actuation making the
    two line lengths differ by ``2 * delta_m``
    (``delta_m = (L_left - L_right) / 2`` [m]) tilts the segment about its
    midpoint by

        sin(theta) = -2 * delta * sqrt(c - delta^2) / (h * b),
        c = h^2 + (b / 2)^2.

    The roll is *kinematically prescribed* by the inextensible lines — no
    moment balance involved; the lines carry the roll reaction. ``delta_m`` is
    clamped just inside the geometric limit
    (:func:`steering_delta_limit`). Positive ``delta_m`` (left line longer)
    gives negative ``theta``. Small-delta gain:
    ``dtheta/ddelta = -2 sqrt(c) / (h b)`` [rad/m].
    """
    h, b = float(steering_h), float(steering_b)
    c = h**2 + (b / 2.0) ** 2
    hb = h * b
    if hb <= 1e-12:
        return 0.0
    delta_lim = steering_delta_limit(h, b)
    delta = float(np.clip(delta_m, -0.999 * delta_lim, 0.999 * delta_lim))
    arg = -(2.0 * delta * np.sqrt(max(c - delta**2, 0.0))) / hb
    return float(np.degrees(np.arcsin(np.clip(arg, -1.0, 1.0))))


def turn_radius_vs_steering_delta(
    body_aero: VsmBodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: AWETrimSystemModel,
    steering_deltas_m: Sequence[float],
    x_guess: np.ndarray,
    *,
    steering_h: float,
    steering_b: float,
    tip_midpoint: np.ndarray,
    solver: VsmSolver | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-2,
    max_nfev: int | None = None,
) -> dict[str, Any]:
    """Map a geometric bridle-steering input ``delta`` [m] to the trimmed turn.

    For each steering line-length half-difference in ``steering_deltas_m`` the
    kite roll is *kinematically prescribed* by the bridle triangle
    (:func:`roll_angle_from_steering_delta` with ``steering_h`` /
    ``steering_b``): the baseline geometry is rotated by ``theta`` about the
    course axis through ``tip_midpoint`` (the midpoint of the steering-line
    attachment segment, which the rigid-segment model holds fixed), and the
    quasi-steady trim is solved with the roll DOF pinned
    (``prescribed_roll_deg=0`` relative to the rotated baseline) and the
    roll-moment residual dropped — the steering lines carry that reaction,
    reported as ``reaction_roll_moment_nm``.

    ``center_of_gravity`` is held at its unrotated location: the KCU term
    (usually dominant) does not roll with the wing and the wing CG sits near
    the rotation point, so its shift is second order.

    Points are solved outward from zero delta in each sign direction, each
    warm-started from the previous solution.

    Returns
    -------
    dict
        Arrays (one entry per input delta, in input order) under keys
        ``steering_delta_m``, ``roll_prescribed_deg`` (= ``roll_body_deg``),
        ``aero_roll_deg``, ``course_rate``, ``turn_radius``,
        ``speed_tangential``, ``cl``, ``cd``, ``lift_over_drag``,
        ``tether_force``, ``side_slip_deg``, ``reaction_roll_moment_nm``,
        ``success_physical``; scalars ``k_steering_effective``
        (fitted ``d(phi_a)/d(delta)`` [rad/m], ``nan`` if not fittable),
        ``k_roll_geometric`` (analytic ``|dtheta/ddelta|`` at 0 [rad/m]),
        ``steering_h``, ``steering_b``; plus the raw ``records`` list.
    """
    deltas = np.asarray(list(steering_deltas_m), dtype=float).reshape(-1)
    bounds_lower = _as_5vector(bounds_lower, "bounds_lower")
    bounds_upper = _as_5vector(bounds_upper, "bounds_upper")
    x_guess = np.clip(_as_5vector(x_guess, "x_guess"), bounds_lower, bounds_upper)
    tip_midpoint = _as_3vector(tip_midpoint)
    course_axis = _as_3vector(axes.course)
    if solver is None:
        solver = _default_vsm_solver(_as_3vector(reference_point))

    def _solve_one(delta: float, warm: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        theta = roll_angle_from_steering_delta(steering_h, steering_b, delta)
        body_i = copy.deepcopy(body_aero)
        if theta != 0.0:
            body_i.rotate(angle_deg=theta, axis=course_axis, point=tip_midpoint)
        res, _ = solve_vsm_quasi_steady_trim(
            body_aero=body_i,
            center_of_gravity=center_of_gravity,
            reference_point=reference_point,
            system_model=system_model,
            x_guess=warm,
            solver=solver,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            include_gravity=include_gravity,
            axes=axes,
            moment_tolerance=moment_tolerance,
            max_nfev=max_nfev,
            prescribed_roll_deg=0.0,  # roll is baked into the rotated baseline
        )
        res["roll_prescribed_deg"] = theta
        return res, np.clip(
            np.asarray(res["opt_x"], dtype=float), bounds_lower, bounds_upper
        )

    # Solve outward from zero in each direction, warm-started from straight flight.
    order = np.argsort(np.abs(deltas))
    results: list[dict[str, Any] | None] = [None] * deltas.size
    warm_pos = x_guess.copy()
    warm_neg = x_guess.copy()
    for idx in order:
        delta = float(deltas[idx])
        if delta >= 0.0:
            res, warm_pos = _solve_one(delta, warm_pos)
        else:
            res, warm_neg = _solve_one(delta, warm_neg)
        results[idx] = res

    def _turn_radius(res: dict[str, Any]) -> float:
        opt_x = np.asarray(res["opt_x"], dtype=float)
        v_tau, course_rate = float(opt_x[0]), float(opt_x[4])
        return float(v_tau / abs(course_rate)) if abs(course_rate) > 1e-9 else np.inf

    def _lift_over_drag(res: dict[str, Any]) -> float:
        cl, cd = res.get("cl"), res.get("cd")
        if cl is None or cd is None or float(cd) == 0.0:
            return float("nan")
        return float(cl) / float(cd)

    thetas = np.array([float(r["roll_prescribed_deg"]) for r in results])
    out: dict[str, Any] = {
        "steering_delta_m": deltas,
        "roll_prescribed_deg": thetas,
        "roll_body_deg": thetas,
        "aero_roll_deg": np.array([float(r["aero_roll_deg"]) for r in results]),
        "course_rate": np.array([float(r["opt_x"][4]) for r in results]),
        "turn_radius": np.array([_turn_radius(r) for r in results]),
        "speed_tangential": np.array([float(r["opt_x"][0]) for r in results]),
        "cl": np.array([float(r["cl"]) for r in results]),
        "cd": np.array([float(r["cd"]) for r in results]),
        "lift_over_drag": np.array([_lift_over_drag(r) for r in results]),
        "tether_force": np.array([float(r["tether_force"]) for r in results]),
        "side_slip_deg": np.array([float(r["side_slip_deg"]) for r in results]),
        "reaction_roll_moment_nm": np.array(
            [float(r.get("reaction_roll_moment_nm", np.nan)) for r in results]
        ),
        "success_physical": np.array([bool(r["success_physical"]) for r in results]),
        "records": results,
    }

    # Effective aero-roll gain d(phi_a)/d(delta) [rad/m] fitted across the sweep.
    k_steering_effective = float("nan")
    phi_a_rad = np.deg2rad(out["aero_roll_deg"])
    good = np.isfinite(phi_a_rad) & out["success_physical"]
    if np.count_nonzero(good) >= 2 and np.ptp(deltas[good]) > 1e-9:
        k_steering_effective = float(np.polyfit(deltas[good], phi_a_rad[good], 1)[0])
    out["k_steering_effective"] = k_steering_effective
    c = float(steering_h) ** 2 + (float(steering_b) / 2.0) ** 2
    out["k_roll_geometric"] = float(
        2.0 * np.sqrt(c) / (float(steering_h) * float(steering_b))
    )
    out["steering_h"] = float(steering_h)
    out["steering_b"] = float(steering_b)
    return out


def solve_vsm_qs_trim_with_williams_tether(
    body_aero: VsmBodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: AWETrimSystemModel,
    x_guess: np.ndarray,
    *,
    williams_x_guess: np.ndarray | None = None,
    williams_bounds_lower: np.ndarray | None = None,
    williams_bounds_upper: np.ndarray | None = None,
    solver: VsmSolver | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    transformation_c_from_vsm: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = True,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-2,
    max_nfev: int | None = None,
    gamma_tolerance: float = 1e-6,
    gamma_loop: str = "base",
    tether_model: str = "williams",
) -> tuple[dict[str, Any], VsmBodyAerodynamics]:
    """Coupled VSM trim with a consistent (off-radial) tether force.

    Unlike a radial-tether approximation, the tether's own off-radial reaction
    (aerodynamic drag + weight) enters the kite force balance — a large effect
    for long tethers in crosswind flight (tether drag is a dominant AWES loss).

    ``tether_model`` selects the tether:

      * ``"williams"`` (default) — the full Williams distributed-mass shape. The
        kite-end tether vector is baked in as the trim resultant (Williams
        "collapsed" mode), so the 3-D force balance ``F_tether = -net`` holds by
        construction and the tether shape only has to reach the ground anchor.
        6 unknowns ``[speed, roll, pitch, yaw, course_rate, tether_length]`` /
        6 residuals ``[cmx, cmy, cmz, ground(3)]``. Per-node local apparent wind,
        so the tether drag is resolved along the (curved) tether.
      * ``"rigid_lumped"`` — the ROM's ``RigidLumpedTether`` model: a radial
        tension plus a lumped off-radial drag + half-weight evaluated at the
        kite, no shape/ground closure. 6 unknowns
        ``[speed, roll, pitch, yaw, course_rate, tension]`` / 6 residuals
        ``[cmx, cmy, cmz, cfx, cfy, cfz]``. Cheaper/simpler and matches the ROM
        cycle model, but overestimates tether drag (kite-station apparent wind
        applied to the whole tether).

    Both feed the kite ``net = total_aero_force + inertial + gravity`` (wing +
    KCU masses); ``r_kite`` is ``distance_radial * axes.radial`` in the trim
    (VSM) frame. See ``src/awetrim/system/`` for the tether models.
    """
    from awetrim.system.williams_tether import WilliamsTether
    from awetrim.utils.reference_frames import transformation_C_from_W

    tether = getattr(system_model, "tether", None)
    if not isinstance(tether, WilliamsTether):
        raise TypeError(
            "solve_vsm_qs_trim_with_williams_tether requires a WilliamsTether "
            f"instance on system_model.tether; got {type(tether).__name__}."
        )

    bounds_lower = _as_5vector(bounds_lower, "bounds_lower")
    bounds_upper = _as_5vector(bounds_upper, "bounds_upper")
    x_guess = _as_5vector(x_guess, "x_guess")
    center_of_gravity = _as_3vector(center_of_gravity)
    reference_point = _as_3vector(reference_point)
    transformation_c_from_vsm = np.asarray(transformation_c_from_vsm, dtype=float)

    distance_radial = float(
        _numeric_value_for_symbol(system_model, "distance_radial")
        if hasattr(system_model, "distance_radial")
        else 200.0
    )
    if distance_radial <= 0.0:
        raise ValueError("distance_radial must be positive for Williams integration.")

    # Frame angles (used below for the wind-frame transforms and r_kite).
    angle_az = float(_numeric_value_for_symbol(system_model, "angle_azimuth"))
    angle_elev = float(_numeric_value_for_symbol(system_model, "angle_elevation"))
    angle_course = float(_numeric_value_for_symbol(system_model, "angle_course"))

    _valid_models = ("williams", "rigid_lumped")
    if tether_model not in _valid_models:
        raise ValueError(
            f"tether_model must be one of {_valid_models}; got {tether_model!r}."
        )

    # Single tether unknown per model: Williams solves the tether length (the
    # kite-end force vector is baked in as the resultant); rigid-lumped solves
    # the radial tension magnitude.
    if tether_model == "rigid_lumped":
        _wg = np.array([3.0e3], dtype=float)
        _wlb = np.array([0.0], dtype=float)
        _wub = np.array([1.0e6], dtype=float)
    else:  # williams
        _wg = np.array([distance_radial * 1.02], dtype=float)
        _wlb = np.array([0.99 * distance_radial], dtype=float)
        _wub = np.array([1.4 * distance_radial], dtype=float)

    if williams_x_guess is None:
        williams_x_guess = _wg
    williams_x_guess = np.asarray(williams_x_guess, dtype=float).reshape(-1)

    if williams_bounds_lower is None:
        williams_bounds_lower = _wlb
    if williams_bounds_upper is None:
        williams_bounds_upper = _wub

    lb = np.concatenate([bounds_lower, williams_bounds_lower])
    ub = np.concatenate([bounds_upper, williams_bounds_upper])
    x0 = np.concatenate(
        [np.clip(x_guess, bounds_lower, bounds_upper), williams_x_guess]
    )

    if solver is None:
        solver = _default_vsm_solver(reference_point, gamma_tolerance, gamma_loop)

    # Seed the system-model kinematics so the symbolic
    # ``velocity_rotation_course_frame`` (which depends on speed_tangential and
    # the course rate) can be evaluated numerically below. Mirrors the seed
    # block in ``solve_vsm_quasi_steady_trim``.
    system_model.speed_tangential = float(x_guess[0])
    _set_course_rate_body(system_model, float(x_guess[4]))

    # --- Capture env values from system_model. The Williams tether reads
    # wind/rho/g/omega via the explicit ``env`` argument; nothing is stored on
    # the tether instance. ---
    wind = getattr(system_model, "wind", None)
    mass_wing_value = float(_system_model_mass_wing(system_model))
    mass_kcu = float(
        getattr(getattr(system_model, "kite", system_model), "mass_kcu", 0.0)
    )
    mass_total = mass_wing_value + mass_kcu

    # --- Frame transformations into the wind frame Williams expects. ---
    # The trim residual operates in VSM body axes; system_model angles are in
    # the system course frame; the wind frame is the world frame rotated about
    # +z by -direction_wind so that wind blows along +x (= Williams' wind law).
    T_Csm_from_W = np.asarray(
        ca.DM(transformation_C_from_W(angle_az, angle_elev, angle_course)).full(),
        dtype=float,
    )
    T_W_from_Csm = T_Csm_from_W.T
    # The trim code uses transformation_c_from_vsm to move course-frame
    # quantities into the VSM body frame (the matrix is its own inverse).
    T_Csm_from_VSM = transformation_c_from_vsm
    direction_wind = float(
        getattr(getattr(system_model, "wind", None), "direction_wind", 0.0)
    )
    T_Wind_from_W = np.array(
        [
            [np.cos(-direction_wind), -np.sin(-direction_wind), 0.0],
            [np.sin(-direction_wind), np.cos(-direction_wind), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # Composed transformations.
    T_Wind_from_VSM = T_Wind_from_W @ T_W_from_Csm @ T_Csm_from_VSM
    T_Wind_from_Csm = T_Wind_from_W @ T_W_from_Csm

    # Resolve the system-model rotational velocity (course frame) and rotate
    # it into the wind frame for the Williams force balance on each tether
    # node. In a steady straight-line state this is zero; for turning states
    # (downloop, figure-8) it gives a nonzero omega that the tether needs to
    # account for via the per-node `v_n = omega x r_n` and `a_n = omega x v_n`
    # terms.
    if hasattr(system_model, "velocity_rotation_course_frame"):
        omega_course = _as_numeric_3vector(
            system_model, system_model.velocity_rotation_course_frame
        )
        omega_wind = T_Wind_from_Csm @ omega_course
    else:
        omega_wind = np.zeros(3)
    omega_wind_dm = ca.DM(omega_wind)

    # --- Tether force at the kite, per model. -----------------------------
    # ``williams``: symbolic shape in "collapsed" mode -- the kite-end tether
    # vector is supplied as a parameter (the trim resultant), so the 3-D force
    # balance is baked in and only ground closure remains. ``rigid_lumped``:
    # the ROM's RigidLumpedTether, evaluated numerically at the seeded state.
    coupled_fun = None
    rl_tether = None
    if tether_model == "williams":
        _ktv = ca.MX.sym("kite_tension_vector", 3)
        _sh = tether.tether_shape_symbolic(
            env=system_model,
            r_kite=tether.r_kite_sym,
            kite_tension_vector=_ktv,
            tether_length=tether.tether_length,
            omega=omega_wind_dm,
        )
        coupled_fun = ca.Function(
            "williams_collapsed",
            [tether.tether_length, ca.vertcat(tether.r_kite_sym, _ktv)],
            [_sh["ground_position"], _sh["positions"], _sh["tensions"]],
            ["length", "p"],
            ["ground", "positions", "tensions"],
        )
    else:  # rigid_lumped
        from awetrim.system.tether import RigidLumpedTether

        rl_tether = RigidLumpedTether(
            diameter=tether.diameter_tether, density=tether.density_tether
        )

    # Kite position in the wind frame: spherical (azimuth, elevation, distance)
    # in the world frame, then rotate by -direction_wind about +z.
    r_kite_world = distance_radial * np.array(
        [
            np.cos(angle_elev) * np.cos(angle_az),
            np.cos(angle_elev) * np.sin(angle_az),
            np.sin(angle_elev),
        ],
        dtype=float,
    )
    r_kite_wind = T_Wind_from_W @ r_kite_world
    wind_z0 = getattr(wind, "z0", 0.07)
    if (
        wind is not None
        and wind.wind_model == "logarithmic"
        and r_kite_wind[2] <= float(wind_z0)
    ):
        raise ValueError(
            f"Williams tether (log-law wind) needs the kite above the wind "
            f"roughness height z0={wind_z0:.4g} m, but r_kite_wind[z]"
            f"={r_kite_wind[2]:.4g} m (angle_elevation="
            f"{np.rad2deg(angle_elev):.2f} deg, distance_radial="
            f"{distance_radial:.2f} m). Either raise the elevation, or set "
            "the wind model to 'uniform'."
        )

    # --- Trim residual closure (mirrors solve_vsm_quasi_steady_trim). ---
    working_body = copy.deepcopy(body_aero)
    baseline_sections, baseline_spanwise = _baseline_geometry(working_body)
    projected_area_cache: dict[str, float] = {}
    trim_payload_cache: dict[bytes, dict[str, Any]] = {}

    def _compute_trim_payload(x: np.ndarray) -> dict[str, Any]:
        speed_tangential, roll_deg, pitch_deg, yaw_deg, course_rate_body = x

        _set_body_attitude_from_baseline(
            working_body,
            baseline_sections=baseline_sections,
            baseline_spanwise=baseline_spanwise,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            axes=axes,
            reference_point=reference_point,
        )

        _set_course_rate_body(system_model, course_rate_body)
        system_model.speed_tangential = speed_tangential

        accel_course = _acceleration_course_body(system_model)
        inertial_force_wing = -mass_wing_value * _as_3vector(
            transformation_c_from_vsm @ accel_course
        )
        inertial_force_kcu = -mass_kcu * _as_3vector(
            transformation_c_from_vsm @ accel_course
        )
        gravity_force_total = _as_3vector(
            transformation_c_from_vsm @ _force_gravity(system_model)
        )

        va = _as_numeric_3vector(
            system_model,
            transformation_c_from_vsm @ system_model.velocity_apparent_wind,
        )
        umag = float(np.linalg.norm(va))
        if umag <= 0.0:
            raise ValueError("Apparent wind magnitude must be positive.")
        aoa_deg = float(np.rad2deg(np.arctan2(va[2], va[0])))
        beta_deg = float(np.rad2deg(np.arctan2(va[1], np.hypot(va[0], va[2]))))

        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_deg,
            side_slip=beta_deg,
            body_rates=course_rate_body,
            body_axis=-axes.radial,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )
        res = solver.solve(working_body)

        total_aero_force = np.array(
            [float(res.get(k, np.nan)) for k in ("Fx", "Fy", "Fz")],
            dtype=float,
        )
        cmx = float(res.get("cmx", np.nan))
        cmy = float(res.get("cmy", np.nan))
        cmz = float(res.get("cmz", np.nan))

        if "projected_area" not in projected_area_cache:
            projected_area_cache["projected_area"] = float(
                working_body.wings[0].compute_projected_area()
            )
            projected_area_cache["max_chord"] = max(
                float(panel.chord) for panel in working_body.panels
            )
        projected_area = projected_area_cache["projected_area"]
        max_chord = projected_area_cache["max_chord"]
        q_inf = 0.5 * float(solver.rho) * umag**2
        denom_m = q_inf * projected_area * max_chord
        denom_f = q_inf * projected_area

        moment_vec = np.cross(
            center_of_gravity - reference_point,
            inertial_force_wing + inertial_force_kcu,
        )
        if include_gravity:
            moment_vec += np.cross(
                center_of_gravity - reference_point, gravity_force_total
            )
        dcm = moment_vec / denom_m
        cmx += dcm[0]
        cmy += dcm[1]
        cmz += dcm[2]

        net_force = (
            total_aero_force
            + inertial_force_wing
            + inertial_force_kcu
            + (gravity_force_total if include_gravity else 0.0)
        )
        cfx = float(np.dot(net_force, axes.course) / denom_f)
        cfy = float(np.dot(net_force, axes.normal) / denom_f)

        # Rigid-lumped off-radial tether force (drag + half-weight) at the
        # seeded state, transformed into the trim (VSM) frame like the inertial
        # and gravity terms. The radial tension is added separately as the free
        # unknown in the residual.
        f_tether_offrad_vsm = None
        if rl_tether is not None:
            drag_c = _as_numeric_3vector(
                system_model, rl_tether.drag_tether_at_kite_for(system_model)
            )
            grav_c = _as_numeric_3vector(
                system_model,
                rl_tether.force_gravity_tether_at_kite_for(system_model),
            )
            f_tether_offrad_vsm = _as_3vector(
                transformation_c_from_vsm @ (drag_c + grav_c)
            )

        return {
            "trim_res": np.array([cmx, cmy, cmz, cfx, cfy], dtype=float),
            "force_kite_resultant": net_force,
            "total_aero_force": total_aero_force,
            "va": va,
            "umag": umag,
            "res": res,
            "aoa_deg": aoa_deg,
            "beta_deg": beta_deg,
            "denom_f": denom_f,
            "f_tether_offrad_vsm": f_tether_offrad_vsm,
        }

    def _trim_payload(x: np.ndarray) -> dict[str, Any]:
        # Cache the (expensive) trim VSM solve on the 5-vector trim state.
        # ``least_squares`` builds its Jacobian by finite-differencing all 8
        # unknowns, but the 3 Williams perturbations leave the trim state
        # ``x[:5]`` untouched and would otherwise re-run an identical VSM solve
        # (~3 of every 8 Jacobian solves wasted). The cache returns the same
        # residual bytes the finite difference would have produced, so the
        # outer solve is numerically unchanged. Keyed on exact bytes (only true
        # repeats hit); capped to bound memory. NOTE: a cache hit does not
        # re-apply the attitude to ``working_body`` -- the caller re-runs
        # ``_compute_trim_payload`` once at the optimum before returning the
        # body, so its geometry always reflects the final state.
        key = np.asarray(x, dtype=float).tobytes()
        cached = trim_payload_cache.get(key)
        if cached is not None:
            return cached
        payload = _compute_trim_payload(x)
        if len(trim_payload_cache) > 128:
            trim_payload_cache.clear()
        trim_payload_cache[key] = payload
        return payload

    T_VSM_from_Wind = T_Wind_from_VSM.T

    def joint_residual(x: np.ndarray) -> np.ndarray:
        x_trim = np.asarray(x[:5], dtype=float)
        payload = _trim_payload(x_trim)
        net_force_vsm = payload["force_kite_resultant"]

        if tether_model == "rigid_lumped":
            # 3-D force balance: aero+inertial+gravity + [lumped off-radial drag
            # + half-weight] + radial tension. No ground closure.
            tension = float(x[5])
            f_tether = payload["f_tether_offrad_vsm"] - tension * axes.radial
            total_force = net_force_vsm + f_tether
            cf = np.array(
                [
                    np.dot(total_force, axes.course),
                    np.dot(total_force, axes.normal),
                    np.dot(total_force, axes.radial),
                ],
                dtype=float,
            ) / payload["denom_f"]
            return np.concatenate([payload["trim_res"][:3], cf])

        # williams (collapsed): kite-end tether vector = trim resultant, so the
        # 3-D force balance holds by construction; only moments and the tether
        # ground closure remain. Ground residual normalised by distance_radial
        # so it is comparable in magnitude to the dimensionless moments.
        F_kite_wind = T_Wind_from_VSM @ net_force_vsm
        p = np.concatenate([r_kite_wind, F_kite_wind])
        ground = np.asarray(
            coupled_fun(length=float(x[5]), p=p)["ground"]
        ).reshape(3)
        return np.concatenate([payload["trim_res"][:3], ground / distance_radial])

    opt = least_squares(
        joint_residual,
        x0,
        jac="2-point",
        bounds=(lb, ub),
        max_nfev=max_nfev,
    )
    # Per-solve diagnostics at DEBUG level: a wind-window sweep runs thousands of
    # these, so printing them unconditionally floods the console (and, under a
    # process pool, interleaves 8 workers' output and adds real stdout I/O cost).
    # Recover them with ``logging.getLogger().setLevel(logging.DEBUG)``.
    logging.debug(
        "[williams-trim] status=%s  nfev=%s  cost=%.3e  optimality=%.3e",
        opt.status,
        opt.nfev,
        opt.cost,
        opt.optimality,
    )
    logging.debug("[williams-trim] message: %s", opt.message)
    # active_mask: 0=interior, -1=at lb, +1=at ub
    logging.debug("[williams-trim] active_mask: %s", opt.active_mask)
    logging.debug("[williams-trim] x*: trim=%s  williams=%s", opt.x[:5], opt.x[5:])

    res_at_opt = joint_residual(opt.x)
    cm_res = res_at_opt[:3]
    ground_res = (
        np.zeros(3)
        if tether_model == "rigid_lumped"
        else res_at_opt[-3:] * distance_radial
    )
    logging.debug("[williams-trim] cm_res [cmx cmy cmz] = %s", cm_res)
    logging.debug("[williams-trim] ground_res   [gx gy gz] (m)        = %s", ground_res)
    logging.debug(
        "[williams-trim] ||ground_res|| = %.4e m  (distance_radial = %.2f m)",
        float(np.linalg.norm(ground_res)),
        distance_radial,
    )

    # Fresh (uncached) evaluation at the optimum so ``working_body`` ends in the
    # trimmed attitude regardless of what the cache last computed.
    payload = _compute_trim_payload(opt.x[:5])
    F_kite_vsm = payload["force_kite_resultant"]
    F_kite_wind = T_Wind_from_VSM @ F_kite_vsm

    physical_success = bool(all(abs(float(c)) < moment_tolerance for c in cm_res))

    if tether_model == "rigid_lumped":
        tension_opt = float(opt.x[5])
        f_tether_vsm = payload["f_tether_offrad_vsm"] - tension_opt * axes.radial
        tether_force = float(np.linalg.norm(f_tether_vsm))
        positions = np.zeros((0, 3))
        tensions = np.zeros((0, 3))
        elev_last = az_last = tether_length = float("nan")
        cfx, cfy, cfz = float(res_at_opt[3]), float(res_at_opt[4]), float(res_at_opt[5])
    else:  # williams (collapsed)
        tether_length = float(opt.x[5])
        _out = coupled_fun(
            length=tether_length, p=np.concatenate([r_kite_wind, F_kite_wind])
        )
        positions = np.asarray(_out["positions"])
        tensions = np.asarray(_out["tensions"])
        _dir = F_kite_wind / (float(np.linalg.norm(F_kite_wind)) + 1e-12)
        elev_last = float(np.arcsin(np.clip(_dir[2], -1.0, 1.0)))
        az_last = float(np.arctan2(_dir[1], _dir[0]))
        tether_force = float(np.linalg.norm(F_kite_wind))
        cfx = cfy = cfz = 0.0  # force balance baked in

    result: dict[str, Any] = {
        "opt_x": np.asarray(opt.x[:5], dtype=float),
        "cm": np.asarray(cm_res, dtype=float),
        "cfx": float(cfx),
        "cfy": float(cfy),
        "cfz": float(cfz),
        "tether_model": tether_model,
        "success": bool(opt.success),
        "success_physical": physical_success,
        # Chord-referenced aerodynamic angles, same convention as
        # solve_vsm_quasi_steady_trim: apparent wind relative to the mid-span
        # centre chord, falling back to the course-frame inflow angle when the
        # VSM result does not expose the centre-chord values.
        "aoa_deg": float(
            payload["res"].get("alpha_center_chord_deg", payload["aoa_deg"])
        ),
        "aoa_course_deg": payload["aoa_deg"],
        "side_slip_deg": float(
            payload["res"].get("beta_center_chord_deg", payload["beta_deg"])
        ),
        "side_slip_course_deg": payload["beta_deg"],
        "aero_roll_deg": float("nan"),
        "cl": payload["res"].get("cl"),
        "cd": payload["res"].get("cd"),
        "tether_force": float(tether_force),
        "va_vel_world": payload["va"],
        "Umag": payload["umag"],
        "total_aero_force_vec": payload["total_aero_force"],
        "force_kite_resultant": F_kite_wind,
        "force_kite_resultant_vsm": F_kite_vsm,
        "r_kite": r_kite_wind,
        "r_kite_world": r_kite_world,
        "williams_x": np.asarray(opt.x[5:], dtype=float),
        "williams_elevation_last_deg": float(np.rad2deg(elev_last)),
        "williams_azimuth_last_deg": float(np.rad2deg(az_last)),
        "williams_tether_length": float(tether_length),
        "williams_ground_residual": ground_res,
        "williams_positions": positions,
        "williams_tensions": tensions,
        "optimizer": opt,
    }
    return result, working_body


#: Canonical full-state ordering used by the stability linearisation.
#:
#:   index | name  | meaning
#:   ------|-------|--------------------------------
#:     0   | u     | body x velocity (course axis)
#:     1   | v     | body y velocity (normal axis)
#:     2   | w     | body z velocity (radial axis)  -- vertical speed
#:     3   | z     | radial position perturbation along course-frame z/radial axis
#:     4   | phi   | roll angle
#:     5   | theta | pitch angle
#:     6   | psi   | yaw angle
#:     7   | p     | body roll rate
#:     8   | q     | body pitch rate
#:     9   | r     | body yaw rate
#: Canonical free rigid-body states. The 6-DOF body has 12 states; ``x``, ``y``
#: and the lateral velocity ``v`` are held fixed, leaving these 9 free states.
ALL_STATE_NAMES: tuple[str, ...] = (
    "u",
    "w",
    "z",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
)

#: Subsets used by the default decoupled (long + lat) split.
LONG_STATES: tuple[str, ...] = ("u", "w", "z", "theta", "q")
LAT_STATES: tuple[str, ...] = ("phi", "psi", "p", "r")

#: Default selection — current behaviour, *no* vertical speed `w`.
DEFAULT_STATES: tuple[str, ...] = (
    "u",
    "theta",
    "q",
    "phi",
    "psi",
    "p",
    "r",
)

#: Row index of each force/moment output in J_full and central_diff_col output.
#:   0..2 = F_course, F_normal, F_radial
#:   3..5 = M_course, M_normal, M_radial
_FORCE_OUTPUT_ROW = {"u": 0, "v": 1, "w": 2}
_MOMENT_OUTPUT_ROW = {"p": 3, "q": 4, "r": 5}
_KINEMATIC_RATE = {"z": "w", "phi": "p", "theta": "q", "psi": "r"}

#: Index of each body-rate state in the (course, normal, radial) principal-axis
#: ordering used by the gyroscopic coupling term (p=course, q=normal, r=radial).
_RATE_AXIS_INDEX = {"p": 0, "q": 1, "r": 2}


def _state_indices(states: Sequence[str]) -> list[int]:
    """Map state names to their column index in J_full / A_full."""
    full_idx = {name: idx for idx, name in enumerate(ALL_STATE_NAMES)}
    try:
        return [full_idx[s] for s in states]
    except KeyError as exc:
        raise ValueError(
            f"Unknown stability state {exc.args[0]!r}. "
            f"Valid names: {list(ALL_STATE_NAMES)}"
        ) from None


def _gyroscopic_rate_coupling(
    omega_c_body: np.ndarray,
    inertia: np.ndarray,
) -> np.ndarray:
    """Linearised gyroscopic coupling of the body rates from course-frame transport.

    The CG moment equation of the fast subsystem retains the transport term of
    the course frame (Cayon & Schmehl, Eq. ``moment_eom_cg``)::

        I_cg @ omega_dot + Omega_C x (I_cg @ omega) = M_ext

    so ``omega_dot`` gains ``-I_cg^{-1} (Omega_C x I_cg @ omega)``. With
    ``omega = Omega_C + omega_rel`` and ``Omega_C`` fixed by the trajectory, the
    Jacobian of this term with respect to the body-rate states ``(p, q, r)`` is
    the constant matrix ``-I_cg^{-1} [Omega_C]_x I_cg``, returned in the
    ``(course, normal, radial) = (p, q, r)`` axis basis.

    Parameters
    ----------
    omega_c_body
        Course-frame transport rate ``Omega_C`` in the ``(course, normal,
        radial)`` axis basis.
    inertia
        Full 3x3 inertia tensor ``I_cg`` in the same axis basis (need not be
        diagonal — off-diagonal products of inertia are honoured).
    """
    inertia = np.asarray(inertia, dtype=float)
    wx, wy, wz = (float(v) for v in np.asarray(omega_c_body, dtype=float))
    skew = np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]],
        dtype=float,
    )
    # -I^{-1} [Omega_C]_x I
    return -np.linalg.solve(inertia, skew @ inertia)


def _course_transport_rate_axes(
    system_model: AWETrimSystemModel | None,
    axes: AxisDefinition,
    course_rate: float,
    speed_tangential: float,
    *,
    full: bool,
    transformation_c_from_vsm: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Baseline course-frame transport rate ``Omega_C`` for the stability model.

    Returns ``(omega_c_axes, omega_c_world, is_full)``:

    - ``omega_c_axes`` — ``Omega_C`` resolved in the ``(course, normal, radial)``
      principal-axis basis, used by the gyroscopic term in ``_build_state_space``.
    - ``omega_c_world`` — the same vector in the world/VSM basis, used as the
      aerodynamic body-rate baseline in ``eval_force_moment``.

    Reduced form (turn-rate-law reduction, Eq. before ``inertial_reduced``)::

        Omega_C = -course_rate * e_radial            # radial component only

    Full form (``Kinematics.velocity_rotation_course_frame``), matching the
    paper's ``Omega_C`` with the radial component
    ``-chi_dot_turn = (v_tau/r) tan(beta) sin(chi) - chi_dot`` and the
    great-circle normal component ``v_tau/r``::

        Omega_C = [0, v_tau/r, (v_tau/r) tan(beta) sin(chi) - chi_dot]  (course)

    The course-frame vector is mapped to the world/VSM frame with the same
    ``transformation_c_from_vsm`` the trim uses for every course-frame quantity,
    so the flipped course/normal axes are handled consistently (the radial
    component, and therefore the reduced form, is invariant under the flip). The
    full form needs ``system_model``; without it the reduced form is returned.
    """
    R_body = np.array([axes.course, axes.normal, axes.radial], dtype=float)
    reduced_axes = np.array([0.0, 0.0, -float(course_rate)], dtype=float)
    if not full or system_model is None:
        return reduced_axes, R_body.T @ reduced_axes, False
    _set_course_rate_body(system_model, float(course_rate))
    system_model.speed_tangential = float(speed_tangential)
    omega_c_course = np.asarray(
        _as_numeric_3vector(system_model, system_model.velocity_rotation_course_frame),
        dtype=float,
    ).reshape(3)
    omega_c_world = np.asarray(transformation_c_from_vsm, dtype=float) @ omega_c_course
    return R_body @ omega_c_world, omega_c_world, True


def _build_state_space(
    J_full: np.ndarray,
    states: Sequence[str],
    *,
    mass: float,
    inertia: np.ndarray,
    omega_c_body: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (J_sub, A) for a chosen subset of states.

    J_sub keeps all 6 force/moment rows so it can be inspected even when the
    caller drops some states. A has one row per state — dynamics for velocity
    and rate states, kinematics (phi_dot=p etc.) for attitude states. Kinematic
    rows whose paired rate is not in `states` collapse to a zero row.

    ``inertia`` is the full 3x3 tensor ``I`` in the ``(course, normal, radial)``
    axis basis. The body-rate rows solve ``I omega_dot = M`` as a coupled block,
    ``omega_dot = I^{-1} M``, so any off-diagonal products of inertia (e.g. from
    a tensor rotated into the stability frame at the trim attitude) couple the
    moment channels. A per-row scalar division is only recovered when ``I`` is
    diagonal.

    When ``omega_c_body`` (the course-frame transport rate ``Omega_C`` in the
    same axis basis) is given, the rate rows also carry the linearised
    gyroscopic coupling ``-I^{-1} [Omega_C]_x I`` between the body rates, so the
    moment dynamics implement ``I omega_dot + Omega_C x I omega = M`` rather than
    the torque-free ``I omega_dot = M``.
    """
    cols = _state_indices(states)
    J_sub = J_full[:, cols]

    inertia = np.asarray(inertia, dtype=float)
    # omega_dot from the moment channels: I^{-1} @ [M_course; M_normal; M_radial].
    moment_rows = J_sub[
        [_MOMENT_OUTPUT_ROW["p"], _MOMENT_OUTPUT_ROW["q"], _MOMENT_OUTPUT_ROW["r"]], :
    ]
    rate_accel_from_moment = np.linalg.solve(inertia, moment_rows)  # (3, n)
    gyro = (
        _gyroscopic_rate_coupling(omega_c_body, inertia)
        if omega_c_body is not None
        else None
    )
    n = len(states)
    A = np.zeros((n, n))
    for i, s in enumerate(states):
        if s in _FORCE_OUTPUT_ROW:
            A[i, :] = J_sub[_FORCE_OUTPUT_ROW[s], :] / mass
        elif s in _MOMENT_OUTPUT_ROW:
            ax = _RATE_AXIS_INDEX[s]
            A[i, :] = rate_accel_from_moment[ax, :]
            if gyro is not None:
                for j, s2 in enumerate(states):
                    if s2 in _RATE_AXIS_INDEX:
                        A[i, j] += gyro[ax, _RATE_AXIS_INDEX[s2]]
        elif s in _KINEMATIC_RATE:
            rate = _KINEMATIC_RATE[s]
            if rate in states:
                A[i, states.index(rate)] = 1.0
    return J_sub, A


def _eig_block(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if A.size == 0:
        return np.zeros(0, dtype=complex), np.zeros((0, 0), dtype=complex)
    return np.linalg.eig(A)


def _timescales_from_eigs(eigvals: np.ndarray) -> np.ndarray:
    real_parts = np.real(eigvals)
    abs_re = np.abs(real_parts)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(abs_re > 1e-12, 1.0 / abs_re, np.inf)


def compute_vsm_trim_stability_derivatives(
    body_aero: VsmBodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    x_trim: np.ndarray,
    trim_result: Mapping[str, Any],
    *,
    solver: VsmSolver | None = None,
    system_model: AWETrimSystemModel | None = None,
    axes: AxisDefinition = DEFAULT_AXES,
    mass: float = 15.0,
    inertia_xx: float = 100.0,
    inertia_yy: float = 19.43,
    inertia_zz: float = 100.0,
    distance_radial: float | None = None,
    eps_vel: float = 0.1,
    eps_angle_deg: float = 0.5,
    eps_rate: float = 0.01,
    eps_position: float = 0.5,
    states: Sequence[str] | None = None,
    coupled: bool = False,
    full_omega_c: bool = True,
    rotate_inertia_by_trim: bool = True,
) -> dict[str, Any]:
    """Compute aerodynamic stability derivatives around a VSM trim state.

    Parameters
    ----------
    states
        Subset of :data:`ALL_STATE_NAMES` to use for the *selected* state-space
        block returned alongside the full coupled and default decoupled blocks.
        When ``None`` (default), the function only returns the full and default
        decoupled blocks — preserving the historical API.
    coupled
        If ``True`` and ``states`` is given, the selected states are assembled
        into a single coupled A matrix. If ``False``, the selection is split
        into a longitudinal sub-block (states in :data:`LONG_STATES`) and a
        lateral sub-block (states in :data:`LAT_STATES`).
    full_omega_c
        If ``True`` (default), the course-frame transport rate ``Omega_C`` that
        drives both the aerodynamic body-rate and the gyroscopic moment term is
        the full ``[0, v_tau/r, (v_tau/r) tan(beta) sin(chi) - chi_dot]`` taken
        from ``system_model`` (requires it). If ``False`` — or when no
        ``system_model`` is supplied — the radial-only reduction
        ``-chi_dot * e_radial`` is used, matching the closed-form turn-rate law.
        The result records the resolved choice under ``omega_c_model``.
    rotate_inertia_by_trim
        If ``True`` (default), the principal moments ``diag(I_xx, I_yy, I_zz)``
        are rotated into the stability frame by the trim attitude,
        ``I_stab = R(roll0, pitch0, yaw0) diag(I) R^T``, so the rotational
        dynamics use the full tensor (products of inertia and all) as a coupled
        ``omega_dot = I_stab^{-1} M`` block. This is correct when the stability
        axes are the (space-fixed) course axes and the kite is tilted from them
        by the trim attitude. Set ``False`` when the stability axes are the
        body-fixed principal axes (``--stability-frame body``), where the
        inertia is already diagonal and no rotation applies.

    Always-present outputs
    ----------------------
    ``J_full`` (6, 9), ``A_full`` (9, 9), ``eig_full``, ``vec_full``,
    ``Tfast_full``, ``stable_full``, ``state_names_full``, ``output_names``.

    Default decoupled outputs (always present, shape preserved for back-compat)
    --------------------------------------------------------------------------
    ``J_long`` (3, 3), ``J_lat`` (3, 4), ``A_long`` (3, 3), ``A_lat`` (4, 4),
    ``eig_long``, ``eig_lat``, ``vec_long``, ``vec_lat``, ``Tfast_long``,
    ``Tfast_lat``, ``stable_long``, ``stable_lat``.

    Selection outputs (only present when ``states`` is given or ``coupled``)
    -----------------------------------------------------------------------
    Coupled selection (``coupled=True``):
        ``J_selected``, ``A_selected``, ``eig_selected``, ``vec_selected``,
        ``Tfast_selected``, ``stable_selected``, ``states_selected``.
    Decoupled selection (``coupled=False`` with explicit ``states``):
        Additionally ``J_selected_long``, ``A_selected_long``, ...,
        ``J_selected_lat``, ``A_selected_lat``, ... plus the partitioned state
        name lists.
    """

    center_of_gravity = _as_3vector(center_of_gravity)
    reference_point = _as_3vector(reference_point)
    x_trim = _as_5vector(x_trim, "x_trim")
    if solver is None:
        solver = _default_vsm_solver(reference_point)

    speed_tangential, roll0, pitch0, yaw0, course_rate0 = x_trim
    va_world = trim_result.get("va_vel_world")
    if va_world is None:
        va_world = trim_result.get("va")
    if va_world is None:
        wind_world = trim_result.get("wind_vel_world")
        kite_world = trim_result.get("kite_vel_world")
        if wind_world is not None and kite_world is not None:
            va_world = _as_3vector(wind_world) - _as_3vector(kite_world)
    if va_world is None:
        raise KeyError(
            "trim_result is missing apparent-wind data; expected 'va_vel_world', "
            "'va', or both 'wind_vel_world' and 'kite_vel_world'."
        )
    va_trim = _as_3vector(va_world)
    f_tether = np.array([0.0, 0.0, -float(trim_result["tether_force"])], dtype=float)
    r_arm = reference_point - center_of_gravity
    moment_tether_at_cg = np.cross(r_arm, f_tether)
    distance_radial_trim = (
        float(distance_radial)
        if distance_radial is not None and float(distance_radial) > 0.0
        else None
    )

    working_body = copy.deepcopy(body_aero)
    baseline_sections, baseline_spanwise = _baseline_geometry(working_body)
    projected_area = float(body_aero.wings[0].compute_projected_area())
    max_chord = max(float(panel.chord) for panel in body_aero.panels)

    def _make_williams_fixed_length_solver():
        if system_model is None or "williams_tether_length" not in trim_result:
            return None
        try:
            from awetrim.system.williams_tether import WilliamsTether
            from awetrim.utils.reference_frames import transformation_Wind_from_C
        except ImportError:
            return None
        tether = getattr(system_model, "tether", None)
        # Match by class name too: ``isinstance`` can miss the WilliamsTether
        # when ``awetrim`` is importable via two paths (giving two distinct
        # class objects), which would silently drop the radial dependency.
        if not (
            isinstance(tether, WilliamsTether)
            or type(tether).__name__ == "WilliamsTether"
        ):
            return None

        _set_course_rate_body(system_model, float(course_rate0))
        system_model.speed_tangential = float(speed_tangential)

        angle_az = float(_numeric_value_for_symbol(system_model, "angle_azimuth"))
        angle_elev = float(_numeric_value_for_symbol(system_model, "angle_elevation"))
        angle_course = float(_numeric_value_for_symbol(system_model, "angle_course"))
        direction_wind = float(
            getattr(getattr(system_model, "wind", None), "direction_wind", 0.0)
        )
        T_wind_from_course = np.asarray(
            ca.DM(
                transformation_Wind_from_C(
                    angle_az, angle_elev, angle_course, direction_wind
                )
            ).full(),
            dtype=float,
        )
        if hasattr(system_model, "velocity_rotation_course_frame"):
            omega_course = _as_numeric_3vector(
                system_model, system_model.velocity_rotation_course_frame
            )
            omega_wind = T_wind_from_course @ omega_course
        else:
            omega_wind = np.zeros(3)

        tension_sym = ca.MX.sym("tension_tether_kite_fixed_length")
        elevation_sym = ca.MX.sym("elevation_last_element_fixed_length")
        azimuth_sym = ca.MX.sym("azimuth_last_element_fixed_length")
        r_kite_sym = ca.MX.sym("r_kite_fixed_length", 3)
        tether_length = float(trim_result["williams_tether_length"])
        shape = tether.tether_shape_symbolic(
            env=system_model,
            r_kite=r_kite_sym,
            tension_kite=tension_sym,
            omega=ca.DM(omega_wind),
            tether_length=tether_length,
            elevation_last=elevation_sym,
            azimuth_last=azimuth_sym,
        )
        x_sym = ca.vertcat(tension_sym, elevation_sym, azimuth_sym)
        residual_fun = ca.Function(
            "williams_fixed_length_residual",
            [x_sym, r_kite_sym],
            [shape["ground_position"]],
            ["x", "r_kite"],
            ["residual"],
        )
        jac_fun = ca.Function(
            "williams_fixed_length_residual_jac",
            [x_sym, r_kite_sym],
            [ca.jacobian(shape["ground_position"], x_sym)],
            ["x", "r_kite"],
            ["jac"],
        )
        force_fun = ca.Function(
            "williams_fixed_length_force_kite",
            [x_sym, r_kite_sym],
            [shape["tether_force_kite"]],
            ["x", "r_kite"],
            ["force_kite"],
        )

        x0 = np.array(
            [
                float(trim_result["tether_force"]),
                np.deg2rad(float(trim_result.get("williams_elevation_last_deg", 0.0))),
                np.deg2rad(float(trim_result.get("williams_azimuth_last_deg", 0.0))),
            ],
            dtype=float,
        )
        r0_wind = (
            _as_3vector(trim_result["r_kite"])
            if "r_kite" in trim_result
            else T_wind_from_course @ np.array([0.0, 0.0, distance_radial_trim or 0.0])
        )

        def solve_force(radial_offset: float) -> np.ndarray:
            r_kite = r0_wind + T_wind_from_course @ (
                float(radial_offset) * DEFAULT_AXES.radial
            )

            def residual(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    residual_fun(x=np.asarray(x, dtype=float), r_kite=r_kite)[
                        "residual"
                    ]
                ).reshape(-1)

            def jac(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    jac_fun(x=np.asarray(x, dtype=float), r_kite=r_kite)["jac"]
                )

            sol = least_squares(
                residual,
                x0,
                jac=jac,
                bounds=(
                    [0.0, -np.pi / 2 + 1e-3, -2.0 * np.pi],
                    [np.inf, np.pi / 2 - 1e-3, 2.0 * np.pi],
                ),
                max_nfev=200,
            )
            # ``least_squares`` reports ``success=False`` when it exhausts
            # ``max_nfev`` even if it has already converged. The ground-position
            # residual (metres) is the physically meaningful convergence check,
            # so accept the solve whenever that residual is below tolerance.
            if np.linalg.norm(sol.fun) > 1e-3:
                raise RuntimeError(
                    "Williams fixed-length radial perturbation solve failed: "
                    f"{sol.message}; residual={sol.fun}"
                )
            force_wind = np.asarray(
                force_fun(x=sol.x, r_kite=r_kite)["force_kite"]
            ).reshape(3)
            return T_wind_from_course.T @ force_wind

        return solve_force

    williams_fixed_length_force = _make_williams_fixed_length_solver()

    # Baseline course-frame transport rate Omega_C (full vs radial-only). Used
    # both as the steady aerodynamic body-rate seen by the VSM solve
    # (omega_c_world) and as the Omega_C of the gyroscopic moment term in the
    # state-space (omega_c_axes). Computing it once keeps the aero and the
    # inertial coupling on the *same* Omega_C, so the linearisation point stays
    # a consistent trim state.
    omega_c_axes, omega_c_world, omega_c_is_full = _course_transport_rate_axes(
        system_model,
        axes,
        float(course_rate0),
        float(speed_tangential),
        full=full_omega_c,
    )

    # Inertia tensor in the stability axis basis. In the course frame the kite's
    # principal axes are tilted from the (space-fixed) stability axes by the trim
    # attitude, so the principal moments are rotated in with the same rotation
    # the aerodynamics uses: I_stab = R diag(I) R^T. This makes the moment rows a
    # coupled I^{-1} M block and lets the gyroscopic term carry the resulting
    # products of inertia. In the body-fixed principal-axis frame the tensor is
    # already diagonal, so the rotation is skipped.
    inertia_principal = np.diag(
        [float(inertia_xx), float(inertia_yy), float(inertia_zz)]
    )
    if rotate_inertia_by_trim:
        _R_attitude = _compose_attitude_rotation(
            roll_deg=float(roll0),
            pitch_deg=float(pitch0),
            yaw_deg=float(yaw0),
            axes=axes,
        )
        inertia_stability = _R_attitude @ inertia_principal @ _R_attitude.T
    else:
        inertia_stability = inertia_principal

    warned_williams_force = False

    def tether_force_for(radial_offset: float) -> np.ndarray:
        nonlocal warned_williams_force
        if williams_fixed_length_force is None:
            return f_tether.copy()
        try:
            return williams_fixed_length_force(radial_offset)
        except RuntimeError as exc:
            if not warned_williams_force:
                print(
                    "Warning: Williams fixed-length radial perturbation failed; "
                    "falling back to baseline tether force."
                )
                print(f"  reason: {exc}")
                warned_williams_force = True
            return f_tether.copy()

    def eval_force_moment(
        delta_v: np.ndarray,
        omega_perturb: np.ndarray,
        radial_position_offset: float = 0.0,
        delta_roll_deg: float = 0.0,
        delta_pitch_deg: float = 0.0,
        delta_yaw_deg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        _set_body_attitude_from_baseline(
            working_body,
            baseline_sections=baseline_sections,
            baseline_spanwise=baseline_spanwise,
            roll_deg=roll0 + delta_roll_deg,
            pitch_deg=pitch0 + delta_pitch_deg,
            yaw_deg=yaw0 + delta_yaw_deg,
            axes=axes,
            reference_point=reference_point,
        )
        va_pert = va_trim - delta_v
        umag = np.linalg.norm(va_pert)
        aoa_deg = np.rad2deg(np.arctan2(va_pert[2], va_pert[0]))
        beta_deg = np.rad2deg(np.arctan2(va_pert[1], np.hypot(va_pert[0], va_pert[2])))
        omega_total = omega_c_world + omega_perturb
        omega_mag = np.linalg.norm(omega_total)
        omega_axis = omega_total / omega_mag if omega_mag > 1e-12 else axes.radial

        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_deg,
            side_slip=beta_deg,
            body_rates=omega_mag,
            body_axis=omega_axis,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )
        res = solver.solve(working_body)
        f_aero = np.array(
            [
                float(res.get("Fx", 0.0)),
                float(res.get("Fy", 0.0)),
                float(res.get("Fz", 0.0)),
            ],
            dtype=float,
        )
        q_inf = 0.5 * float(solver.rho) * umag**2
        denom = q_inf * projected_area * max_chord if projected_area > 0 else 1.0
        moment_aero_at_ref = (
            np.array(
                [
                    float(res.get("cmx", 0.0)),
                    float(res.get("cmy", 0.0)),
                    float(res.get("cmz", 0.0)),
                ],
                dtype=float,
            )
            * denom
        )

        speed_tangential_eff = float(speed_tangential) + float(
            np.dot(delta_v, axes.course)
        )
        distance_radial_eff = (
            distance_radial_trim + radial_position_offset
            if distance_radial_trim is not None
            else None
        )
        f_tether_eff = tether_force_for(radial_position_offset)
        moment_tether_eff = np.cross(r_arm, f_tether_eff)
        f_inertial = np.zeros(3, dtype=float)
        f_inertial[1] = mass * speed_tangential_eff * float(course_rate0)
        if distance_radial_eff is not None and distance_radial_eff > 0.0:
            f_inertial[2] = mass * speed_tangential_eff**2 / distance_radial_eff

        moment_at_cg = moment_aero_at_ref + np.cross(r_arm, f_aero) + moment_tether_eff
        force_at_cg = f_aero + f_tether_eff + f_inertial
        return force_at_cg, moment_at_cg

    zero3 = np.zeros(3, dtype=float)
    eps_angle_rad = np.deg2rad(eps_angle_deg)

    # Rotation matrix from world frame to body frame (rows = body axes).
    R_body = np.array([axes.course, axes.normal, axes.radial], dtype=float)

    def central_diff_col(
        delta_v: np.ndarray,
        omega_perturb: np.ndarray,
        step: float,
        radial_position_offset: float = 0.0,
        droll: float = 0.0,
        dpitch: float = 0.0,
        dyaw: float = 0.0,
    ) -> np.ndarray:
        force_plus, moment_plus = eval_force_moment(
            delta_v, omega_perturb, radial_position_offset, droll, dpitch, dyaw
        )
        force_minus, moment_minus = eval_force_moment(
            -delta_v, -omega_perturb, -radial_position_offset, -droll, -dpitch, -dyaw
        )
        d_force = (force_plus - force_minus) / (2.0 * step)
        d_moment = (moment_plus - moment_minus) / (2.0 * step)
        # Project world-frame force and moment onto body axes so that outputs
        # are consistent with the body-frame inputs (perturbations along axes.*).
        d_force_body = R_body @ d_force
        d_moment_body = R_body @ d_moment
        return np.array(
            [
                d_force_body[0],
                d_force_body[1],
                d_force_body[2],
                d_moment_body[0],
                d_moment_body[1],
                d_moment_body[2],
            ]
        )

    radial_eps = float(eps_position)
    if (
        williams_fixed_length_force is not None
        and "williams_tether_length" in trim_result
    ):
        r_norm = float(
            np.linalg.norm(_as_3vector(trim_result.get("r_kite", [0.0, 0.0, 0.0])))
        )
        slack = float(trim_result["williams_tether_length"]) - r_norm
        if slack > 1e-8:
            radial_eps = min(radial_eps, 0.25 * slack)
        else:
            radial_eps = min(radial_eps, 1e-5)
        radial_eps = max(radial_eps, 1e-7)

    # Build the full 6×9 numerical Jacobian (rows = body force+moment outputs,
    # cols = state perturbations in canonical ALL_STATE_NAMES order). The lateral
    # velocity ``v`` is held fixed (not a free state), so it is never perturbed.
    columns = {
        "u": central_diff_col(+eps_vel * axes.course, zero3, eps_vel),
        "w": central_diff_col(+eps_vel * axes.radial, zero3, eps_vel),
        "z": central_diff_col(
            zero3, zero3, radial_eps, radial_position_offset=radial_eps
        ),
        "phi": central_diff_col(zero3, zero3, eps_angle_rad, droll=eps_angle_deg),
        "theta": central_diff_col(zero3, zero3, eps_angle_rad, dpitch=eps_angle_deg),
        "psi": central_diff_col(zero3, zero3, eps_angle_rad, dyaw=eps_angle_deg),
        "p": central_diff_col(zero3, eps_rate * axes.course, eps_rate),
        "q": central_diff_col(zero3, eps_rate * axes.normal, eps_rate),
        "r": central_diff_col(zero3, eps_rate * axes.radial, eps_rate),
    }
    J_full = np.column_stack([columns[name] for name in ALL_STATE_NAMES])

    # Course-frame transport rate Omega_C (computed once above, in the rate-row
    # (course, normal, radial) basis). It drives the gyroscopic term
    # Omega_C x I omega of the CG moment equation and is the same Omega_C the
    # aerodynamic body-rate uses in eval_force_moment. With the full form it also
    # carries the great-circle normal component v_tau/r, which couples p <-> r
    # inside the lateral block; the radial-only reduction leaves only p <-> q.
    omega_c_body = omega_c_axes

    _, A_full = _build_state_space(
        J_full,
        ALL_STATE_NAMES,
        mass=mass,
        inertia=inertia_stability,
        omega_c_body=omega_c_body,
    )
    eig_full, vec_full = _eig_block(A_full)

    # ---- Backward-compatible default decoupled blocks -------------------
    # These split longitudinal (u, theta, q) and lateral (phi, psi, p, r) and use
    # the diagonal principal moments (not the trim-rotated tensor I_stab), so they
    # stay the historical decoupled approximation; the full-tensor treatment lives
    # in A_full and the selected/coupled blocks. The gyroscopic term Omega_C x I
    # omega couples p <-> q (straddles the split, dropped here, retained in
    # A_full) and — with the full Omega_C — p <-> r (within the lateral block,
    # added to a_lat below). The radial-only reduction has no p <-> r term.
    # J_long keeps the historical (3, 3) shape: rows = [F_x, F_z, M_y]
    # (i.e. the longitudinal force/moment channels), cols = [u, theta, q].
    long_default_state_idx = _state_indices(["u", "theta", "q"])
    long_out_rows = [0, 2, 4]  # F_course, F_radial, M_normal
    j_long = J_full[np.ix_(long_out_rows, long_default_state_idx)]

    a_long = np.zeros((3, 3))
    a_long[0, :] = j_long[0, :] / mass
    a_long[1, :] = [0.0, 0.0, 1.0]
    a_long[2, :] = j_long[2, :] / inertia_yy

    lat_default_state_idx = _state_indices(["phi", "psi", "p", "r"])
    lat_out_rows = [1, 3, 5]  # F_normal, M_course, M_radial
    j_lat = J_full[np.ix_(lat_out_rows, lat_default_state_idx)]

    a_lat = np.zeros((4, 4))
    a_lat[0, :] = [0.0, 0.0, 1.0, 0.0]
    a_lat[1, :] = [0.0, 0.0, 0.0, 1.0]
    a_lat[2, :] = j_lat[1, :] / inertia_xx
    a_lat[3, :] = j_lat[2, :] / inertia_zz
    # Within-lateral gyroscopic coupling p <-> r from the great-circle normal
    # component of Omega_C (identically zero for the radial-only reduction).
    # Uses the diagonal principal inertia, consistent with the decoupled blocks.
    _gyro_lat = _gyroscopic_rate_coupling(omega_c_axes, inertia_principal)
    a_lat[2, 3] += _gyro_lat[_RATE_AXIS_INDEX["p"], _RATE_AXIS_INDEX["r"]]
    a_lat[3, 2] += _gyro_lat[_RATE_AXIS_INDEX["r"], _RATE_AXIS_INDEX["p"]]

    eig_long, vec_long = _eig_block(a_long)
    eig_lat, vec_lat = _eig_block(a_lat)

    result: dict[str, Any] = {
        # Default (back-compat) decoupled blocks
        "J_long": j_long,
        "J_lat": j_lat,
        "A_long": a_long,
        "A_lat": a_lat,
        "eig_long": eig_long,
        "eig_lat": eig_lat,
        "vec_long": vec_long,
        "vec_lat": vec_lat,
        "Tfast_long": _timescales_from_eigs(eig_long),
        "Tfast_lat": _timescales_from_eigs(eig_lat),
        "stable_long": bool(np.all(np.real(eig_long) < 0.0)),
        "stable_lat": bool(np.all(np.real(eig_lat) < 0.0)),
        # Full 9-state coupled system (always available for inspection)
        "J_full": J_full,
        "A_full": A_full,
        "eig_full": eig_full,
        "vec_full": vec_full,
        "Tfast_full": _timescales_from_eigs(eig_full),
        "stable_full": bool(np.all(np.real(eig_full) < 0.0)),
        "state_names_full": list(ALL_STATE_NAMES),
        "output_names": [
            "F_course",
            "F_normal",
            "F_radial",
            "M_course",
            "M_normal",
            "M_radial",
        ],
        # Tether transfer quantities
        "F_tether": f_tether,
        "M_tether_at_CG": moment_tether_at_cg,
        "tether_radial_position_model": (
            "williams_fixed_length"
            if williams_fixed_length_force is not None
            else "constant_trim_force"
        ),
        "radial_position_state": "z",
        "eps_position": float(eps_position),
        "eps_position_used": float(radial_eps),
        # Course-frame transport rate used for the aero body-rate + gyroscopic term.
        "omega_c_model": "full" if omega_c_is_full else "radial_only",
        "omega_c_axes": np.asarray(omega_c_axes, dtype=float),
        # Inertia tensor used for the rotational dynamics of A_full / A_selected.
        "inertia_stability": np.asarray(inertia_stability, dtype=float),
        "inertia_rotated_by_trim": bool(rotate_inertia_by_trim),
    }

    # ---- User-selected sub-block (custom state set / coupling) ----------
    if states is not None or coupled:
        sel_states = list(states) if states is not None else list(DEFAULT_STATES)
        unknown = [s for s in sel_states if s not in ALL_STATE_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown stability state {unknown[0]!r}. "
                f"Valid names: {list(ALL_STATE_NAMES)}"
            )
        # Deduplicate while preserving order.
        seen: set[str] = set()
        sel_states = [s for s in sel_states if not (s in seen or seen.add(s))]

        if coupled:
            J_sel, A_sel = _build_state_space(
                J_full,
                sel_states,
                mass=mass,
                inertia=inertia_stability,
                omega_c_body=omega_c_body,
            )
            eig_sel, vec_sel = _eig_block(A_sel)
            result.update(
                {
                    "J_selected": J_sel,
                    "A_selected": A_sel,
                    "eig_selected": eig_sel,
                    "vec_selected": vec_sel,
                    "Tfast_selected": _timescales_from_eigs(eig_sel),
                    "stable_selected": (
                        bool(np.all(np.real(eig_sel) < 0.0)) if A_sel.size > 0 else True
                    ),
                    "states_selected": sel_states,
                    "coupled_selected": True,
                }
            )
        else:
            sel_long = [s for s in sel_states if s in LONG_STATES]
            sel_lat = [s for s in sel_states if s in LAT_STATES]
            J_sel_long, A_sel_long = _build_state_space(
                J_full,
                sel_long,
                mass=mass,
                inertia=inertia_stability,
                omega_c_body=omega_c_body,
            )
            J_sel_lat, A_sel_lat = _build_state_space(
                J_full,
                sel_lat,
                mass=mass,
                inertia=inertia_stability,
                omega_c_body=omega_c_body,
            )
            eig_sel_long, vec_sel_long = _eig_block(A_sel_long)
            eig_sel_lat, vec_sel_lat = _eig_block(A_sel_lat)
            result.update(
                {
                    "J_selected_long": J_sel_long,
                    "J_selected_lat": J_sel_lat,
                    "A_selected_long": A_sel_long,
                    "A_selected_lat": A_sel_lat,
                    "eig_selected_long": eig_sel_long,
                    "eig_selected_lat": eig_sel_lat,
                    "vec_selected_long": vec_sel_long,
                    "vec_selected_lat": vec_sel_lat,
                    "Tfast_selected_long": _timescales_from_eigs(eig_sel_long),
                    "Tfast_selected_lat": _timescales_from_eigs(eig_sel_lat),
                    "stable_selected_long": (
                        bool(np.all(np.real(eig_sel_long) < 0.0))
                        if A_sel_long.size > 0
                        else True
                    ),
                    "stable_selected_lat": (
                        bool(np.all(np.real(eig_sel_lat) < 0.0))
                        if A_sel_lat.size > 0
                        else True
                    ),
                    "states_selected": sel_states,
                    "states_selected_long": sel_long,
                    "states_selected_lat": sel_lat,
                    "coupled_selected": False,
                }
            )

    return result


def _as_sequence(value: Sequence[float] | float) -> list[float]:
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def run_vsm_quasi_steady_sweep(
    *,
    build_body: Callable[[dict[str, float]], VsmBodyAerodynamics],
    system_model: AWETrimSystemModel,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    x_guess: np.ndarray,
    principal_axis: str,
    secondary_axis: str,
    sweep_values: Mapping[str, Sequence[float] | float],
    update_system_model: (
        Callable[[AWETrimSystemModel, dict[str, float]], None] | None
    ) = None,
    solver_factory: Callable[[np.ndarray], VsmSolver] | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    transformation_c_from_vsm: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-4,
    return_timing_breakdown: bool = False,
    max_nfev: int | None = None,
) -> list[dict[str, Any]]:
    """Run a warm-started principal/secondary VSM aerodynamic trim sweep."""

    if principal_axis not in sweep_values:
        raise KeyError(f"principal_axis '{principal_axis}' missing from sweep_values")
    if secondary_axis not in sweep_values:
        raise KeyError(f"secondary_axis '{secondary_axis}' missing from sweep_values")

    principal_values = _as_sequence(sweep_values[principal_axis])
    secondary_values = _as_sequence(sweep_values[secondary_axis])
    if principal_axis == secondary_axis:
        secondary_values = [secondary_values[0]]

    base_values = {key: _as_sequence(value)[0] for key, value in sweep_values.items()}
    rows: list[dict[str, Any]] = []

    for secondary_value in secondary_values:
        current_guess = _as_5vector(x_guess, "x_guess").copy()
        for principal_value in principal_values:
            case_values = dict(base_values)
            case_values[principal_axis] = principal_value
            case_values[secondary_axis] = secondary_value
            if update_system_model is not None:
                update_system_model(system_model, case_values)

            solver = (
                solver_factory(reference_point)
                if solver_factory is not None
                else _default_vsm_solver(_as_3vector(reference_point))
            )
            result, body = solve_vsm_quasi_steady_trim(
                body_aero=build_body(case_values),
                center_of_gravity=center_of_gravity,
                reference_point=reference_point,
                system_model=system_model,
                x_guess=current_guess,
                solver=solver,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                transformation_c_from_vsm=transformation_c_from_vsm,
                include_gravity=include_gravity,
                axes=axes,
                moment_tolerance=moment_tolerance,
                return_timing_breakdown=return_timing_breakdown,
                max_nfev=max_nfev,
            )
            rows.append(
                {
                    "principal_axis": principal_axis,
                    "secondary_axis": secondary_axis,
                    "principal_value": principal_value,
                    "secondary_value": secondary_value,
                    "case_values": case_values,
                    "result": result,
                    "body": body,
                }
            )
            if result.get("success", False):
                current_guess = np.asarray(result["opt_x"], dtype=float)

    return rows


def vsm_quasi_steady_sweep_to_dataframe(sweep_rows: Sequence[Mapping[str, Any]]):
    """Convert VSM aerodynamic sweep rows into a flat pandas DataFrame."""
    import pandas as pd

    rows = []
    for row in sweep_rows:
        result = row["result"]
        opt_x = np.asarray(result["opt_x"], dtype=float)
        cmx, cmy, cmz = np.asarray(result["cm"], dtype=float)
        rows.append(
            {
                "principal_axis": row["principal_axis"],
                "secondary_axis": row["secondary_axis"],
                "principal_value": float(row["principal_value"]),
                "secondary_value": float(row["secondary_value"]),
                "speed_tangential": float(opt_x[0]),
                "angle_roll_body_deg": float(opt_x[1]),
                "angle_pitch_body_deg": float(opt_x[2]),
                "angle_yaw_body_deg": float(opt_x[3]),
                "timeder_angle_course_body": float(opt_x[4]),
                "aoa_center_deg": float(result["aoa_deg"]),
                "aoa_course_deg": float(result["aoa_course_deg"]),
                "beta_center_deg": float(result["side_slip_deg"]),
                "beta_course_deg": float(result["side_slip_course_deg"]),
                "aero_roll_deg": float(result["aero_roll_deg"]),
                "cl": float(result["cl"]),
                "cd": float(result["cd"]),
                "cmx": float(cmx),
                "cmy": float(cmy),
                "cmz": float(cmz),
                "norm_cm": float(np.linalg.norm([cmx, cmy, cmz])),
                "cfx": float(result["cfx"]),
                "cfy": float(result["cfy"]),
                "success": bool(result["success"]),
                "success_physical": bool(result["success_physical"]),
            }
        )
    return pd.DataFrame(rows)


def plot_vsm_quasi_steady_sweep(
    df: Any,
    principal_axis: str,
    secondary_axis: str,
    *,
    show: bool = True,
) -> tuple[Any, Any] | None:
    """Plot standard VSM aerodynamic quasi-steady sweep figures."""
    import matplotlib.pyplot as plt

    if df.empty:
        return None

    x_col = "principal_value"
    line_col = "secondary_value"
    fig1, ax1 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for sec_val in sorted(df[line_col].dropna().unique()):
        sub = df[df[line_col] == sec_val].sort_values(x_col)
        label = f"{secondary_axis}={sec_val:.3f}"
        ax1[0].plot(sub[x_col], sub["timeder_angle_course_body"], "o-", label=label)
        ax1[1].plot(sub[x_col], sub["beta_center_deg"], "o-", label=label)
        ax1[2].plot(sub[x_col], sub["aero_roll_deg"], "o-", label=label)

    ax1[0].axhline(0, color="k", linewidth=0.8)
    ax1[0].set_ylabel("course rate [rad/s]")
    ax1[0].legend()
    ax1[1].set_ylabel("Sideslip center [deg]")
    ax1[2].set_xlabel(principal_axis)
    ax1[2].set_ylabel("Aero roll angle [deg]")
    fig1.suptitle(
        f"VSM aerodynamic quasi-steady sweep (x={principal_axis}, series={secondary_axis})",
        y=0.995,
    )
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for sec_val in sorted(df[line_col].dropna().unique()):
        sub = df[df[line_col] == sec_val].sort_values(x_col)
        label = f"{secondary_axis}={sec_val:.3f}"
        ax2[0].plot(sub[x_col], sub["aoa_center_deg"], "o-", label=label)
        ax2[1].plot(sub[x_col], sub["cl"], "o-", label=label)
        ax2[2].plot(sub[x_col], sub["cd"], "o-", label=label)

    ax2[0].set_ylabel("AoA center [deg]")
    ax2[0].legend()
    ax2[1].set_ylabel("Lift coeff")
    ax2[2].set_ylabel("Drag coeff")
    ax2[2].set_xlabel(principal_axis)
    fig2.tight_layout()
    if show:
        plt.show()
    return fig1, fig2


# Compatibility aliases for scripts migrated from Vortex-Step-Method.
solve_quasi_steady_state = solve_vsm_quasi_steady_trim
compute_stability_derivatives = compute_vsm_trim_stability_derivatives
run_quasi_steady_sweep = run_vsm_quasi_steady_sweep
quasi_steady_sweep_rows_to_dataframe = vsm_quasi_steady_sweep_to_dataframe
plot_quasi_steady_sweep_dataframe = plot_vsm_quasi_steady_sweep


__all__ = [
    "DEFAULT_AXES",
    "DEFAULT_BOUNDS_LOWER",
    "DEFAULT_BOUNDS_UPPER",
    "DEFAULT_TRANSFORMATION_C_FROM_VSM",
    "AxisDefinition",
    "compute_stability_derivatives",
    "compute_vsm_trim_stability_derivatives",
    "plot_quasi_steady_sweep_dataframe",
    "plot_vsm_quasi_steady_sweep",
    "quasi_steady_sweep_rows_to_dataframe",
    "run_quasi_steady_sweep",
    "run_vsm_quasi_steady_sweep",
    "solve_quasi_steady_state",
    "solve_vsm_quasi_steady_trim",
    "vsm_quasi_steady_sweep_to_dataframe",
]

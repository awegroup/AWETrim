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
import math
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import casadi as ca
import numpy as np
from scipy.optimize import least_squares

from awetrim.environment.profile_laws import LOG_BASED_MODELS
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
    is_with_artificial_viscosity: bool = False,
    artificial_viscosity_factor: float = 0.035,
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
    # ``is_with_artificial_viscosity`` enables the parameter-free Li/Gaunaa
    # spanwise artificial viscosity in the gamma loop (same option the
    # aerostructural side exposes via as_config aerodynamic.*). It exists to
    # stabilise the loop around stall: a trim within ~1 deg of the stall margin
    # can make the relaxed-Picard iteration oscillate indefinitely, and the
    # unconverged gamma then corrupts the outer trim residuals.
    return Solver(
        reference_point=reference_point,
        gamma_initial_distribution_type="zero",
        allowed_error=allowed_error,
        gamma_loop_type=gamma_loop_type,
        is_with_artificial_viscosity=is_with_artificial_viscosity,
        artificial_viscosity_factor=artificial_viscosity_factor,
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


def _trim_omega_c_vsm(
    system_model: "AWETrimSystemModel",
    transformation_c_from_vsm: np.ndarray,
    course_rate_body: float,
    axes: AxisDefinition,
) -> np.ndarray:
    """Full course-frame transport rate ``Omega_C`` in the trim (VSM) frame.

    This is the rotation rate of the steadily co-rotating equilibrium (Cayon &
    Schmehl), carrying the great-circle normal component ``v_tau/r`` in
    addition to the radial turning component. It drives both the aerodynamic
    body rates and the gyroscopic couple of the trim, and equals the
    ``Omega_C`` baseline of the stability linearisation. Must be called after
    the system model carries the current iterate's ``speed_tangential`` and
    course rate. Falls back to the radial-only reduction
    ``-course_rate * e_radial`` when the system model does not expose
    ``velocity_rotation_course_frame``.
    """
    if hasattr(system_model, "velocity_rotation_course_frame"):
        omega_c_course = _as_numeric_3vector(
            system_model, system_model.velocity_rotation_course_frame
        )
        return np.asarray(transformation_c_from_vsm, dtype=float) @ np.asarray(
            omega_c_course, dtype=float
        ).reshape(3)
    return -float(course_rate_body) * _as_3vector(axes.radial)


def _gyroscopic_trim_moment(
    omega_c_vsm: np.ndarray,
    inertia_cg: np.ndarray,
    *,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
) -> np.ndarray:
    """Gyroscopic couple ``-Omega_C x (I_cg Omega_C)`` of the steady
    co-rotation, in the trim (VSM) frame [N m].

    ``inertia_cg`` is the kite inertia tensor about the CG in the reference
    (zero-attitude) geometry basis; it is rotated by the trim attitude exactly
    like the geometry. This is the CG part of the inertial moment about the
    tether attachment (Cayon & Schmehl, Eq. ``inertial_moment_general``): the
    parallel-axis remainder of ``-Omega_C x I_B Omega_C`` is supplied by
    crossing the CG arm with the centripetal part of the inertial force in the
    trim residual (the two are equal by the Jacobi identity,
    ``c x (-m W x (W x c)) = -W x (m(|c|^2 1 - c c^T) W)``), so the transfer-
    from-CG assembly used by the trim solvers is complete — do not switch this
    couple to ``I_B`` or the parallel-axis part would be counted twice.
    """
    omega_c = _as_3vector(omega_c_vsm)
    rotation = _compose_attitude_rotation(
        roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg, axes=axes
    )
    inertia_vsm = rotation @ np.asarray(inertia_cg, dtype=float) @ rotation.T
    return -np.cross(omega_c, inertia_vsm @ omega_c)


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
    inertia_cg: np.ndarray | None = None,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-2,
    return_timing_breakdown: bool = False,
    max_nfev: int | None = None,
    prescribed_roll_deg: float | None = None,
    gamma_tolerance: float = 1e-6,
    gamma_loop: str = "base",
    is_with_artificial_viscosity: bool = False,
    artificial_viscosity_factor: float = 0.035,
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

    ``inertia_cg`` is the optional kite inertia tensor about the CG in the
    reference (zero-attitude) geometry basis. When given, the gyroscopic
    couple ``-Omega_C x (I_cg Omega_C)`` of the steady co-rotation is added to
    the moment balance (Cayon & Schmehl, Eq. ``inertial_moment_general``); a
    body-scale term, of order (kite size)/(tether length) relative to the
    translational inertial moment. ``None`` (default) omits it.

    The aerodynamic body rates are the full course-frame transport rate
    ``Omega_C`` of the steady co-rotation (great-circle normal component
    ``v_tau/r`` included), matching the ``Omega_C`` baseline of the stability
    linearisation; the radial-only reduction is used only when the system
    model does not expose ``velocity_rotation_course_frame``.

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
        solver = _default_vsm_solver(
            reference_point,
            gamma_tolerance,
            gamma_loop,
            is_with_artificial_viscosity=is_with_artificial_viscosity,
            artificial_viscosity_factor=artificial_viscosity_factor,
        )

    def evaluate_kinematics(x: np.ndarray) -> dict[str, np.ndarray]:
        speed_tangential, roll_deg, pitch_deg, yaw_deg, course_rate_body = x
        _set_course_rate_body(system_model, course_rate_body)
        system_model.speed_tangential = speed_tangential

        # Transport rate of the steady co-rotation and the CG arm rotated with
        # the attitude iterate — the geometry rotates about the reference
        # point, so the (body-fixed) CG position does too.
        omega_c_vsm = _trim_omega_c_vsm(
            system_model, transformation_c_from_vsm, course_rate_body, axes
        )
        attitude_rotation = _compose_attitude_rotation(
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg, axes=axes
        )
        cg_arm = attitude_rotation @ (center_of_gravity - reference_point)

        # Exact inertial reaction of the co-rotating kite, -m a_cg (Cayon &
        # Schmehl, Eq. ``qs_inertial_force``): the transport part -m a_B plus
        # the centripetal part of the CG offset about the attachment point.
        # Crossing ``cg_arm`` with this force in the moment balance supplies
        # the parallel-axis part of -Omega_C x I_B Omega_C (Jacobi identity),
        # which is why ``_gyroscopic_trim_moment`` keeps the CG tensor.
        inertial_force = -_system_model_mass_total(system_model) * (
            _as_3vector(
                transformation_c_from_vsm @ _acceleration_course_body(system_model)
            )
            + np.cross(omega_c_vsm, np.cross(omega_c_vsm, cg_arm))
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
            "omega_c_vsm": omega_c_vsm,
            "cg_arm": cg_arm,
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

        # Full transport rate of the steady co-rotation (computed once in
        # ``evaluate_kinematics`` for this iterate): the body rotates at
        # Omega_C, so the aero sees the full rotation (great-circle normal
        # component included), matching the stability baseline.
        omega_c_vsm = _as_3vector(kin["omega_c_vsm"])
        cg_arm = _as_3vector(kin["cg_arm"])
        omega_c_mag = float(np.linalg.norm(omega_c_vsm))
        omega_c_axis = (
            omega_c_vsm / omega_c_mag
            if omega_c_mag > 1e-12
            else -_as_3vector(axes.radial)
        )

        # rates_in_body_frame=True: use the axis as-is (world components).
        # False would multiply by the body's ``geometry_rotation``, silently
        # tilting the world-frame Omega_C for steering-rolled bodies
        # (``body.rotate(...)``); identity-rotation bodies are unaffected.
        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_course_deg,
            side_slip=beta_course_deg,
            body_rates=omega_c_mag,
            body_axis=omega_c_axis,
            reference_point=reference_point,
            rates_in_body_frame=True,
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

        moment_vec = np.cross(cg_arm, inertial_force)
        if include_gravity:
            moment_vec += np.cross(cg_arm, gravity_force)
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

        # Gyroscopic couple of the steady co-rotation (point-independent).
        if inertia_cg is not None:
            gyro_cm = (
                _gyroscopic_trim_moment(
                    omega_c_vsm,
                    inertia_cg,
                    roll_deg=roll_deg,
                    pitch_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                    axes=axes,
                )
                / denom
            )
            cmx += gyro_cm[0]
            cmy += gyro_cm[1]
            cmz += gyro_cm[2]

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
    inertia_cg: np.ndarray | None = None,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-2,
    max_nfev: int | None = None,
    gamma_tolerance: float = 1e-6,
    gamma_loop: str = "base",
    is_with_artificial_viscosity: bool = False,
    artificial_viscosity_factor: float = 0.035,
    tether_model: str = "williams",
    prescribed_roll_deg: float | None = None,
    gamma_seed: np.ndarray | None = None,
) -> tuple[dict[str, Any], VsmBodyAerodynamics]:
    """Coupled VSM trim with a consistent (off-radial) tether force.

    ``gamma_seed`` (optional, one value per panel): initial circulation guess
    passed to EVERY inner VSM solve. Near stall the gamma solution is
    multi-valued and a cold-started loop can converge onto a different branch
    than the one a deformed geometry was produced with; seeding each residual
    evaluation from the same fixed vector (e.g. the aerostructural coupled
    solver's converged circulation) selects the intended branch while keeping
    the evaluation deterministic and smooth in the trim unknowns (unlike
    history-dependent warm chaining). A seeded solve that fails to converge
    falls back to a cold start for that evaluation.

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

    ``inertia_cg`` optionally adds the gyroscopic couple of the steady
    co-rotation to the moment balance, as in
    :func:`solve_vsm_quasi_steady_trim`.

    ``prescribed_roll_deg`` pins the roll DOF to a kinematically prescribed
    value (geometric bridle steering: pass the pre-rotated baseline body and
    ``prescribed_roll_deg=0``, exactly as in :func:`solve_vsm_quasi_steady_trim`
    / :func:`turn_radius_vs_steering_delta`). The roll unknown and the
    roll-moment residual are dropped (5 unknowns / 5 residuals); ``cmx``
    becomes the reaction the steering lines carry, reported as
    ``reaction_roll_moment_nm``.
    """
    from awetrim.system.williams_tether import WilliamsTether
    from awetrim.utils.reference_frames import transformation_C_from_W

    tether = getattr(system_model, "tether", None)
    # Only the ``williams`` branch needs the Williams object itself (it calls
    # ``tether_shape_symbolic`` and owns the tether-length unknown). The
    # ``rigid_lumped`` branch just rebuilds a RigidLumpedTether from the
    # diameter/density, so requiring a WilliamsTether there would lock out
    # callers that legitimately carry another tether model -- e.g. the
    # aerostructural coupled solver, whose system model holds a
    # RigidLumpedTether.
    if tether_model == "williams" and not isinstance(tether, WilliamsTether):
        raise TypeError(
            "solve_vsm_qs_trim_with_williams_tether with tether_model="
            "'williams' requires a WilliamsTether instance on "
            f"system_model.tether; got {type(tether).__name__}."
        )
    if tether_model == "rigid_lumped" and not (
        hasattr(tether, "diameter_tether") and hasattr(tether, "density_tether")
    ):
        raise TypeError(
            "tether_model='rigid_lumped' requires a tether exposing "
            "diameter_tether and density_tether on system_model.tether; got "
            f"{type(tether).__name__}."
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
        solver = _default_vsm_solver(
            reference_point,
            gamma_tolerance,
            gamma_loop,
            is_with_artificial_viscosity=is_with_artificial_viscosity,
            artificial_viscosity_factor=artificial_viscosity_factor,
        )

    _gamma_seed = None if gamma_seed is None else np.asarray(gamma_seed, dtype=float)
    try:
        _solve_accepts_gamma_seed = (
            "gamma_distribution" in inspect.signature(solver.solve).parameters
        )
    except (TypeError, ValueError):
        _solve_accepts_gamma_seed = False

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

    # Tether rotation rate ``omega`` (course-frame transport rate) in the wind
    # frame, driving the per-node ``v_n = omega x r_n`` / ``a_n = omega x v_n``
    # tether drag (zero in a steady straight-line state; nonzero for turning
    # states -- downloop, figure-8). It depends on the trim state (v_tau,
    # course_rate), so it MUST be recomputed at every residual evaluation (it is
    # a live input to ``coupled_fun`` below). Freezing it at the initial guess
    # makes the tether see the wrong rotation rate away from the seed -- the
    # tether tip sweeps ~omega*r ~ tens of m/s, not a small effect -- and yields
    # seed-dependent spurious trim roots.
    def _tether_omega_wind(
        speed_tangential: float, course_rate_body: float
    ) -> np.ndarray:
        if not hasattr(system_model, "velocity_rotation_course_frame"):
            return np.zeros(3)
        _set_course_rate_body(system_model, course_rate_body)
        system_model.speed_tangential = speed_tangential
        return T_Wind_from_Csm @ _as_numeric_3vector(
            system_model, system_model.velocity_rotation_course_frame
        )

    # --- Tether force at the kite, per model. -----------------------------
    # ``williams``: symbolic shape in "collapsed" mode -- the kite-end tether
    # vector is supplied as a parameter (the trim resultant), so the 3-D force
    # balance is baked in and only ground closure remains. ``omega`` (the tether
    # rotation rate) is a Function INPUT, supplied live per residual evaluation
    # (see _tether_omega_wind). ``rigid_lumped``: the ROM's RigidLumpedTether,
    # evaluated numerically at the current state.
    coupled_fun = None
    rl_tether = None
    if tether_model == "williams":
        _ktv = ca.MX.sym("kite_tension_vector", 3)
        _omega_sym = ca.MX.sym("omega_wind_shape", 3)
        _sh = tether.tether_shape_symbolic(
            env=system_model,
            r_kite=tether.r_kite_sym,
            kite_tension_vector=_ktv,
            tether_length=tether.tether_length,
            omega=_omega_sym,
        )
        coupled_fun = ca.Function(
            "williams_collapsed",
            [tether.tether_length, ca.vertcat(tether.r_kite_sym, _ktv), _omega_sym],
            [_sh["ground_position"], _sh["positions"], _sh["tensions"]],
            ["length", "p", "omega"],
            ["ground", "positions", "tensions"],
        )
    elif tether_model == "rigid_lumped":
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
        and wind.wind_model in LOG_BASED_MODELS
        and r_kite_wind[2] <= float(wind_z0)
    ):
        raise ValueError(
            f"Williams tether ({wind.wind_model} wind) needs the kite above the wind "
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

        # Transport rate of the steady co-rotation and the CG arm rotated with
        # the attitude iterate (the geometry rotates about the reference
        # point, so the combined wing+KCU CG does too). Computed before the
        # inertial terms because both need them.
        omega_c_vsm = _trim_omega_c_vsm(
            system_model, transformation_c_from_vsm, course_rate_body, axes
        )
        attitude_rotation = _compose_attitude_rotation(
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg, axes=axes
        )
        cg_arm = attitude_rotation @ (center_of_gravity - reference_point)

        accel_course = _acceleration_course_body(system_model)
        inertial_force_wing = -mass_wing_value * _as_3vector(
            transformation_c_from_vsm @ accel_course
        )
        inertial_force_kcu = -mass_kcu * _as_3vector(
            transformation_c_from_vsm @ accel_course
        )
        # Centripetal part of the CG offset about the attachment point (Cayon
        # & Schmehl, Eq. ``qs_inertial_force``), with the combined mass at the
        # combined CG. Crossing ``cg_arm`` with it in the moment balance
        # supplies the parallel-axis part of -Omega_C x I_B Omega_C (Jacobi
        # identity), completing the I_cg gyroscopic couple below.
        inertial_force_offset = -(mass_wing_value + mass_kcu) * np.cross(
            omega_c_vsm, np.cross(omega_c_vsm, cg_arm)
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

        # Full transport rate of the steady co-rotation (computed above): the
        # body rotates at Omega_C, so the aero sees the full rotation
        # (great-circle normal component included), matching the stability
        # baseline.
        omega_c_mag = float(np.linalg.norm(omega_c_vsm))
        omega_c_axis = (
            omega_c_vsm / omega_c_mag
            if omega_c_mag > 1e-12
            else -_as_3vector(axes.radial)
        )

        # rates_in_body_frame=True: use the axis as-is (world components); see
        # the note in solve_vsm_quasi_steady_trim.
        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_deg,
            side_slip=beta_deg,
            body_rates=omega_c_mag,
            body_axis=omega_c_axis,
            reference_point=reference_point,
            rates_in_body_frame=True,
        )
        if _solve_accepts_gamma_seed and _gamma_seed is not None:
            res = solver.solve(working_body, gamma_distribution=_gamma_seed)
            if not bool(res.get("gamma_converged", True)):
                res = solver.solve(working_body)  # cold retry
        else:
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
            cg_arm,
            inertial_force_wing + inertial_force_kcu + inertial_force_offset,
        )
        if include_gravity:
            moment_vec += np.cross(cg_arm, gravity_force_total)
        dcm = moment_vec / denom_m
        cmx += dcm[0]
        cmy += dcm[1]
        cmz += dcm[2]

        # Gyroscopic couple of the steady co-rotation (point-independent).
        if inertia_cg is not None:
            gyro_cm = (
                _gyroscopic_trim_moment(
                    omega_c_vsm,
                    inertia_cg,
                    roll_deg=roll_deg,
                    pitch_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                    axes=axes,
                )
                / denom_m
            )
            cmx += gyro_cm[0]
            cmy += gyro_cm[1]
            cmz += gyro_cm[2]

        net_force = (
            total_aero_force
            + inertial_force_wing
            + inertial_force_kcu
            + inertial_force_offset
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
            # Kite weight and d'Alembert inertial reaction, surfaced so the
            # coupled aerostructural solver can distribute them over the
            # structural nodes. It reads results["gravity_force"] /
            # ["inertial_force"] and silently falls back to ZERO when they are
            # absent -- which would deform the kite with no weight at all.
            "gravity_force": (
                gravity_force_total if include_gravity else np.zeros(3, dtype=float)
            ),
            "inertial_force": (
                inertial_force_wing + inertial_force_kcu + inertial_force_offset
            ),
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
            cf = (
                np.array(
                    [
                        np.dot(total_force, axes.course),
                        np.dot(total_force, axes.normal),
                        np.dot(total_force, axes.radial),
                    ],
                    dtype=float,
                )
                / payload["denom_f"]
            )
            return np.concatenate([payload["trim_res"][:3], cf])

        # williams (collapsed): kite-end tether vector = trim resultant, so the
        # 3-D force balance holds by construction; only moments and the tether
        # ground closure remain. Ground residual normalised by distance_radial
        # so it is comparable in magnitude to the dimensionless moments.
        F_kite_wind = T_Wind_from_VSM @ net_force_vsm
        p = np.concatenate([r_kite_wind, F_kite_wind])
        omega_use = _tether_omega_wind(float(x[0]), float(x[4]))
        ground = np.asarray(
            coupled_fun(length=float(x[5]), p=p, omega=omega_use)["ground"]
        ).reshape(3)
        return np.concatenate([payload["trim_res"][:3], ground / distance_radial])

    if prescribed_roll_deg is None:
        opt = least_squares(
            joint_residual,
            x0,
            jac="2-point",
            bounds=(lb, ub),
            max_nfev=max_nfev,
        )
        opt_x_full = np.asarray(opt.x, dtype=float)
    else:
        # Kinematic bridle steering: roll is a constraint, not a DOF. Optimise
        # the remaining unknowns against [cmy, cmz, closure(3)]; cmx becomes
        # the roll reaction the steering lines must carry.
        _free = np.array([0, 2, 3, 4, 5], dtype=int)
        _roll_fixed = float(prescribed_roll_deg)

        def _to_full(x5: np.ndarray) -> np.ndarray:
            x = np.empty(6, dtype=float)
            x[_free] = np.asarray(x5, dtype=float)
            x[1] = _roll_fixed
            return x

        opt = least_squares(
            lambda x5: joint_residual(_to_full(x5))[1:],
            np.clip(x0[_free], lb[_free], ub[_free]),
            jac="2-point",
            bounds=(lb[_free], ub[_free]),
            max_nfev=max_nfev,
        )
        opt_x_full = _to_full(np.asarray(opt.x, dtype=float))
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
    logging.debug(
        "[williams-trim] x*: trim=%s  williams=%s", opt_x_full[:5], opt_x_full[5:]
    )

    res_at_opt = joint_residual(opt_x_full)
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
    payload = _compute_trim_payload(opt_x_full[:5])
    F_kite_vsm = payload["force_kite_resultant"]
    F_kite_wind = T_Wind_from_VSM @ F_kite_vsm

    if prescribed_roll_deg is None:
        physical_success = bool(all(abs(float(c)) < moment_tolerance for c in cm_res))
    else:
        # cmx is the bridle-line reaction, not a residual to drive to zero.
        physical_success = bool(
            all(abs(float(c)) < moment_tolerance for c in cm_res[1:])
        )

    if tether_model == "rigid_lumped":
        tension_opt = float(opt_x_full[5])
        f_tether_vsm = payload["f_tether_offrad_vsm"] - tension_opt * axes.radial
        tether_force = float(np.linalg.norm(f_tether_vsm))
        positions = np.zeros((0, 3))
        tensions = np.zeros((0, 3))
        elev_last = az_last = tether_length = float("nan")
        cfx, cfy, cfz = float(res_at_opt[3]), float(res_at_opt[4]), float(res_at_opt[5])
    else:  # williams (collapsed)
        tether_length = float(opt_x_full[5])
        _out = coupled_fun(
            length=tether_length,
            p=np.concatenate([r_kite_wind, F_kite_wind]),
            omega=_tether_omega_wind(float(opt_x_full[0]), float(opt_x_full[4])),
        )
        positions = np.asarray(_out["positions"])
        tensions = np.asarray(_out["tensions"])
        _dir = F_kite_wind / (float(np.linalg.norm(F_kite_wind)) + 1e-12)
        elev_last = float(np.arcsin(np.clip(_dir[2], -1.0, 1.0)))
        az_last = float(np.arctan2(_dir[1], _dir[0]))
        tether_force = float(np.linalg.norm(F_kite_wind))
        cfx = cfy = cfz = 0.0  # force balance baked in

    # Aerodynamic roll phi_a, same construction as solve_vsm_quasi_steady_trim:
    # the aero-force resultant decomposed in the (lift, side) basis built from
    # the apparent wind and the radial axis. payload["total_aero_force"] and
    # payload["va"] share the course basis with axes.radial (validated: the
    # symmetric zero-steering trim gives phi_a ~ 0 there, and the force is
    # dominantly radial).
    _va_unit = np.asarray(payload["va"], dtype=float)
    _va_unit = _va_unit / (float(np.linalg.norm(_va_unit)) + 1e-12)
    _lift_dir = axes.radial - np.dot(axes.radial, _va_unit) * _va_unit
    _side_dir = np.cross(_lift_dir, _va_unit)
    _f_aero = np.asarray(payload["total_aero_force"], dtype=float)
    aero_roll_deg = float(
        np.rad2deg(np.arctan2(np.dot(_f_aero, _side_dir), np.dot(_f_aero, _lift_dir)))
    )

    result: dict[str, Any] = {
        "opt_x": np.asarray(opt_x_full[:5], dtype=float),
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
        "aero_roll_deg": aero_roll_deg,
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
        "williams_x": np.asarray(opt_x_full[5:], dtype=float),
        "williams_elevation_last_deg": float(np.rad2deg(elev_last)),
        "williams_azimuth_last_deg": float(np.rad2deg(az_last)),
        "williams_tether_length": float(tether_length),
        "williams_ground_residual": ground_res,
        "williams_positions": positions,
        "williams_tensions": tensions,
        # Panel-level VSM outputs, same keys the tetherless trim exposes. The
        # aerostructural coupling needs ``F_distribution`` (the per-panel force
        # it maps onto the structural nodes) and ``alpha_at_ac`` (stall
        # detection); without them this solver cannot drive a coupled solve.
        "gravity_force": payload["gravity_force"],
        "inertial_force": payload["inertial_force"],
        "F_distribution": payload["res"].get("F_distribution"),
        "panel_cp_locations": payload["res"].get("panel_cp_locations"),
        "alpha_at_ac": payload["res"].get("alpha_at_ac"),
        "gamma_distribution": payload["res"].get("gamma_distribution"),
        "optimizer": opt,
    }
    if prescribed_roll_deg is not None:
        result["prescribed_roll_deg"] = float(prescribed_roll_deg)
        # denom_m = q A c_max = denom_f * c_max (see _compute_trim_payload).
        result["reaction_roll_moment_nm"] = float(
            cm_res[0] * payload["denom_f"] * projected_area_cache["max_chord"]
        )
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
#: Canonical free rigid-body states. The 6-DOF body has 12 states; the
#: positions ``x`` and ``y`` are held fixed (lateral translation on the tether
#: sphere belongs to the slow trajectory dynamics), leaving these 10 free
#: states. The lateral velocity ``v`` IS a fast state: sideslip relief
#: (beta_a = psi - v/V_a) and the side-force damping Y_v act on the mode
#: timescale, and prescribing the velocity direction instead locks sideslip
#: to the yaw attitude and overdrives the roll-yaw cross stiffness.
ALL_STATE_NAMES: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "z",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
)

#: Position-augmented state set of the optional 12-state block
#: (``position_states=True``): the tangential positions relative to the
#: (rotating + reeling) trim reference point, resolved in the frozen
#: stability axes — ``x`` along ``axes.course``, ``y`` along ``axes.normal``.
#: Appended (not inserted) so rows/cols 0..9 of every augmented object align
#: with the 10-state objects. ``ALL_STATE_NAMES`` itself stays 10 entries —
#: J_full/A_full shapes and every downstream consumer keep the historical
#: contract.
AUG_STATE_NAMES: tuple[str, ...] = (*ALL_STATE_NAMES, "x", "y")

#: Differential states of the optional course-rate block
#: (``course_rate_state=True``): the 10 canonical states with the lateral
#: velocity ``v`` REMOVED. In the course frame the kite velocity is
#: ``[v_tau, 0, v_r]`` by construction (paper Eq. ``absolute_velocity_paper``),
#: so a normal velocity component does not exist and the lateral degree of
#: freedom is the relative turn rate ``chi_dot_turn``. That rate is not a
#: differential state -- ``a_n = -v_tau chi_dot_turn`` (Eq.
#: ``normal_acceleration_turn``) determines it algebraically, and nothing in
#: the rigid-body equations produces ``chi_ddot_turn`` -- so it is eliminated
#: through the normal equation and reported as an output instead.
CHI_STATE_NAMES: tuple[str, ...] = tuple(s for s in ALL_STATE_NAMES if s != "v")

#: Subsets used by the default decoupled (long + lat) split.
LONG_STATES: tuple[str, ...] = ("u", "w", "z", "theta", "q")
LAT_STATES: tuple[str, ...] = ("v", "phi", "psi", "p", "r")

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

#: Column/row index of the lateral velocity state in the canonical ordering.
#: The normal translational equation is the one the course-rate closure
#: consumes, so this index appears in both course-rate variants.
_I_V_STATE = ALL_STATE_NAMES.index("v")

#: Index of each body-rate state in the (course, normal, radial) principal-axis
#: ordering used by the gyroscopic coupling term (p=course, q=normal, r=radial).
_RATE_AXIS_INDEX = {"p": 0, "q": 1, "r": 2}


def _state_indices(
    states: Sequence[str],
    full_state_names: Sequence[str] = ALL_STATE_NAMES,
) -> list[int]:
    """Map state names to their column index in J_full / A_full.

    ``full_state_names`` selects the canonical column ordering — the 10-state
    :data:`ALL_STATE_NAMES` (default) or the position-augmented
    :data:`AUG_STATE_NAMES`.
    """
    full_idx = {name: idx for idx, name in enumerate(full_state_names)}
    try:
        return [full_idx[s] for s in states]
    except KeyError as exc:
        raise ValueError(
            f"Unknown stability state {exc.args[0]!r}. "
            f"Valid names: {list(full_state_names)}"
        ) from None


def _skew(vec: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix ``[v]_x`` with ``[v]_x a = v x a``."""
    vx, vy, vz = (float(v) for v in np.asarray(vec, dtype=float).reshape(3))
    return np.array(
        [[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]],
        dtype=float,
    )


def corotating_state_transform(
    v_kite_trim_axes: np.ndarray,
    omega_c_axes: np.ndarray,
    state_names: Sequence[str] = ALL_STATE_NAMES,
) -> np.ndarray:
    """Constant map ``T`` from frozen-axes to co-rotating-axes perturbation states.

    :func:`compute_vsm_trim_stability_derivatives` resolves every perturbation
    state in the FROZEN stability axes — the course axes of the trim state,
    space-fixed while the body rotates relative to them. The paper convention
    (Cayon & Schmehl, Sect. ``qs_equilibrium``) instead resolves the states in
    the same course axes CARRIED BY THE BODY about the tether attachment:
    axes that coincide with the course axes at trim and co-rotate with the
    perturbed attitude. (This is a component-basis choice, distinct from the
    principal-body-axes ``--stability-frame body`` option.) At a fixed trim
    the two descriptions differ by the constant linear map built here,

        dv_corot     = dv_frozen     + [v_kite_trim]_x dTheta
        domega_corot = domega_frozen + [Omega_C]_x     dTheta

    with the attitude angles and radial position unchanged, so

        A_corot = T @ A_frozen @ inv(T),    vec_corot = T @ vec_frozen,

    and the eigenvalues — hence every stability margin and boundary — are
    invariant. The gravity-tilt force terms and the transport-corrected
    attitude kinematics ``dTheta_dot = K (domega_corot - [Omega_C]_x dTheta)``
    of the co-rotating description (Trevisi 2024, Eq. 8.18) emerge
    automatically from the similarity transform. Inputs are components in the
    stability-axes basis ``(course, normal, radial)``; both are available in
    the result dict as ``v_kite_trim_axes`` and ``omega_c_axes``, and the
    assembled matrix itself is returned there as ``T_corotating_from_frozen``.
    ``T`` is unipotent (identity plus a nilpotent attitude coupling), so
    ``inv(T) = 2 I - T`` exactly.
    """
    idx = {name: i for i, name in enumerate(state_names)}
    transform = np.eye(len(idx), dtype=float)
    attitude_cols = ("phi", "theta", "psi")
    blocks = (
        (("u", "v", "w"), _skew(v_kite_trim_axes)),
        (("p", "q", "r"), _skew(omega_c_axes)),
    )
    for row_names, skew_block in blocks:
        for i, row in enumerate(row_names):
            for j, col in enumerate(attitude_cols):
                if row in idx and col in idx:
                    transform[idx[row], idx[col]] = skew_block[i, j]
    return transform


def cg_position_state_transform(
    cg_offset_axes: np.ndarray,
    state_names: Sequence[str] = AUG_STATE_NAMES,
) -> np.ndarray:
    """Constant map ``S`` from B-referenced to CG-referenced position states.

    The equations of motion are written at the tether attachment B, and the
    position states ``(x, y, z)`` are B's position. Carrying the CG's position
    instead is a different STATE PARAMETERISATION of the same dynamics — the
    reference point of the mass matrix is untouched. With ``c`` the CG offset
    from B (stability-axes components, :func:`_skew` conventions),

        delta_r_cg = delta_r_B + delta_Theta x c = delta_r_B - [c]_x delta_Theta

    and everything else unchanged, so ``S`` is the identity plus one nilpotent
    position-from-attitude block. Hence

        A_cg = S @ A_B @ inv(S),   vec_cg = S @ vec_B,

    a similarity transform: **the eigenvalues are identical**.

    That invariance is the answer to a natural objection — "rotating about the
    CG moves B, so the tether must be re-solved, so the spectrum must change".
    The tether response IS there, in the attitude columns of ``A_cg``; it is
    exactly cancelled by the position columns, because the two descriptions
    differ only in which point's displacement is called a state. Physically,
    which point you track cannot change whether the motion grows.

    The cancellation needs the position DOF to exist in the first place. Apply
    this to the 12-state block (:data:`AUG_STATE_NAMES`), where all three
    components of ``delta_Theta x c`` are states. On the 10-state block only
    the radial component has a state, the tangential part of the induced
    displacement has nowhere to go, and the map is no longer invertible in the
    modelled space: the spectrum then does move, but as an artifact of the
    incomplete position representation rather than as physics.

    Choosing the rotation origin is not free alongside this: the attitude
    column is ``d/dTheta`` at other states fixed, so carrying ``r_B`` means
    rotating about B and carrying ``r_cg`` means rotating about the CG.
    """
    idx = {name: i for i, name in enumerate(state_names)}
    transform = np.eye(len(idx), dtype=float)
    skew_c = _skew(cg_offset_axes)
    for i, pos in enumerate(("x", "y", "z")):
        for j, att in enumerate(("phi", "theta", "psi")):
            if pos in idx and att in idx:
                # (delta_Theta x c)_i = -([c]_x delta_Theta)_i
                transform[idx[pos], idx[att]] = -skew_c[i, j]
    return transform


def _parallel_axis_term(mass: float, cg_offset: np.ndarray) -> np.ndarray:
    """Parallel-axis inertia shift ``m (|c|^2 1 - c c^T)`` from the CG to the
    reference point, for a CG offset ``c`` (Cayon & Schmehl, Eq.
    ``parallel_axis``)."""
    c = _as_3vector(cg_offset)
    return float(mass) * (float(c @ c) * np.eye(3) - np.outer(c, c))


def _bpoint_mass_matrix(
    mass: float, cg_offset: np.ndarray, inertia_b: np.ndarray
) -> np.ndarray:
    """Coupled 6x6 mass matrix of the B-point (non-barycentric) equations,

        [[ m 1,      -m [c]_x ],
         [ m [c]_x,   I_B     ]],

    acting on ``(v_dot, omega_dot)`` (Cayon & Schmehl Eqs. ``force_eom_B`` /
    ``moment_eom_B``; Trevisi 2024, Eq. 5.4). Components of ``cg_offset`` and
    ``inertia_b`` must share one basis; the result is in that basis.
    """
    c_skew = _skew(cg_offset)
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = float(mass) * np.eye(3)
    matrix[:3, 3:] = -float(mass) * c_skew
    matrix[3:, :3] = float(mass) * c_skew
    matrix[3:, 3:] = np.asarray(inertia_b, dtype=float)
    return matrix


def _strip_theory_added_mass(
    body_aero: VsmBodyAerodynamics,
    reference_point: np.ndarray,
    *,
    rho: float,
) -> np.ndarray:
    """Strip-theory apparent-mass (added-mass) matrix about ``reference_point``.

    Each spanwise panel entrains the 2-D flat-plate apparent mass
    ``m' = rho pi c^2 / 4`` per unit span (Lissaman & Brown's parafoil
    treatment), resisting acceleration along the panel normal only. Assembled
    about the reference point this gives the symmetric 6x6 matrix

        M_a = sum_i m_i [[ N_i,        -N_i [r_i]_x        ],
                         [ [r_i]_x N_i, -[r_i]_x N_i [r_i]_x ]],

    with ``N_i = n_i n_i^T`` the normal projector and ``r_i`` the panel
    MID-CHORD (centroid) offset — the 2-D apparent-mass loading of a plate
    acts at mid-chord (the non-circulatory centre of unsteady thin-airfoil
    theory), NOT at the VSM 3/4-chord control point. Each block pair is the
    exact rank-1 spatial inertia ``m_i g g^T`` with ``g = (n_i, r_i x n_i)``,
    so the transfer to the reference point is the rigid (Kirchhoff) transfer
    law by construction. Acts on ``(v_dot, omega_dot)`` like
    :func:`_bpoint_mass_matrix` — simply ADD it to the rigid mass matrix; the
    aerodynamic right-hand side is unchanged. Neglected: the strip's own
    rotary apparent inertia about its mid-chord (``rho pi c^4/128`` per unit
    span, ~2 kg m^2 for the V3). Strip theory over an arc canopy carries no
    3-D/aspect-ratio reduction and no Munk rotation terms: treat magnitudes
    as +-30%.
    """
    matrix = np.zeros((6, 6), dtype=float)
    origin = _as_3vector(reference_point)
    for panel in body_aero.panels:
        corners = np.asarray(panel.corner_points, dtype=float)
        width = float(np.linalg.norm(corners[3] - corners[0]))
        chord = float(panel.chord)
        m_i = float(rho) * np.pi * chord**2 / 4.0 * width
        normal = getattr(panel, "x_airf", None)
        if normal is None:
            normal = np.cross(corners[2] - corners[0], corners[3] - corners[1])
        normal = _as_3vector(normal)
        normal = normal / np.linalg.norm(normal)
        projector = np.outer(normal, normal)
        r_skew = _skew(corners.mean(axis=0) - origin)
        matrix[:3, :3] += m_i * projector
        matrix[:3, 3:] += -m_i * projector @ r_skew
        matrix[3:, :3] += m_i * r_skew @ projector
        matrix[3:, 3:] += -m_i * r_skew @ projector @ r_skew
    return matrix


def _attitude_generator_matrix(
    *,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
) -> np.ndarray:
    """Rotation-vector generators of the (roll, pitch, yaw) attitude
    increments, as columns, in world components.

    ``_compose_attitude_rotation`` builds ``R = R_yaw(e_r) R_pitch(e_n)
    R_roll(e_c)``, so a small increment of each angle at the attitude
    ``(roll, pitch, yaw)`` rotates the body about

        a_roll  = R_yaw R_pitch e_course,
        a_pitch = R_yaw e_normal,
        a_yaw   = e_radial.

    The columns coincide with the axes triad only at zero pitch and yaw; the
    attitude-rate kinematics therefore use the inverse of this matrix rather
    than identity rows.
    """
    yaw_matrix = _rotation_matrix(axes.radial, yaw_deg)
    pitch_matrix = _rotation_matrix(axes.normal, pitch_deg)
    return np.column_stack(
        [
            yaw_matrix @ pitch_matrix @ _as_3vector(axes.course),
            yaw_matrix @ _as_3vector(axes.normal),
            _as_3vector(axes.radial),
        ]
    )


def _gyroscopic_rate_coupling(
    omega_c_body: np.ndarray,
    inertia: np.ndarray,
) -> np.ndarray:
    """Linearised gyroscopic coupling of the body rates (closed form).

    For the Euler moment equation with the full body angular velocity::

        I @ omega_dot + omega x (I @ omega) = M_ext

    ``omega_dot`` gains ``-I^{-1} (omega x I @ omega)``. With
    ``omega = Omega_C + omega_rel`` and ``Omega_C`` fixed by the trajectory,
    linearising at the trim (``omega = Omega_C``) gives two first-order
    pieces, ``Omega_C x (I @ d_omega)`` and ``d_omega x (I @ Omega_C)``,
    so the Jacobian with respect to the body-rate states ``(p, q, r)`` is the
    constant matrix ``-I^{-1} ([Omega_C]_x I - [I Omega_C]_x)``,
    returned in the ``(course, normal, radial) = (p, q, r)`` axis basis.

    Since the B-point rework, the gyroscopic term ``-omega x I_B omega`` is
    part of the right-hand side that ``eval_force_moment`` returns, so its
    derivative arrives in ``J_full`` through the finite differencing and this
    closed form is no longer added during assembly. It is kept as a physics
    reference for the verification harness and tests.

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
    skew_w = _skew(omega_c_body)
    skew_iw = _skew(inertia @ np.asarray(omega_c_body, dtype=float))
    # -I^{-1} ([Omega_C]_x I - [I Omega_C]_x)
    return -np.linalg.solve(inertia, skew_w @ inertia - skew_iw)


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
    mass_matrix: np.ndarray,
    kinematic_map: np.ndarray | None = None,
    full_state_names: Sequence[str] = ALL_STATE_NAMES,
    position_transport: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (J_sub, A) for a chosen subset of states.

    J_sub keeps all 6 force/moment rows so it can be inspected even when the
    caller drops some states. A has one row per state — dynamics for velocity
    and rate states, kinematics for attitude/position states. Kinematic rows
    whose paired rate states are absent collapse to zero rows.

    ``full_state_names`` selects the canonical column ordering of ``J_full``
    (:data:`ALL_STATE_NAMES` default, :data:`AUG_STATE_NAMES` for the
    position-augmented block). ``position_transport`` is the frozen
    course-frame transport rate ``Omega_C`` in the stability-axes basis: the
    frozen axes rotate at that constant rate, so a position perturbation
    ``delta_r = (x, y, z)`` relative to the (rotating + reeling) trim
    reference point obeys ``delta_r_dot = delta_v - Omega_C x delta_r``. With
    ``None`` (default) only the paired-velocity terms remain — exact for the
    10-state set, whose sole position state z has no transport coupling when
    x = y = 0 are frozen.

    ``J_full`` rows are the complete B-point right-hand sides
    ``[F_course, F_normal, F_radial, M_course, M_normal, M_radial]`` about the
    tether attachment, differenced with respect to the canonical states —
    including gravity, tether, and every velocity-product inertial term, so no
    analytic gyroscopic correction is added here.

    ``mass_matrix`` is the coupled 6x6 B-point mass matrix
    ``[[m 1, -m [c]_x], [m [c]_x, I_B]]`` in the ``(course, normal, radial)``
    axis basis (Cayon & Schmehl Eqs. ``force_eom_B``/``moment_eom_B``; Trevisi
    2024, Eq. 5.4): the translational and angular accelerations follow from
    one joint solve, which carries the translational–rotational coupling of
    the CG offset.

    ``kinematic_map`` (3x3) maps the body-rate perturbations ``(p, q, r)`` to
    the attitude-angle rates ``(phi_dot, theta_dot, psi_dot)``. It is the
    inverse of the attitude-increment generator matrix at the trim attitude
    (see :func:`_attitude_generator_matrix`); ``None`` uses identity, exact
    only at zero trim pitch and yaw.
    """
    cols = _state_indices(states, full_state_names)
    J_sub = J_full[:, cols]

    # (v_dot; omega_dot) columns from the joint 6x6 solve, rows in the
    # (course, normal, radial) basis: 0..2 translational, 3..5 angular.
    accel = np.linalg.solve(np.asarray(mass_matrix, dtype=float), J_sub)
    kin_map = (
        np.eye(3) if kinematic_map is None else np.asarray(kinematic_map, dtype=float)
    )
    _ATTITUDE_ROW = {"phi": 0, "theta": 1, "psi": 2}
    # Position states, their paired velocities, and their axis index in the
    # (course, normal, radial) delta_r ordering of the transport coupling.
    _POSITION_ROW = {"x": ("u", 0), "y": ("v", 1), "z": ("w", 2)}
    transport_skew = (
        _skew(position_transport) if position_transport is not None else None
    )

    n = len(states)
    A = np.zeros((n, n))
    for i, s in enumerate(states):
        if s in _FORCE_OUTPUT_ROW:
            A[i, :] = accel[_FORCE_OUTPUT_ROW[s], :]
        elif s in _MOMENT_OUTPUT_ROW:
            A[i, :] = accel[_MOMENT_OUTPUT_ROW[s], :]
        elif s in _POSITION_ROW:
            velocity, axis = _POSITION_ROW[s]
            if velocity in states:
                A[i, states.index(velocity)] = 1.0
            if transport_skew is not None:
                # delta_r_dot = delta_v - Omega_C x delta_r (frozen axes
                # rotating at the constant trim transport rate).
                for other, (_, other_axis) in _POSITION_ROW.items():
                    if other in states:
                        A[i, states.index(other)] -= transport_skew[axis, other_axis]
        elif s in _ATTITUDE_ROW:
            for rate, ax in _RATE_AXIS_INDEX.items():
                if rate in states:
                    A[i, states.index(rate)] = kin_map[_ATTITUDE_ROW[s], ax]
    return J_sub, A


def _eig_block(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if A.size == 0:
        return np.zeros(0, dtype=complex), np.zeros((0, 0), dtype=complex)
    return np.linalg.eig(A)


def _panel_stall_onsets_rad(body_aero: Any) -> np.ndarray:
    """Per-panel 2-D stall-onset AoA [rad] from the panels' polar tables.

    The onset is the first interior Cl maximum in the positive-Cl region of
    ``panel.panel_polar_data`` (columns ``alpha [rad], cl, ...``) — the same
    definition the VSM solver's post-stall artificial-viscosity gate uses.
    Panels without a usable table (e.g. test mocks) get ``inf`` (never
    limiting). Used by :func:`compute_vsm_trim_stability_derivatives` to
    check that the linearisation point itself is attached.
    """
    onsets: list[float] = []
    for panel in getattr(body_aero, "panels", None) or ():
        onset = float("inf")
        table = getattr(panel, "panel_polar_data", None)
        if table is not None:
            arr = np.asarray(table, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                alpha, cl = arr[:, 0], arr[:, 1]
                positive = np.flatnonzero(cl > 0.0)
                for k in positive[1:-1]:
                    if cl[k] > cl[k - 1] and cl[k] > cl[k + 1]:
                        onset = float(alpha[k])
                        break
        onsets.append(onset)
    return np.asarray(onsets, dtype=float)


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
    inertia_cg: np.ndarray | None = None,
    distance_radial: float | None = None,
    eps_vel: float = 0.1,
    eps_angle_deg: float = 0.5,
    eps_rate: float = 0.01,
    eps_position: float = 0.5,
    eps_course_rate: float = 0.02,
    states: Sequence[str] | None = None,
    coupled: bool = False,
    full_omega_c: bool = True,
    rotate_inertia_by_trim: bool = True,
    include_gravity: bool = True,
    include_added_mass: bool = False,
    position_states: bool = False,
    tether_lateral_feedback: bool = True,
    tether_elastic: bool = False,
    course_rate_state: bool = False,
    transport_rate_follows_states: bool = True,
    transformation_c_from_vsm: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    gamma_seed: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute aerodynamic stability derivatives around a VSM trim state.

    ``gamma_seed`` (optional, one value per panel): initial circulation for
    the BASELINE solve at the trim state (the same branch-selection role it
    plays in ``solve_vsm_qs_trim_with_williams_tether``). The baseline's own
    converged circulation then warm-starts every finite-difference solve, as
    before; without a seed the baseline is cold-started.

    The fast subsystem is the coupled B-point (tether-attachment) system of
    Cayon & Schmehl Eqs. ``force_eom_B``/``moment_eom_B``: moments are taken
    about the attachment point (so no tether moment exists for any perturbed
    motion), the CG offset couples the translational and rotational channels
    through the 6x6 mass matrix ``[[m 1, -m [c]_x], [m [c]_x, I_B]]``, and the
    velocity-product inertial terms carry the full angular velocity
    ``omega = Omega_C + omega_rel`` with the transport rate frozen at the trim.

    ``include_gravity`` (default ``True``) adds the kite weight to the force
    model and its moment ``x_cg x F_g`` about the attachment point. The force
    is constant in the world frame, but the moment arm rotates with the
    attitude, so gravity DOES contribute to the Jacobian: the pendulum
    stiffness of the offset CG hanging on the tether. Set it to match the
    trim's own ``include_gravity`` (a gravity-free trim wants it ``False``).
    ``transformation_c_from_vsm`` maps the model's course-frame gravity into
    the trim (VSM) frame, matching the trim solver's convention.

    Parameters
    ----------
    inertia_cg
        Optional full 3x3 kite inertia tensor about the CG in the reference
        (zero-attitude) geometry basis — the same convention as
        :func:`solve_vsm_quasi_steady_trim`'s ``inertia_cg``. When given, it
        overrides ``inertia_xx``/``inertia_yy``/``inertia_zz`` and carries the
        products of inertia into the B-point mass matrix and gyroscopic terms.
        When ``None`` (default), the diagonal ``diag(I_xx, I_yy, I_zz)`` is
        used — exact only when the geometry basis is principal (e.g. body
        stability axes from a rigid-body identification).
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
        If ``True`` (default), the CG offset and the CG inertia tensor
        (``inertia_cg`` or ``diag(I_xx, I_yy, I_zz)``) are rotated by the full
        trim attitude plus the perturbation, matching the aerodynamic
        geometry — correct when the stability axes are the (space-fixed)
        course axes and the kite is tilted from them by the trim attitude.
        Set ``False`` when the stability axes are the body-fixed principal
        axes (``--stability-frame body``): the baseline tensors stay
        principal/diagonal and only the attitude perturbation rotates them.
    include_added_mass
        If ``True``, the strip-theory apparent-mass matrix of the canopy
        (:func:`_strip_theory_added_mass`, evaluated on the zero-attitude
        geometry and rotated with the attitude exactly like the structural
        inertia) is ADDED to the coupled 6x6 mass matrix — in the ``A_full``
        assembly, the decoupled-block divisors, and ``nonlinear_rhs``. The
        aerodynamic right-hand side is unchanged: the quasi-steady VSM solve
        knows nothing of accelerations, so the fluid reaction to *acceleration*
        must enter through the mass matrix or not at all. For ram-air kites
        the entrained air is comparable to the structural mass and reshapes
        the lateral eigenvalues; default ``False`` preserves the historical
        rigid-only behaviour. The result records ``added_mass_model`` and
        ``added_mass_matrix_axes``.
    position_states
        If ``True``, additionally build the position-augmented 12-state block
        over :data:`AUG_STATE_NAMES` — the 10 canonical states plus the
        tangential positions ``x`` (along ``axes.course``) and ``y`` (along
        ``axes.normal``) relative to the (rotating + reeling) trim reference
        point, resolved in the frozen stability axes. Physics of the extra
        columns (every existing output is untouched):

        * **tether**: the ground anchor is fixed, so a lateral kite
          displacement rotates the tether — the Williams fixed-length re-solve
          used for the ``z`` column is evaluated at the displaced ``r_kite``
          (spherical-pendulum stiffness ``~T/r`` plus sag reorientation);
          without a Williams tether a straight-tether analytic tilt
          ``-(T/r)(1 - r_hat r_hat^T)`` is used (recorded under
          ``tether_position_model_aug``);
        * **wind shear**: the kite inflow is evaluated at the displaced
          height through ``system_model.wind.speed_wind_at_height``; the
          augmented block's own ``z`` column carries the same shear term for
          internal consistency (the historical shear-free ``z`` column of
          ``J_full``/``A_full`` is preserved);
        * **kinematics**: the frozen axes rotate at the constant trim
          transport rate, so ``delta_r_dot = delta_v - Omega_C x delta_r``
          (the 10-state ``z_dot = w`` row is the exact x = y = 0 restriction);
        * gravity and course-frame-orientation columns are zero by
          construction in the frozen axes, and ``Omega_C`` stays frozen —
          the augmentation adds position *feedback*, not transport-rate
          perturbations.

        Requires ``system_model`` (with a wind model exposing
        ``speed_wind_at_height``) and a positive ``distance_radial``; raises
        ``ValueError`` otherwise. Outputs: ``J_aug`` (6, 12), ``A_aug``
        (12, 12), ``eig_aug``, ``vec_aug``, ``Tfast_aug``, ``stable_aug``,
        ``state_names_aug``, ``T_corotating_from_frozen_aug``,
        ``tether_position_model_aug``, ``eps_position_lateral_used``, and the
        independent-verification callables ``nonlinear_rhs_aug`` /
        ``nonlinear_rhs_aug_full``.
    tether_lateral_feedback
        Whether a LATERAL position offset changes the tether force (default
        ``True``, the historical behaviour described above). Set ``False`` to
        make the tether translate WITH the kite: only the radial component of
        the position offset reaches the tether model, so the ``x`` and ``y``
        columns carry no tether force and need no re-solve, while the radial
        (``z``) channel is untouched. This removes the spherical-pendulum
        restoring term — appropriate when the pendulum is considered to belong
        to the slow trajectory subsystem rather than to this frozen-slow-state
        block, in which case the default double-counts it. Under a uniform wind
        the ``x``/``y`` columns then vanish identically (aerodynamics, gravity
        and the centrifugal term are all blind to a tangential displacement).
        Applies to the finite-difference columns and to ``nonlinear_rhs_aug``
        alike (both route through one projection), and is recorded in
        ``tether_position_model_aug`` with a ``_radial_only`` suffix and in
        ``tether_lateral_feedback``.
    course_rate_state
        If ``True``, additionally build the course-rate block: the 9
        differential states of :data:`CHI_STATE_NAMES` (the canonical set with
        the lateral velocity ``v`` removed) plus the relative turn rate
        ``chi_dot_turn`` as an *algebraic* variable.

        The course frame is defined by the velocity direction, so the kite
        velocity is ``[v_tau, 0, v_r]`` by construction (Eq.
        ``absolute_velocity_paper``) and has no normal component to integrate.
        The lateral degree of freedom is instead the relative turn rate, which
        cannot be a differential state — ``a_n = -v_tau chi_dot_turn`` (Eq.
        ``normal_acceleration_turn``) determines it, and the rigid-body
        equations produce no ``chi_ddot_turn``. It is therefore eliminated:
        one extra finite-difference column ``J_course_rate = d(F, M) /
        d(chi_dot_turn)`` is taken by perturbing the trim course rate (which
        moves ``Omega_C``'s radial entry, exactly ``-chi_dot_turn``, in *both*
        the aerodynamic body-rate baseline and the inertial terms), and the
        vanishing frame-relative normal acceleration closes the system:

            ``delta_chi_dot_turn = chi_turn_gain_row @ delta_state_9``.

        **Result: ``F_Omega`` vanishes on every differential row, so the
        elimination is a no-op and ``A_chi`` equals the 10-state block with
        ``v`` pinned to zero.** Two exact facts force this. (i) ``chi_dot_turn``
        is a trajectory-CURVATURE term, not a body rate — the body rate equals
        ``Omega_C`` only AT TRIM — so it changes no apparent wind and reaches
        the equations only through the transport inertial force
        ``-m (Omega_C x v)``, whose derivative is purely NORMAL (with
        ``Omega_C = [0, v_tau/r, -chi_dot_turn]`` and ``v = [v_tau, 0, v_r]``
        the cross product is ``[w_n v_r, w_r v_tau, -w_n v_tau]``, and the turn
        rate sits in the middle slot alone). (ii) That force acts at the CG, so
        its right-hand side is the pair ``[P, x_cg x P]``, which the coupled
        B-point mass matrix maps to ``[P/m, 0]`` — pure translation, zero
        angular acceleration. Hence the only nonzero entry of the column is the
        constraint row, ``G_Omega = -v_tau`` exactly (no aerodynamic content).

        The block is therefore worth running not for a turn-rate feedback —
        there is none — but because it is the *correct* 9-state model, and
        because ``chi_turn_gain_row`` is the per-mode turn-rate OUTPUT.
        (Exception: with ``include_added_mass=True`` the mass matrix is no
        longer the rigid B-point one, so ``F_Omega`` is only approximately
        zero.)

        The tether is held at its baseline across this column. That is not just
        convenience: the tether runs along ``e_r`` and the turn-rate
        perturbation is a rotation ABOUT ``e_r``, which moves no node of a
        straight tether at all (only sag responds, at second order).

        Outputs: ``J_course_rate`` (6,), ``A_chi``, ``eig_chi``, ``vec_chi``,
        ``Tfast_chi``, ``stable_chi``, ``state_names_chi``,
        ``chi_turn_gain_row``, ``chi_turn_denominator``,
        ``chi_turn_closure_singular`` (set when the closure denominator
        ``d(a_normal)/d(chi_dot_turn)`` collapses — expect it only at very low
        ``v_tau``), ``eps_course_rate``, and the independent-verification
        callables ``nonlinear_rhs_chi`` / ``nonlinear_rhs_chi_full``.

        The same block also builds the **v-retained** variant ``A_chi10``
        (``eig_chi10``, ``vec_chi10``, ``Tfast_chi10``, ``state_names_chi10``,
        ``chi10_gain_row``): 10 differential states plus the same algebraic
        variable, closing on ``delta_v_dot = 0`` instead of ``delta_v = 0``.
        Both readings come from the one normal momentum equation — with two
        unknowns ``(v_dot, chi_dot_turn)`` and one equation, either the frame
        is frozen and the equation integrates ``v_dot`` (the baseline
        10-state block) or the frame FOLLOWS the velocity direction and the
        equation determines ``chi_dot_turn``. Pinning ``v`` additionally
        forbids a standing normal offset; retaining it lets the frame carry
        one along. The ``v`` row is then identically zero, so the matrix is
        block upper triangular and ``spec(A_chi10) = spec(A_chi)`` plus one
        structural zero (the neutral constant-sideslip mode) — the ``v``
        COLUMN survives, so a standing sideslip still drives the other nine
        equations aerodynamically, visible in the eigenvectors rather than
        the spectrum.

    Always-present outputs
    ----------------------
    ``J_full`` (6, 10), ``A_full`` (10, 10), ``eig_full``, ``vec_full``,
    ``Tfast_full``, ``stable_full``, ``state_names_full``, ``output_names``,
    and ``nonlinear_rhs`` — a callable ``f(delta_state) -> xdot`` for the
    nonlinear fast subsystem, assembled directly from the governing equations
    (independent of ``A_full``): ``f(0)`` is the trim equilibrium residual and
    central-differencing it cross-checks ``A_full``. See the callable's own
    docstring.

    Numerical hygiene: every finite-difference solve is warm-started from the
    baseline (trim-state) circulation and convergence-checked with one cold
    retry — cold-started gamma loops scatter the small-difference lateral
    moment derivatives (yaw channel / small Izz) enough to flip eigenvalue
    signs. Diagnostics: ``n_unconverged_perturbation_solves`` /
    ``perturbation_solves_converged`` (distrust the eigenvalues when False),
    ``gamma_warm_start_used``, and the attached-flow check of the
    linearisation point itself, ``stall_margin_min_deg_at_trim`` /
    ``n_stalled_panels_at_trim`` (first-interior-Cl-peak onsets per panel
    polar vs the baseline ``alpha_at_ac``; NaN/None when the solver or body
    does not expose that information). A trim on a post-stall branch (rigged
    kites can have multiple pitch equilibria) makes the derivatives
    unreliable regardless of differencing care.

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
    # Kite-end tether force vector in the trim (VSM) frame. The coupled trims
    # store the net kite resultant, whose negative is the tether reaction by
    # force balance (paper Eq. ``tether_moment_cg`` uses the full vector);
    # fall back to the radial-only direction for trims without that key.
    _f_kite_vsm = trim_result.get("force_kite_resultant_vsm")
    if _f_kite_vsm is not None:
        f_tether = -_as_3vector(_f_kite_vsm)
    else:
        f_tether = np.array(
            [0.0, 0.0, -float(trim_result["tether_force"])], dtype=float
        )
    r_arm = reference_point - center_of_gravity
    moment_tether_at_cg = np.cross(r_arm, f_tether)
    distance_radial_trim = (
        float(distance_radial)
        if distance_radial is not None and float(distance_radial) > 0.0
        else None
    )
    # Radial (reeling) speed of the trim, for the exact d'Alembert inertial
    # force -m (Omega_C x v); None falls back to the two-term reduction.
    speed_radial_trim = (
        float(_numeric_value_for_symbol(system_model, "speed_radial"))
        if system_model is not None and hasattr(system_model, "speed_radial")
        else None
    )

    if position_states:
        _missing = None
        if system_model is None:
            _missing = "a system_model"
        elif distance_radial_trim is None:
            _missing = "a positive distance_radial"
        elif not hasattr(getattr(system_model, "wind", None), "speed_wind_at_height"):
            _missing = "a wind model exposing speed_wind_at_height"
        if _missing is not None:
            raise ValueError(
                f"position_states=True requires {_missing}: the x/y position "
                "columns need the tether geometry (fixed ground anchor) and "
                "the wind profile at the displaced kite height."
            )

    working_body = copy.deepcopy(body_aero)
    baseline_sections, baseline_spanwise = _baseline_geometry(working_body)
    # The linearisation applies the trim attitude (roll0, pitch0, yaw0) to the
    # baseline. If ``body_aero`` is the *solved* trim body it already carries that
    # attitude (``geometry_rotation``), so capturing its geometry as the baseline
    # would DOUBLE the rotation -> a doubly-rotated kite and a wrong aero moment.
    # Un-rotate the baseline back to the reference (zero-attitude) geometry about
    # the same reference_point, so the attitude is applied exactly once. No-op
    # for an un-rotated body (geometry_rotation == I).
    _R_body = np.asarray(
        getattr(working_body, "geometry_rotation", np.eye(3)), dtype=float
    )
    if _R_body.shape == (3, 3) and not np.allclose(_R_body, np.eye(3)):
        _R_inv = _R_body.T
        _origin = _as_3vector(reference_point)
        baseline_sections = [
            [
                (
                    _origin + _R_inv @ (np.asarray(le, float) - _origin),
                    _origin + _R_inv @ (np.asarray(te, float) - _origin),
                )
                for (le, te) in wing_secs
            ]
            for wing_secs in baseline_sections
        ]
        baseline_spanwise = [_R_inv @ np.asarray(sp, float) for sp in baseline_spanwise]
    projected_area = float(body_aero.wings[0].compute_projected_area())
    max_chord = max(float(panel.chord) for panel in body_aero.panels)

    def _make_frame_chain() -> dict[str, np.ndarray] | None:
        """Wind-frame chain shared by the tether position models and the
        wind-at-height (shear) evaluation of the position-augmented block.

        Returns ``{"T_wind_from_course", "r0_wind"}`` — the course->wind map
        at the trim window position and the trim kite position in the wind
        frame (z up, wind along +x). Built once; historically this lived
        inside the Williams fixed-length builder, but the straight-tether
        fallback and the shear term of ``position_states`` need it for
        non-Williams trims too. Gated so plain (non-Williams, non-position)
        runs execute the exact historical sequence with no side effects.
        """
        if system_model is None:
            return None
        if not (position_states or "williams_tether_length" in trim_result):
            return None
        try:
            from awetrim.utils.reference_frames import transformation_Wind_from_C
        except ImportError:
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
        r0_wind = (
            _as_3vector(trim_result["r_kite"])
            if "r_kite" in trim_result
            else T_wind_from_course @ np.array([0.0, 0.0, distance_radial_trim or 0.0])
        )
        return {"T_wind_from_course": T_wind_from_course, "r0_wind": r0_wind}

    frame_chain = _make_frame_chain()
    if position_states and frame_chain is None:
        raise ValueError(
            "position_states=True requires awetrim.utils.reference_frames "
            "(transformation_Wind_from_C) to build the wind-frame chain."
        )

    def _make_williams_fixed_length_solver():
        if frame_chain is None or "williams_tether_length" not in trim_result:
            return None
        try:
            from awetrim.system.williams_tether import WilliamsTether
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

        T_wind_from_course = frame_chain["T_wind_from_course"]
        r0_wind = frame_chain["r0_wind"]

        # Course-frame transport rate ``omega`` (wind frame) at THIS trim. It is
        # deliberately frozen here and baked into the fixed-length tether solve --
        # do NOT convert it to a live per-solve input by analogy with the trim
        # solver (``_tether_omega_wind``). That freeze was a bug there because the
        # root-finder moved (v_tau, course_rate) away from the seed, so a
        # seed-frozen omega no longer matched the state being solved. Here the
        # state is fixed: omega is evaluated at the converged trim (course_rate0,
        # speed_tangential from x_trim, set just above) and this solver is only
        # used for the radial-position (z) perturbation column, which does not move
        # v_tau or course_rate. Freezing omega across that column is exactly
        # consistent with the aerodynamic body-rate baseline ``omega_c_world`` and
        # the gyroscopic ``Omega_C`` -- the whole fast-subsystem linearisation
        # holds the trajectory-set transport rate fixed (the paper's fast/slow
        # separation). The only omega it neglects is d(v_tau/r)/dz over one radial
        # step, verified at ~1e-5 rad/s (~0.006% of ||omega_c||): negligible.
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
        length_sym = ca.MX.sym("tether_length_unstrained_fixed_length")
        tether_length = float(trim_result["williams_tether_length"])
        # Elasticity: build the shape with WilliamsTether's own per-element
        # stretch model, l_s = (T_local/EA + 1) l_unstrained, temporarily
        # enabling the flag on the shared object for the SYMBOLIC BUILD only
        # (the flag is read at construction; the trim solve elsewhere is
        # untouched). The inextensible model has a hard feasibility boundary
        # — an attitude swing of B that consumes the sag slack has NO
        # solution, tension diverging as the sag straightens — which the
        # finite EA replaces with a steep but smooth stiffness ~EA/L.
        _elastic_build = bool(tether_elastic) or bool(getattr(tether, "elastic", False))
        _elastic_prev = bool(getattr(tether, "elastic", False))
        try:
            tether.elastic = _elastic_build
            shape = tether.tether_shape_symbolic(
                env=system_model,
                r_kite=r_kite_sym,
                tension_kite=tension_sym,
                omega=ca.DM(omega_wind),
                tether_length=length_sym,
                elevation_last=elevation_sym,
                azimuth_last=azimuth_sym,
            )
        finally:
            tether.elastic = _elastic_prev
        x_sym = ca.vertcat(tension_sym, elevation_sym, azimuth_sym)
        residual_fun = ca.Function(
            "williams_fixed_length_residual",
            [x_sym, r_kite_sym, length_sym],
            [shape["ground_position"]],
            ["x", "r_kite", "length"],
            ["residual"],
        )
        jac_fun = ca.Function(
            "williams_fixed_length_residual_jac",
            [x_sym, r_kite_sym, length_sym],
            [ca.jacobian(shape["ground_position"], x_sym)],
            ["x", "r_kite", "length"],
            ["jac"],
        )
        force_fun = ca.Function(
            "williams_fixed_length_force_kite",
            [x_sym, r_kite_sym, length_sym],
            [shape["tether_force_kite"]],
            ["x", "r_kite", "length"],
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

        def _solve_at(
            r_kite_i: np.ndarray,
            x_seed: np.ndarray,
            budget: int,
            length_i: float,
        ):
            def residual(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    residual_fun(
                        x=np.asarray(x, dtype=float),
                        r_kite=r_kite_i,
                        length=length_i,
                    )["residual"]
                ).reshape(-1)

            def jac(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    jac_fun(
                        x=np.asarray(x, dtype=float),
                        r_kite=r_kite_i,
                        length=length_i,
                    )["jac"]
                )

            return least_squares(
                residual,
                np.asarray(x_seed, dtype=float),
                jac=jac,
                bounds=(
                    [0.0, -np.pi / 2 + 1e-3, -2.0 * np.pi],
                    [np.inf, np.pi / 2 - 1e-3, 2.0 * np.pi],
                ),
                max_nfev=budget,
            )

        # Unstrained length. Inextensible build: the trim length verbatim.
        # Elastic build: calibrated by a secant iteration so the elastic
        # solve at ZERO offset reproduces the trim tension exactly — the
        # trim state must stay an equilibrium of the perturbed force model.
        unstrained_length = tether_length
        if _elastic_build:
            ea = float(getattr(tether, "EA", 0.0) or 0.0)
            if ea <= 0.0:
                raise ValueError(
                    "tether_elastic=True needs a positive WilliamsTether.EA "
                    "(E and diameter from the system config)."
                )
            t0 = float(trim_result["tether_force"])

            def _tension_at_zero(length_i: float) -> float:
                sol0 = _solve_at(r0_wind, x0, 2000, length_i)
                if np.linalg.norm(sol0.fun) > 1e-3:
                    raise RuntimeError(
                        "Elastic tether length calibration solve failed at "
                        f"L0={length_i:.6f} m: {sol0.message}"
                    )
                return float(sol0.x[0])

            # Secant on g(L0) = T(L0) - T0, seeded by the uniform-strain
            # estimate and the raw trim length.
            l_a = tether_length / (1.0 + t0 / ea)
            l_b = tether_length
            g_a = _tension_at_zero(l_a) - t0
            for _ in range(12):
                g_b = _tension_at_zero(l_b) - t0
                if abs(g_b) <= 1e-6 * max(t0, 1.0):
                    break
                if g_b == g_a:
                    break
                l_a, l_b, g_a = l_b, l_b - g_b * (l_b - l_a) / (g_b - g_a), g_b
            unstrained_length = l_b

        def solve_force(offset_axes: np.ndarray) -> np.ndarray:
            # Position offset in the stability (VSM) axes -> course -> wind
            # frame. The VSM->course flip matters for the lateral components;
            # the historical radial-only offset was flip-invariant. ``x0`` is
            # the stateless trim seed for EVERY call — never chained across
            # offsets, which would bias the central differences.
            offset_course = np.asarray(transformation_c_from_vsm, dtype=float) @ (
                np.asarray(offset_axes, dtype=float).reshape(3)
            )
            r_kite = r0_wind + T_wind_from_course @ offset_course

            # ``least_squares`` reports ``success=False`` when it exhausts
            # ``max_nfev`` even if it has already converged. The ground-position
            # residual (metres) is the physically meaningful convergence check,
            # so accept the solve whenever that residual is below tolerance.
            sol = _solve_at(r_kite, x0, 200, unstrained_length)
            if np.linalg.norm(sol.fun) > 1e-3:
                # The fixed-length tether has a solution at EVERY offset probed
                # here (B stays well within tether reach — a rotation about the
                # CG changes the anchor distance only at second order), so a
                # residual above tolerance is a seed/basin failure of the
                # root-finder, not physics. Continuation: walk the offset from
                # the trim solution to the target, carrying the solution as
                # the seed. Stateless across calls — each call starts from the
                # same trim seed — so central differences stay unbiased.
                x_seed = x0
                for frac in (0.25, 0.5, 0.75, 1.0):
                    sol = _solve_at(
                        r0_wind + T_wind_from_course @ (frac * offset_course),
                        x_seed,
                        2000,
                        unstrained_length,
                    )
                    x_seed = np.asarray(sol.x, dtype=float)
            if np.linalg.norm(sol.fun) > 1e-3:
                raise RuntimeError(
                    "Williams fixed-length position perturbation solve failed: "
                    f"{sol.message}; residual={sol.fun}"
                )
            force_wind = np.asarray(
                force_fun(x=sol.x, r_kite=r_kite)["force_kite"]
            ).reshape(3)
            # Wind -> course -> VSM. ``T_wind_from_course.T`` alone stops at
            # COURSE-frame components, but the caller sums this force with
            # VSM-frame quantities (the baseline ``f_tether`` is
            # -force_kite_resultant_vsm); the course/normal axes flip
            # ``transformation_c_from_vsm`` (its own inverse) completes the
            # chain. The historical radial-only return hid the missing hop —
            # e_radial is invariant under the flip — so only the small
            # lateral z-derivatives of the tether force carried the wrong
            # sign; for lateral position perturbations the flip is
            # first-order.
            return np.asarray(transformation_c_from_vsm, dtype=float) @ (
                T_wind_from_course.T @ force_wind
            )

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

    # Course-frame components of the trim Omega_C. ``transformation_c_from_vsm``
    # is an involution (diag(-1, -1, 1)), so it maps world -> course as well.
    _flip_c_vsm = np.asarray(transformation_c_from_vsm, dtype=float)
    omega_c_course_trim = _flip_c_vsm @ np.asarray(omega_c_world, dtype=float)

    def _omega_c_for(
        delta_course_rate: float = 0.0,
        speed_tangential_eff: float | None = None,
        distance_radial_eff: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(omega_c_axes, omega_c_world)`` at a perturbed course-frame rate.

        In course-frame components the transport rate is

            ``Omega_C = [0, v_tau/r, -chi_dot_turn]``

        (the paper's ``[0, v_tau/r, (v_tau/r) sin(chi) tan(beta) - chi_dot]``
        with ``chi_dot = chi_dot_gc + chi_dot_turn``). The two entries have
        different status and are perturbed independently here:

        * **normal** ``v_tau/r`` — a KINEMATIC IDENTITY, not a modelling
          choice: it is the rate at which ``e_r`` tilts as the kite flies along
          the sphere. It therefore must follow the fast states ``v_tau`` and
          ``r``, which is what ``speed_tangential_eff`` /
          ``distance_radial_eff`` do. Freezing it drops the response of the
          centrifugal term ``m v_tau^2 / r`` to those states — the term the
          reduced (radial-only) branch carries explicitly, so freezing it also
          makes the two branches disagree.
        * **radial** ``-chi_dot_turn`` — dynamics, set by the normal force
          balance. ``delta_course_rate`` perturbs it (and nothing else), which
          is the ``d/d(chi_dot_turn)`` direction the course-rate block needs.

        Perturbing the normal entry through ``_course_transport_rate_axes``
        would also move the radial entry (via ``chi_dot_gc``) and so contaminate
        the turn rate; the components are therefore rebuilt directly from the
        trim vector.

        The caller must feed the result to the **transport inertial force
        only**. ``Omega_C`` is the rate of the course FRAME; the body rate
        equals it only at trim, and neither entry is a body rate — so this must
        not reach the aerodynamic body rate, the gyroscopic couple, or the
        centripetal CG-offset force.

        All-default arguments short-circuit to the frozen trim pair, so every
        historical evaluation is untouched and free of side effects.
        """
        moves_normal = (
            transport_rate_follows_states
            and omega_c_is_full
            and speed_tangential_eff is not None
            and distance_radial_eff is not None
            and float(distance_radial_eff) > 0.0
        )
        if float(delta_course_rate) == 0.0 and not moves_normal:
            return omega_c_axes, omega_c_world
        omega_course = np.array(omega_c_course_trim, dtype=float)
        if moves_normal:
            omega_course[1] = float(speed_tangential_eff) / float(distance_radial_eff)
        # Radial entry is -chi_dot_turn, so +delta on the turn rate lowers it.
        omega_course[2] = omega_c_course_trim[2] - float(delta_course_rate)
        world_pert = _flip_c_vsm @ omega_course
        # Local axis matrix: R_body is assigned further down, after the
        # baseline eval_force_moment call, so it must not be captured here.
        rows = np.array([axes.course, axes.normal, axes.radial], dtype=float)
        return rows @ world_pert, world_pert

    # Body-fixed CG offset and inertia, rotated with the attitude. The
    # geometry rotates about the reference point, so the CG position and the
    # inertia tensor do too; both feed the coupled B-point mass matrix and the
    # attitude-dependent gravity moment. With ``rotate_inertia_by_trim``
    # (default) the rotation is the full trim attitude plus the perturbation,
    # matching the aerodynamic geometry; with ``False`` (body-fixed principal
    # stability axes) only the perturbation rotates the tensors, so the
    # baseline stays principal/diagonal as before.
    if inertia_cg is None:
        inertia_cg0 = np.diag([float(inertia_xx), float(inertia_yy), float(inertia_zz)])
    else:
        inertia_cg0 = np.asarray(inertia_cg, dtype=float)
        if inertia_cg0.shape != (3, 3):
            raise ValueError("inertia_cg must be a 3x3 CG inertia tensor.")
        if not np.allclose(inertia_cg0, inertia_cg0.T):
            raise ValueError("inertia_cg must be symmetric.")
    cg_offset0 = center_of_gravity - reference_point  # zero-attitude, world

    def _attitude_angles_deg(
        droll_deg: float, dpitch_deg: float, dyaw_deg: float
    ) -> tuple[float, float, float]:
        if rotate_inertia_by_trim:
            return (
                float(roll0) + float(droll_deg),
                float(pitch0) + float(dpitch_deg),
                float(yaw0) + float(dyaw_deg),
            )
        return float(droll_deg), float(dpitch_deg), float(dyaw_deg)

    def _cg_and_inertia_b(
        droll_deg: float = 0.0, dpitch_deg: float = 0.0, dyaw_deg: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """CG offset and I_B about the attachment point (world components) at
        the perturbed attitude."""
        roll_eff, pitch_eff, yaw_eff = _attitude_angles_deg(
            droll_deg, dpitch_deg, dyaw_deg
        )
        rotation = _compose_attitude_rotation(
            roll_deg=roll_eff, pitch_deg=pitch_eff, yaw_deg=yaw_eff, axes=axes
        )
        cg = rotation @ cg_offset0
        inertia_cg_att = rotation @ inertia_cg0 @ rotation.T
        return cg, inertia_cg_att + _parallel_axis_term(mass, cg)

    if rotate_inertia_by_trim:
        _R_attitude = _compose_attitude_rotation(
            roll_deg=float(roll0),
            pitch_deg=float(pitch0),
            yaw_deg=float(yaw0),
            axes=axes,
        )
        inertia_stability = _R_attitude @ inertia_cg0 @ _R_attitude.T
    else:
        inertia_stability = inertia_cg0

    # Strip-theory apparent mass of the canopy, evaluated once on the
    # zero-attitude geometry and rotated with the attitude exactly like the
    # structural tensors (same ``_attitude_angles_deg`` convention). The
    # quasi-steady VSM right-hand side carries no acceleration reaction, so
    # the entrained-air inertia enters through the mass matrix only.
    if include_added_mass:
        _set_body_attitude_from_baseline(
            working_body,
            baseline_sections=baseline_sections,
            baseline_spanwise=baseline_spanwise,
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            axes=axes,
            reference_point=reference_point,
        )
        added_mass0 = _strip_theory_added_mass(
            working_body, reference_point, rho=float(solver.rho)
        )
    else:
        added_mass0 = None

    def _added_mass_world(
        droll_deg: float = 0.0, dpitch_deg: float = 0.0, dyaw_deg: float = 0.0
    ) -> np.ndarray:
        """6x6 added-mass matrix at the perturbed attitude (world components);
        zeros when the model is disabled."""
        if added_mass0 is None:
            return np.zeros((6, 6), dtype=float)
        roll_eff, pitch_eff, yaw_eff = _attitude_angles_deg(
            droll_deg, dpitch_deg, dyaw_deg
        )
        rotation = _compose_attitude_rotation(
            roll_deg=roll_eff, pitch_deg=pitch_eff, yaw_deg=yaw_eff, axes=axes
        )
        block_rotation = np.zeros((6, 6), dtype=float)
        block_rotation[:3, :3] = rotation
        block_rotation[3:, 3:] = rotation
        return block_rotation @ added_mass0 @ block_rotation.T

    # Kite weight in the trim (VSM) frame, matching the trim's own gravity term
    # (``transformation_c_from_vsm @ force_gravity``). Constant in the world
    # frame -> zero contribution to the finite-difference Jacobian; it only
    # anchors ``nonlinear_rhs(0)`` so the linearisation point is an equilibrium.
    if include_gravity and system_model is not None:
        gravity_force_stab = _as_3vector(
            np.asarray(transformation_c_from_vsm, dtype=float)
            @ _force_gravity(system_model)
        )
    else:
        gravity_force_stab = np.zeros(3, dtype=float)

    warned_williams_force = False

    # --- Numerical hygiene of the finite-difference solves ----------------
    # The lateral moment derivatives (the yaw channel especially) are small
    # differences of large numbers divided by a small inertia; cold-started
    # gamma loops converge to slightly different circulation states whose
    # scatter is visible in those derivatives. Every perturbation solve is
    # therefore warm-started from the BASELINE (trim-state) circulation, and
    # its convergence flag is checked (with one cold retry) instead of being
    # silently ignored. Solvers whose ``solve`` does not accept a
    # ``gamma_distribution`` keyword (e.g. test mocks) opt out gracefully.
    try:
        _solve_accepts_gamma = (
            "gamma_distribution" in inspect.signature(solver.solve).parameters
        )
    except (TypeError, ValueError):
        _solve_accepts_gamma = False
    # Pre-seeding gamma_baseline makes the BASELINE solve itself start from the
    # caller's circulation (branch selection); the baseline's converged gamma
    # replaces it right after, so the FD warm-start behaviour is unchanged.
    gamma_baseline: np.ndarray | None = (
        None if gamma_seed is None else np.asarray(gamma_seed, dtype=float)
    )
    n_unconverged_solves = 0
    # Baseline force anchor for the tether (set after the baseline solve). The
    # stability model's independent fixed-length tether re-solve disagrees with
    # the trim's tether by a constant offset (frame/parametrisation), which
    # otherwise leaves a large ``f(0)`` residual. Subtracting the baseline
    # force_at_cg from the tether pins the trim to a true equilibrium of the
    # force model. It is a *constant* offset -> zero contribution to the
    # finite-difference Jacobian, so the eigenvalues are unchanged; only the
    # tether's radial-offset derivative (unchanged) enters ``A``.
    tether_baseline_anchor: np.ndarray | None = None

    # Straight-tether analytic position model: magnitude-preserving direction
    # change of a straight line to the fixed ground anchor. Central-differencing
    # it yields exactly -(T0/r)(1 - r_hat r_hat^T) — the spherical-pendulum
    # tangential stiffness — with ZERO radial stiffness, consistent with the
    # historical constant-force z model it falls back beside. Only used by the
    # position-augmented block when no Williams tether is available (or when a
    # Williams position solve fails).
    def _make_straight_tether_force():
        if frame_chain is None:
            return None
        T_wind_from_course = frame_chain["T_wind_from_course"]
        r0_wind = frame_chain["r0_wind"]
        tension0 = float(np.linalg.norm(f_tether))
        if tension0 <= 0.0 or float(np.linalg.norm(r0_wind)) <= 0.0:
            return None
        flip = np.asarray(transformation_c_from_vsm, dtype=float)

        def straight_force(offset_axes: np.ndarray) -> np.ndarray:
            offset_course = flip @ np.asarray(offset_axes, dtype=float).reshape(3)
            r_vec = r0_wind + T_wind_from_course @ offset_course
            r_norm = float(np.linalg.norm(r_vec))
            force_wind = -tension0 * r_vec / r_norm
            return flip @ (T_wind_from_course.T @ force_wind)

        return straight_force

    # Built whenever the frame chain allows (cheap, no solve): the 12-state
    # block AND the CG-form evaluator both use it as the fallback for lateral
    # attachment displacements. Reached only via use_position_fallback=True,
    # so the historical position_states=False paths are unaffected.
    straight_tether_force = _make_straight_tether_force()
    # Set True when a Williams position solve fails and the evaluation
    # degrades to a coarser tether model (recorded in the aug diagnostics).
    tether_position_solver_degraded = False

    def tether_force_for(
        offset_axes: np.ndarray, use_position_fallback: bool = False
    ) -> np.ndarray:
        nonlocal warned_williams_force, tether_position_solver_degraded
        if williams_fixed_length_force is not None:
            try:
                return williams_fixed_length_force(offset_axes)
            except RuntimeError as exc:
                if not warned_williams_force:
                    print(
                        "Warning: Williams fixed-length position perturbation "
                        "failed; falling back to a coarser tether model."
                    )
                    print(f"  reason: {exc}")
                    warned_williams_force = True
                tether_position_solver_degraded = True
        if use_position_fallback and straight_tether_force is not None:
            return straight_tether_force(offset_axes)
        return f_tether.copy()

    def eval_force_moment(
        delta_v: np.ndarray,
        omega_perturb: np.ndarray,
        radial_position_offset: float = 0.0,
        delta_roll_deg: float = 0.0,
        delta_pitch_deg: float = 0.0,
        delta_yaw_deg: float = 0.0,
        position_offset_axes: np.ndarray | None = None,
        position_feedback: bool = False,
        delta_course_rate: float = 0.0,
        with_contributions: bool = False,
    ) -> tuple:
        """Complete B-point right-hand side at a perturbed state.

        ``with_contributions=True`` appends a fourth element: the
        per-contributor breakdown of the B-point rows (forces in world
        components; moments about B — the tether carries none, it acts
        through B). Default ``False`` keeps the historical 3-tuple.

        ``delta_course_rate`` perturbs the relative turn rate ``chi_dot_turn``
        (see :func:`_omega_c_for`). It moves the course-frame transport rate
        ``Omega_C`` in the **transport inertial force** ``-m (Omega_C x v)``
        and its moment at the CG, and NOWHERE else: ``chi_dot_turn`` is a
        trajectory-curvature term (``a_n = -v_tau chi_dot_turn``), not a body
        rate, so it changes no apparent wind. The aerodynamic body rate, the
        gyroscopic couple and the centripetal CG-offset force all keep the
        FROZEN trim ``Omega_C`` plus the ``(p, q, r)`` state perturbation --
        the body rate equals ``Omega_C`` only AT TRIM. Default ``0.0`` keeps
        the behaviour of every historical column.

        ``position_offset_axes`` (3-vector, stability axes) is an additional
        kite position offset on top of ``radial_position_offset * e_radial``.
        ``position_feedback`` enables the position-augmented physics: the
        kite inflow is evaluated at the displaced height (wind shear) and a
        missing/failed Williams tether solve falls back to the straight-
        tether direction model instead of the constant baseline force. Both
        default off, so the historical 10-state columns are untouched.

        Returns ``(force, moment_about_B, res)`` in world components — the
        full right-hand side of Cayon & Schmehl Eqs. ``force_eom_B`` /
        ``moment_eom_B`` with the acceleration terms moved to the caller's
        mass-matrix solve: aerodynamic force and moment (the VSM moment is
        already about the reference point = tether attachment), tether force
        (no moment — it acts through the attachment), gravity force and its
        moment ``x_cg x F_g``, the transport inertial force
        ``-m Omega_C x v`` (transport rate frozen at trim) and its moment at
        the CG, the centripetal offset force ``-m omega x (omega x x_cg)``,
        and the gyroscopic moment ``-omega x I_B omega`` — the latter two
        with the full ``omega = Omega_C + omega_rel`` and the CG offset and
        inertia rotated with the perturbed attitude.
        """
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
        # Total kite position offset in the stability axes: the historical
        # radial (z) channel plus the optional lateral position offset.
        offset_axes_total = float(radial_position_offset) * axes.radial
        if position_offset_axes is not None:
            offset_axes_total = offset_axes_total + np.asarray(
                position_offset_axes, dtype=float
            ).reshape(3)
        # Wind shear at the displaced kite: the wind vector is U(h) along +x
        # of the wind frame (z up), so only the wind-frame z-component of the
        # offset changes the inflow. Off (historical frozen inflow) unless
        # the position-augmented block asks for it.
        delta_wind_axes = np.zeros(3, dtype=float)
        if position_feedback and frame_chain is not None:
            T_wc = frame_chain["T_wind_from_course"]
            flip = np.asarray(transformation_c_from_vsm, dtype=float)
            offset_wind = T_wc @ (flip @ offset_axes_total)
            h0 = float(frame_chain["r0_wind"][2])
            z0_wind = float(getattr(system_model.wind, "z0", 0.0) or 0.0)
            h_min = max(1e-6, 1.001 * z0_wind)
            h_pert = max(h0 + float(offset_wind[2]), h_min)
            h_base = max(h0, h_min)
            wind_model = system_model.wind
            d_speed = float(ca.DM(wind_model.speed_wind_at_height(h_pert))) - float(
                ca.DM(wind_model.speed_wind_at_height(h_base))
            )
            delta_wind_axes = flip @ (
                T_wc.T @ np.array([d_speed, 0.0, 0.0], dtype=float)
            )
        va_pert = va_trim + delta_wind_axes - delta_v
        umag = np.linalg.norm(va_pert)
        aoa_deg = np.rad2deg(np.arctan2(va_pert[2], va_pert[0]))
        beta_deg = np.rad2deg(np.arctan2(va_pert[1], np.hypot(va_pert[0], va_pert[2])))
        # BODY angular velocity — deliberately the FROZEN Omega_C, not the
        # chi_dot_turn-perturbed one. Omega_C is the rate of the COURSE FRAME;
        # the body rate coincides with it only AT TRIM (steady turn, body
        # following the velocity). Under perturbation the body rate is carried
        # by the (p, q, r) states alone: chi_dot_turn is a trajectory-curvature
        # (acceleration) term, a_n = -v_tau chi_dot_turn, and changes no
        # apparent wind, so it must not enter the aerodynamic body rate, the
        # gyroscopic couple, or the centripetal CG-offset force. It enters only
        # the transport inertial force below.
        omega_total = omega_c_world + omega_perturb
        omega_mag = np.linalg.norm(omega_total)
        omega_axis = omega_total / omega_mag if omega_mag > 1e-12 else axes.radial

        # rates_in_body_frame=True: use the axis as-is (world components); see
        # the note in solve_vsm_quasi_steady_trim.
        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_deg,
            side_slip=beta_deg,
            body_rates=omega_mag,
            body_axis=omega_axis,
            reference_point=reference_point,
            rates_in_body_frame=True,
        )
        nonlocal n_unconverged_solves
        if _solve_accepts_gamma and gamma_baseline is not None:
            res = solver.solve(working_body, gamma_distribution=gamma_baseline)
            if not bool(res.get("gamma_converged", True)):
                res = solver.solve(working_body)  # cold retry
        else:
            res = solver.solve(working_body)
        if not bool(res.get("gamma_converged", True)):
            n_unconverged_solves += 1
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

        # The kite flies along -e_course in the trim (VSM) frame (course/normal
        # axes flip), so a positive course-axis kite-velocity increment
        # REDUCES the tangential speed.
        speed_tangential_eff = float(speed_tangential) - float(
            np.dot(delta_v, axes.course)
        )
        # Radial distance change: the radial component of the total offset.
        # A pure lateral offset changes |r| only at second order.
        radial_offset_total = float(np.dot(offset_axes_total, axes.radial))
        distance_radial_eff = (
            distance_radial_trim + radial_offset_total
            if distance_radial_trim is not None
            else None
        )
        # Course-frame transport rate at THIS state. Its normal entry v_tau/r
        # is a kinematic identity and follows the perturbed speed and radius;
        # its radial entry carries the turn-rate perturbation. Used for the
        # transport inertial force ONLY (see _omega_c_for).
        _, omega_c_world_eff = _omega_c_for(
            delta_course_rate,
            speed_tangential_eff=speed_tangential_eff,
            distance_radial_eff=distance_radial_eff,
        )
        # The tether sees only the RADIAL part of the position offset unless
        # lateral feedback is on. With it off the tether translates WITH the
        # kite — a lateral displacement carries the attachment along, so the
        # force is unchanged and no re-solve is needed. That deliberately
        # removes the spherical-pendulum restoring term, whose natural home is
        # the slow (trajectory) subsystem, not this frozen-slow-state block.
        tether_offset = offset_axes_total
        if not tether_lateral_feedback:
            tether_offset = (
                float(np.dot(offset_axes_total, axes.radial)) * axes.radial
            )
        f_tether_eff = tether_force_for(
            tether_offset, use_position_fallback=position_feedback
        )
        if tether_baseline_anchor is not None:
            # Anchor the tether to the trim equilibrium: subtract the (constant)
            # baseline force residual so force(0) == 0. The radial-offset
            # dependence of ``tether_force_for`` is untouched, so d/dz is
            # preserved and the Jacobian / eigenvalues do not change.
            f_tether_eff = f_tether_eff - tether_baseline_anchor
        if speed_radial_trim is not None:
            # Exact transport inertial force of the paper's Eq.
            # ``qs_inertial_force``: -m (Omega_C x v), with the transport rate
            # frozen at the trajectory-set value (the fast-subsystem
            # convention) and the kite velocity carrying the perturbation.
            # The trim kite velocity in the VSM frame is
            # T_c_from_vsm @ [v_tau, 0, v_r] = -v_tau e_course + v_r e_radial
            # (course/normal axes flip); ``delta_v`` is the kite-velocity
            # increment in the same frame (``va_pert = va_trim - delta_v``).
            v_kite_eff = (
                -float(speed_tangential) * axes.course
                + float(speed_radial_trim) * axes.radial
                + delta_v
            )
            f_transport = -mass * np.cross(omega_c_world_eff, v_kite_eff)
        else:
            # Radial-only reduction (no system model to supply the radial
            # speed): -m Omega_red x v with Omega_red = -course_rate e_radial
            # and v = -v_eff e_course gives the normal component; the
            # great-circle part (v_tau/r along -e_normal in the VSM frame)
            # contributes the outward centrifugal radial component. Signs
            # follow the kite flying along -e_course.
            #
            # BOTH factors of m v_tau^2 / r are live, matching the full branch
            # (d/dv_tau = 2 m v_tau / r). This used to freeze one factor at the
            # trim speed, which halved that derivative -- harmless while the
            # full branch froze Omega_C entirely and got zero, but a silent
            # factor-2 discrepancy once the full branch became exact.
            f_transport = np.zeros(3, dtype=float)
            f_transport[1] = (
                -mass
                * speed_tangential_eff
                * (float(course_rate0) + float(delta_course_rate))
            )
            if distance_radial_eff is not None and distance_radial_eff > 0.0:
                f_transport[2] = (
                    mass
                    * speed_tangential_eff
                    * speed_tangential_eff
                    / distance_radial_eff
                )

        # Attitude-rotated CG offset and B-point inertia: the centripetal and
        # gyroscopic terms carry the full omega; the gravity and transport
        # moments arise because their forces act at the offset CG. The
        # centripetal force needs no separate moment — its moment about B is
        # the parallel-axis part of -omega x I_B omega (Jacobi identity).
        cg_world, inertia_b_world = _cg_and_inertia_b(
            delta_roll_deg, delta_pitch_deg, delta_yaw_deg
        )
        f_centripetal = -mass * np.cross(omega_total, np.cross(omega_total, cg_world))
        force = f_aero + f_tether_eff + gravity_force_stab + f_transport + f_centripetal
        moment_b = (
            moment_aero_at_ref
            + np.cross(cg_world, gravity_force_stab)
            + np.cross(cg_world, f_transport)
            - np.cross(omega_total, inertia_b_world @ omega_total)
        )
        if with_contributions:
            contributions = {
                "F_aero": np.asarray(f_aero, dtype=float),
                "F_tether": np.asarray(f_tether_eff, dtype=float),
                "F_gravity": np.asarray(gravity_force_stab, dtype=float),
                "F_transport": np.asarray(f_transport, dtype=float),
                "F_centripetal": np.asarray(f_centripetal, dtype=float),
                "M_aero_B": np.asarray(moment_aero_at_ref, dtype=float),
                "M_gravity_B": np.cross(cg_world, gravity_force_stab),
                "M_transport_B": np.cross(cg_world, f_transport),
                "M_gyro_B": -np.cross(omega_total, inertia_b_world @ omega_total),
                "cg_world": np.asarray(cg_world, dtype=float),
            }
            return force, moment_b, res, contributions
        return force, moment_b, res

    zero3 = np.zeros(3, dtype=float)
    eps_angle_rad = np.deg2rad(eps_angle_deg)

    # --- Baseline solve at the (unperturbed) trim state -------------------
    # Provides the warm-start circulation for every finite-difference solve
    # and the attached-flow check of the linearisation point itself: a trim
    # that has drifted onto a post-stall branch (multiple pitch equilibria
    # exist for rigged kites) yields sawtooth circulation states whose
    # derivatives are unreliable no matter how carefully they are differenced.
    _force_baseline, _, _res_baseline = eval_force_moment(zero3, zero3)
    # Anchor every subsequent tether evaluation to this (un-anchored) baseline
    # force so the trim is a true equilibrium of the force model (f(0) == 0;
    # the inertial force is the exact -m (Omega_C x v) when the system model
    # provides the radial speed, so no reduction residual remains).
    tether_baseline_anchor = np.asarray(_force_baseline, dtype=float)
    _gamma_res = _res_baseline.get("gamma_distribution")
    if _gamma_res is not None:
        gamma_baseline = np.asarray(_gamma_res, dtype=float)
    # Warm-started re-evaluation of the trim right-hand side. The anchor was
    # taken from the COLD baseline solve, while every finite-difference
    # evaluation warm-starts from its circulation — so the field the FD sees
    # retains a small warm/cold force residual at zero perturbation (plus the
    # by-design moment residual of a pinned-roll steering trim). Verification
    # needs the RHS the warm-started field actually has at the trim, in both
    # channels; A_full itself is unaffected (a constant cancels in the FD).
    _force0_warm, _moment0_warm, _ = eval_force_moment(zero3, zero3)
    stall_margin_min_deg = float("nan")
    n_stalled_panels: int | None = None
    _stall_onsets = _panel_stall_onsets_rad(working_body)
    _alpha_eff = _res_baseline.get("alpha_at_ac")
    if _alpha_eff is not None and _stall_onsets.size:
        _alpha_eff = np.ravel(np.asarray(_alpha_eff, dtype=float))
        if _alpha_eff.size == _stall_onsets.size:
            _margins_deg = np.rad2deg(_stall_onsets - _alpha_eff)
            stall_margin_min_deg = float(np.min(_margins_deg))
            n_stalled_panels = int(np.sum(_margins_deg <= 0.0))

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
        position_offset_axes: np.ndarray | None = None,
        position_feedback: bool = False,
        delta_course_rate: float = 0.0,
    ) -> np.ndarray:
        offset_plus = (
            None
            if position_offset_axes is None
            else np.asarray(position_offset_axes, dtype=float)
        )
        offset_minus = None if offset_plus is None else -offset_plus
        force_plus, moment_plus, _ = eval_force_moment(
            delta_v,
            omega_perturb,
            radial_position_offset,
            droll,
            dpitch,
            dyaw,
            position_offset_axes=offset_plus,
            position_feedback=position_feedback,
            delta_course_rate=delta_course_rate,
        )
        force_minus, moment_minus, _ = eval_force_moment(
            -delta_v,
            -omega_perturb,
            -radial_position_offset,
            -droll,
            -dpitch,
            -dyaw,
            position_offset_axes=offset_minus,
            position_feedback=position_feedback,
            delta_course_rate=-delta_course_rate,
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
    # cols = state perturbations in canonical ALL_STATE_NAMES order). The
    # lateral-velocity column is the sideslip derivative (va_pert picks up the
    # normal delta_v) plus its transport term; releasing it restores the
    # sideslip relief and Y_v damping a prescribed velocity direction removes.
    columns = {
        "u": central_diff_col(+eps_vel * axes.course, zero3, eps_vel),
        "v": central_diff_col(+eps_vel * axes.normal, zero3, eps_vel),
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

    # Coupled B-point mass matrix at the trim attitude, in the (course,
    # normal, radial) basis of the J rows and states. J_full already contains
    # every velocity-product, gravity, and tether derivative (they live in
    # ``eval_force_moment``), so the assembly is one joint 6x6 solve per
    # column plus the kinematic rows. At a converged trim the right-hand side
    # vanishes, so the attitude dependence of the mass matrix itself
    # contributes nothing to the Jacobian (d(M^-1)/dTheta multiplies RHS = 0).
    c_trim_world, inertia_b_trim_world = _cg_and_inertia_b()
    c_trim_axes = R_body @ c_trim_world
    inertia_b_trim_axes = R_body @ inertia_b_trim_world @ R_body.T
    _block_r_body = np.zeros((6, 6), dtype=float)
    _block_r_body[:3, :3] = R_body
    _block_r_body[3:, 3:] = R_body
    added_mass_trim_axes = _block_r_body @ _added_mass_world() @ _block_r_body.T
    mass_matrix_axes = (
        _bpoint_mass_matrix(mass, c_trim_axes, inertia_b_trim_axes)
        + added_mass_trim_axes
    )

    # Attitude-rate kinematics: (phi_dot, theta_dot, psi_dot) = K (p, q, r).
    # The generators of the yaw(e_r) pitch(e_n) roll(e_c) increment
    # composition tilt away from the frozen axes at nonzero trim pitch/yaw, so
    # K is the generator-matrix inverse rather than identity. Rate
    # perturbations are resolved in the frozen course basis, which already
    # absorbs the transport coupling of a body-frame description (Trevisi
    # 2024, Eq. 8.18) — no -Omega_C x deltaTheta term appears here.
    _roll_eff0, _pitch_eff0, _yaw_eff0 = _attitude_angles_deg(0.0, 0.0, 0.0)
    generator_matrix = _attitude_generator_matrix(
        roll_deg=_roll_eff0, pitch_deg=_pitch_eff0, yaw_deg=_yaw_eff0, axes=axes
    )
    kinematic_map = np.linalg.solve(generator_matrix, R_body.T)

    # ---- CG-form evaluator (attitude perturbations about the CG) ---------
    # One evaluation of the CG-referenced equations of motion (see
    # awetrim.aerodynamics.cg_eom for the equations and their derivation):
    #
    #     m v_dot_cg = F_a + F_t + m g - m Omega_C x v_cg
    #     I_cg omega_dot = M_a,cg + (r_B - r_cg) x F_t - omega x I_cg omega
    #
    # The attitude perturbation holds the CG fixed, so the tether attachment
    # swings, r_B = r_cg - c_att, and the tether is re-solved at the displaced
    # attachment (Williams fixed-length when available, straight-tether
    # fallback otherwise; the tether length is measured to B). Gravity and the
    # transport inertial force act at the CG and carry no moment; the tether
    # torques through its attachment arm -c_att x F_t; the aero moment is
    # transported from the solver reference with the same arm.
    #
    # Shares every closure of the B-form evaluator (solver warm start, tether
    # solve, trim frame quantities), so the two formulations cannot drift.
    # Approximations, both second order in the attitude for this kite: the
    # panel rotational inflow is evaluated about the fixed solver reference
    # (the ~|Omega_C| |delta_B| spurious inflow of a displaced kite, <0.1% of
    # va), and wind shear across the ~0.1 m/deg tangential swing of B is
    # neglected (its radial component is second order).
    def cg_eom_eval(
        delta_roll_deg: float = 0.0,
        delta_pitch_deg: float = 0.0,
        delta_yaw_deg: float = 0.0,
        delta_v_cg: np.ndarray | None = None,
        omega_perturb: np.ndarray | None = None,
        radial_position_offset: float = 0.0,
        delta_course_rate: float = 0.0,
    ) -> dict[str, Any]:
        """CG-form force/moment breakdown at a perturbed state.

        The STATE is ``(v_cg, attitude, omega, z)``: ``delta_v_cg`` perturbs
        the CG velocity (world components), ``omega_perturb`` the body rate,
        ``radial_position_offset`` translates the whole kite radially, and
        the attitude angles rotate it about the CG. Holding ``v_cg`` fixed
        under an attitude or rate perturbation means the ATTACHMENT velocity
        moves, ``v_B = v_cg - omega x c_att`` — the apparent wind carries
        that correction, which is what distinguishes these columns from the
        B-form's (where ``v_B`` is the state).

        ``delta_course_rate`` perturbs the relative turn rate
        ``chi_dot_turn`` with the same routing rule as
        :func:`eval_force_moment`: it moves ``Omega_C`` in the **transport
        inertial force** ``-m (Omega_C x v_cg)`` only — it is trajectory
        curvature, not a body rate, so it changes no apparent wind and must
        not enter the aerodynamic body rate (``omega_total``) or the
        gyroscopic couple. The CG form has no separate centripetal
        CG-offset force (it is absorbed in the transport term through
        ``v_cg = v_B + Omega_C x c``), so the force response to
        ``delta_course_rate`` differs from the B form's by
        ``-m dOmega x (Omega_C x c_att)`` — a few percent. A coordinated
        turn perturbation combines it with the matching body rate
        ``omega_perturb = -delta_course_rate * e_radial``. Default ``0.0``
        keeps every historical evaluation unchanged.

        Returns per-contributor forces (world/VSM components), their moments
        about the (material) CG, the net rows, the resulting ``accel_cg`` /
        ``omega_dot``, and the geometry needed to draw it: the displaced
        tether attachment ``r_B_world``, the CG-rotated LE/TE outlines, and
        the anchor-to-B tether length.
        """
        delta_v_cg = (
            np.zeros(3) if delta_v_cg is None else _as_3vector(delta_v_cg)
        )
        omega_perturb = (
            np.zeros(3) if omega_perturb is None else _as_3vector(omega_perturb)
        )
        # Aero: geometry rotated about B by the same combined-angle
        # composition as eval_force_moment. The CG-rotated kite is that
        # geometry translated by delta_B; a uniform panel inflow makes the
        # translation exact for the force, and the moment transports
        # analytically (the arm from the fixed CG to any material point of
        # either geometry is the same c_att).
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
        # Attitude-rotated CG arm and inertia (same composition as the mass
        # properties of the B form), needed before the aero solve because the
        # attachment velocity depends on it.
        c_att, inertia_b_att = _cg_and_inertia_b(
            delta_roll_deg, delta_pitch_deg, delta_yaw_deg
        )
        inertia_cg_att = inertia_b_att - _parallel_axis_term(mass, c_att)
        omega_total = omega_c_world + omega_perturb

        # v_cg is the STATE: frozen at its trim value (v_B_trim + Omega_C x
        # c_trim) plus the explicit perturbation. The attachment velocity
        # follows the attitude and rate, v_B = v_cg - omega x c_att, and the
        # apparent wind sees v_B.
        v_b_trim_world = -float(speed_tangential) * axes.course + (
            float(speed_radial_trim) * axes.radial
            if speed_radial_trim is not None
            else np.zeros(3)
        )
        v_cg_trim_world = v_b_trim_world + np.cross(omega_c_world, c_trim_world)
        v_cg_world = v_cg_trim_world + delta_v_cg
        v_b_eff = v_cg_world - np.cross(omega_total, c_att)
        delta_v_b = v_b_eff - v_b_trim_world
        va_pert = va_trim - delta_v_b
        umag = np.linalg.norm(va_pert)
        aoa_deg = np.rad2deg(np.arctan2(va_pert[2], va_pert[0]))
        beta_deg = np.rad2deg(
            np.arctan2(va_pert[1], np.hypot(va_pert[0], va_pert[2]))
        )
        omega_mag = np.linalg.norm(omega_total)
        omega_axis = (
            omega_total / omega_mag if omega_mag > 1e-12 else axes.radial
        )
        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_deg,
            side_slip=beta_deg,
            body_rates=omega_mag,
            body_axis=omega_axis,
            reference_point=reference_point,
            rates_in_body_frame=True,
        )
        if _solve_accepts_gamma and gamma_baseline is not None:
            res = solver.solve(working_body, gamma_distribution=gamma_baseline)
            if not bool(res.get("gamma_converged", True)):
                res = solver.solve(working_body)
        else:
            res = solver.solve(working_body)
        f_aero = np.array(
            [float(res.get(k, 0.0)) for k in ("Fx", "Fy", "Fz")], dtype=float
        )
        q_inf = 0.5 * float(solver.rho) * umag**2
        denom = q_inf * projected_area * max_chord if projected_area > 0 else 1.0
        moment_aero_at_ref = (
            np.array(
                [float(res.get(k, 0.0)) for k in ("cmx", "cmy", "cmz")],
                dtype=float,
            )
            * denom
        )

        # B displacement of a rotation about the CG plus the radial
        # translation of the whole kite (the z state moves CG and B alike).
        radial_shift = float(radial_position_offset) * axes.radial
        delta_b_world = (c_trim_world - c_att) + radial_shift
        r_cg_world = _as_3vector(reference_point) + c_trim_world + radial_shift
        r_b_world = r_cg_world - c_att  # == reference_point + delta_b_world

        # Aero moment about the fixed CG: the solver reports it about the
        # (fixed) reference point of the UNtranslated rotated kite; adding
        # the translation and moving the arm to the CG collapses to one term.
        moment_aero_cg = moment_aero_at_ref - np.cross(c_att, f_aero)

        # Tether at the displaced attachment: full 3-vector offset, length
        # measured to B (Williams fixed-length re-solve when available). The
        # degraded flag is reset around the call so the returned model tag is
        # PER-EVALUATION — a sweep can mark exactly which points fell back.
        nonlocal tether_position_solver_degraded
        _degraded_before = tether_position_solver_degraded
        tether_position_solver_degraded = False
        f_teth = tether_force_for(
            R_body @ delta_b_world, use_position_fallback=True
        )
        if tether_position_solver_degraded:
            tether_model_used = (
                "straight_fallback"
                if straight_tether_force is not None
                else "constant_trim_force"
            )
        elif williams_fixed_length_force is not None:
            tether_model_used = "williams_fixed_length"
        elif straight_tether_force is not None:
            tether_model_used = "straight_tether_analytic"
        else:
            tether_model_used = "constant_trim_force"
        tether_position_solver_degraded = (
            tether_position_solver_degraded or _degraded_before
        )
        moment_teth_cg = np.cross(-c_att, f_teth)

        # Transport inertial force at the CG: -m Omega_C x v_cg with the
        # transport rate treated exactly as in the B form — the kinematic
        # normal entry v_tau/r follows the perturbed course speed and radius
        # (transport_rate_follows_states), the radial (turn-rate) entry stays
        # frozen. At trim this contains the B-form's centripetal offset force
        # exactly (v_cg = v_B + Omega_C x c). Gyroscopic couple with the full
        # omega and the CG inertia — no parallel-axis content here.
        _, omega_c_world_cg = _omega_c_for(
            float(delta_course_rate),
            speed_tangential_eff=float(speed_tangential)
            - float(np.dot(delta_v_b, axes.course)),
            distance_radial_eff=(
                distance_radial_trim + float(radial_position_offset)
                if distance_radial_trim is not None
                else None
            ),
        )
        f_transport_cg = -mass * np.cross(omega_c_world_cg, v_cg_world)
        moment_gyro_cg = -np.cross(omega_total, inertia_cg_att @ omega_total)

        force_net = f_aero + f_teth + gravity_force_stab + f_transport_cg
        moment_net = moment_aero_cg + moment_teth_cg + moment_gyro_cg
        # CG-rotated outline for plotting: B-rotated points + delta_B.
        kite_le = np.array(
            [
                np.asarray(s.LE_point, dtype=float) + delta_b_world
                for w in working_body.wings
                for s in w.sections
            ]
        )
        kite_te = np.array(
            [
                np.asarray(s.TE_point, dtype=float) + delta_b_world
                for w in working_body.wings
                for s in w.sections
            ]
        )
        r_anchor = (
            _as_3vector(reference_point)
            - float(distance_radial_trim) * axes.radial
            if distance_radial_trim is not None
            else None
        )
        return {
            "F_aero": f_aero,
            "M_aero_cg": moment_aero_cg,
            "F_tether": f_teth,
            "M_tether_cg": moment_teth_cg,
            "F_gravity": np.array(gravity_force_stab, dtype=float),
            "F_transport": f_transport_cg,
            "M_gyro_cg": moment_gyro_cg,
            "force_net": force_net,
            "moment_cg_net": moment_net,
            "accel_cg": force_net / mass,
            "omega_dot": np.linalg.solve(inertia_cg_att, moment_net),
            "inertia_cg_att": inertia_cg_att,
            "v_cg_world": np.array(v_cg_world, dtype=float),
            "delta_v_b_world": np.array(delta_v_b, dtype=float),
            "omega_total_world": np.array(omega_total, dtype=float),
            "r_cg_world": r_cg_world,
            "r_B_world": r_b_world,
            "delta_B_world": delta_b_world,
            "c_att_world": np.array(c_att, dtype=float),
            "r_anchor_world": r_anchor,
            "tether_length_to_B": (
                float(np.linalg.norm(r_b_world - r_anchor))
                if r_anchor is not None
                else float("nan")
            ),
            "kite_LE_world": kite_le,
            "kite_TE_world": kite_te,
            "axes_course": np.array(axes.course, dtype=float),
            "axes_normal": np.array(axes.normal, dtype=float),
            "axes_radial": np.array(axes.radial, dtype=float),
            "gamma_converged": bool(res.get("gamma_converged", True)),
            "tether_model_used": tether_model_used,
        }

    _, A_full = _build_state_space(
        J_full,
        ALL_STATE_NAMES,
        mass_matrix=mass_matrix_axes,
        kinematic_map=kinematic_map,
    )
    eig_full, vec_full = _eig_block(A_full)

    # ---- Optional position-augmented 12-state block ----------------------
    # Three extra FD columns on top of the 10-state machinery: x, y (lateral
    # positions) and a recomputed z that carries the same position feedback
    # (wind shear at the displaced height; straight-tether fallback when no
    # Williams solve exists). J_full/A_full above stay the historical
    # shear-free objects; the augmented block is internally consistent with
    # ``nonlinear_rhs_aug``.
    if position_states:
        lateral_eps = float(eps_position)
        if (
            williams_fixed_length_force is not None
            and "williams_tether_length" in trim_result
        ):
            # A lateral offset eps grows |r_kite| by ~eps^2/(2 r): keep that
            # growth within a quarter of the tether slack so the fixed-length
            # solve stays feasible (the radial clamp above does not cover the
            # lateral steps).
            _r_norm_aug = float(
                np.linalg.norm(_as_3vector(trim_result.get("r_kite", [0.0, 0.0, 0.0])))
            )
            _slack_aug = float(trim_result["williams_tether_length"]) - _r_norm_aug
            if _slack_aug > 1e-8 and _r_norm_aug > 0.0:
                lateral_eps = min(
                    lateral_eps, math.sqrt(0.5 * _r_norm_aug * _slack_aug)
                )
            elif _slack_aug <= 1e-8:
                lateral_eps = min(lateral_eps, 1e-3)
            lateral_eps = max(lateral_eps, 1e-7)

        columns_aug = dict(columns)
        columns_aug["z"] = central_diff_col(
            zero3,
            zero3,
            radial_eps,
            radial_position_offset=radial_eps,
            position_feedback=True,
        )
        columns_aug["x"] = central_diff_col(
            zero3,
            zero3,
            lateral_eps,
            position_offset_axes=lateral_eps * axes.course,
            position_feedback=True,
        )
        columns_aug["y"] = central_diff_col(
            zero3,
            zero3,
            lateral_eps,
            position_offset_axes=lateral_eps * axes.normal,
            position_feedback=True,
        )
        J_aug = np.column_stack([columns_aug[name] for name in AUG_STATE_NAMES])
        _, A_aug = _build_state_space(
            J_aug,
            AUG_STATE_NAMES,
            mass_matrix=mass_matrix_axes,
            kinematic_map=kinematic_map,
            full_state_names=AUG_STATE_NAMES,
            position_transport=omega_c_axes,
        )
        eig_aug, vec_aug = _eig_block(A_aug)
        if williams_fixed_length_force is not None:
            tether_position_model_aug = "williams_fixed_length"
            if tether_position_solver_degraded:
                tether_position_model_aug += "_degraded"
        elif straight_tether_force is not None:
            tether_position_model_aug = "straight_tether_analytic"
        else:
            tether_position_model_aug = "constant_trim_force"
        if not tether_lateral_feedback:
            # The name must say so: with the lateral response suppressed the
            # model above still governs the RADIAL channel, but x and y carry
            # no tether force at all.
            tether_position_model_aug += "_radial_only"

    # Kite velocity at trim in the stability axes — input to the co-rotating
    # state transform (course axes carried by the body, see
    # :func:`corotating_state_transform`). Prefer the trim's own world-frame
    # kite velocity; fall back to the tangential speed along the course axis.
    _v_kite_world = trim_result.get("kite_vel_world")
    if _v_kite_world is not None:
        v_kite_trim_axes = R_body @ _as_3vector(_v_kite_world)
    elif system_model is not None and hasattr(system_model, "velocity_kite"):
        v_kite_trim_axes = R_body @ _as_3vector(
            np.asarray(transformation_c_from_vsm, dtype=float)
            @ np.asarray(system_model.velocity_kite, dtype=float).reshape(3)
        )
    else:
        v_kite_trim_axes = np.array([float(speed_tangential), 0.0, 0.0])

    # ---- Optional course-rate (algebraic chi_dot_turn) block --------------
    # Replaces the lateral velocity state by the relative turn rate. The
    # course frame is DEFINED by the velocity direction, so v_n == 0 for all
    # time and its frame-relative rate vanishes identically: the normal row of
    # the B-point acceleration is not integrated but set to zero, and that
    # equation determines chi_dot_turn.
    #
    # NOTE the outcome (see the keyword's docstring): F_Omega is exactly zero
    # on every differential row, so this reduces to A_full with v pinned. The
    # rank-1 term below is kept because it is the general index-1 form, it is
    # what makes that cancellation VERIFIABLE rather than assumed, and it
    # ceases to vanish under include_added_mass=True.
    #
    # Because eval_force_moment already carries the transport inertial force
    # -m (Omega_C x v) -- whose radial Omega_C entry IS -chi_dot_turn -- the
    # closure is simply
    #
    #     e1' M^-1 ( J_9 dx9 + J_chi dchidot ) = 0,
    #
    # i.e. the frame-relative normal acceleration vanishes. Note this form
    # references no course-angle sign convention at all: the only convention
    # involved is the one shared by ``course_rate0``, the trim solver and
    # Omega_C, so the reported ``chi_dot_turn`` is in the trim vector's
    # ``timeder_angle_course`` convention. (Beware: the post-processing
    # helper ``course_vs_yaw_ratio`` defines delta_chi = dv/u0, which is
    # rotation about the OPPOSITE radial sense -- magnitudes agree, signs do
    # not. Do not mix the two.)
    J_course_rate: np.ndarray | None = None
    A_chi: np.ndarray | None = None
    eig_chi: np.ndarray | None = None
    vec_chi: np.ndarray | None = None
    chi_turn_gain_row: np.ndarray | None = None
    A_chi10: np.ndarray | None = None
    eig_chi10: np.ndarray | None = None
    vec_chi10: np.ndarray | None = None
    chi10_gain_row: np.ndarray | None = None
    chi_turn_closure_singular = False
    chi_turn_denominator = float("nan")
    if course_rate_state:
        J_course_rate = central_diff_col(
            zero3,
            zero3,
            float(eps_course_rate),
            delta_course_rate=float(eps_course_rate),
        )
        J_9 = np.column_stack([columns[name] for name in CHI_STATE_NAMES])
        accel_9 = np.linalg.solve(mass_matrix_axes, J_9)  # 6 x 9
        accel_chi = np.linalg.solve(mass_matrix_axes, J_course_rate)  # 6
        # d(a_normal)/d(chi_dot_turn); ~ -v_tau from the transport term alone.
        chi_turn_denominator = float(accel_chi[1])
        _scale = max(1.0, float(abs(v_kite_trim_axes[0])))
        if abs(chi_turn_denominator) < 1e-9 * _scale:
            chi_turn_closure_singular = True
            chi_turn_gain_row = np.zeros(len(CHI_STATE_NAMES), dtype=float)
        else:
            chi_turn_gain_row = -accel_9[1, :] / chi_turn_denominator
        _, A_chi = _build_state_space(
            J_9,
            CHI_STATE_NAMES,
            mass_matrix=mass_matrix_axes,
            kinematic_map=kinematic_map,
            full_state_names=CHI_STATE_NAMES,
        )
        # Substitute dchidot = gain_row @ dx9 back into the dynamic rows. The
        # kinematic rows (z, phi, theta, psi) carry no force term and are
        # untouched.
        for _i, _s in enumerate(CHI_STATE_NAMES):
            if _s in _FORCE_OUTPUT_ROW:
                A_chi[_i, :] += accel_chi[_FORCE_OUTPUT_ROW[_s]] * chi_turn_gain_row
            elif _s in _MOMENT_OUTPUT_ROW:
                A_chi[_i, :] += accel_chi[_MOMENT_OUTPUT_ROW[_s]] * chi_turn_gain_row
        eig_chi, vec_chi = _eig_block(A_chi)

        # ---- Same closure, v RETAINED (10 differential + 1 algebraic) -----
        # Pinning v deletes the sideslip DOF outright. Keeping it reads the
        # SAME closure as what it physically is — "the course frame FOLLOWS
        # the velocity direction", i.e. delta_v_dot = 0 rather than
        # delta_v = 0. With one normal momentum equation and two unknowns
        # (v_dot, chi_dot_turn) there is no third option: either the frame is
        # frozen and the equation integrates v_dot (the baseline 10-state
        # block), or the frame follows and the equation determines
        # chi_dot_turn. This is the latter WITHOUT forbidding an initial
        # normal offset — the frame carries it along instead.
        #
        # Consequences, all structural: the v row is identically zero, so the
        # matrix is block upper triangular in (others | v) and
        #
        #     spec(A_chi10) = spec(A_chi) + one zero eigenvalue,
        #
        # the zero being the neutral constant-sideslip mode. The v COLUMN is
        # retained, so a nonzero v still drives the other nine equations
        # aerodynamically — that is what this variant buys over the pinned
        # block, and it shows up in the eigenVECTORS, not the eigenvalues.
        chi10_gain_row = np.zeros(len(ALL_STATE_NAMES), dtype=float)
        if not chi_turn_closure_singular:
            chi10_gain_row = -A_full[_I_V_STATE, :] / chi_turn_denominator
        A_chi10 = np.array(A_full, dtype=float, copy=True)
        A_chi10[_I_V_STATE, :] = 0.0
        for _i, _s in enumerate(ALL_STATE_NAMES):
            if _s == "v":
                continue
            if _s in _FORCE_OUTPUT_ROW:
                A_chi10[_i, :] += accel_chi[_FORCE_OUTPUT_ROW[_s]] * chi10_gain_row
            elif _s in _MOMENT_OUTPUT_ROW:
                A_chi10[_i, :] += accel_chi[_MOMENT_OUTPUT_ROW[_s]] * chi10_gain_row
        eig_chi10, vec_chi10 = _eig_block(A_chi10)

    # ---- Turn-rate OUTPUT row of the primary 10-state block ---------------
    # chi_dot_turn is not a state of that block, but it IS a linear functional
    # of it. Reading the normal equation a_n = -v_tau chi_dot_turn on the
    # FREE-v block instead of the pinned one gives
    #
    #     delta_chi_dot_turn = -(normal acceleration row) / G_Omega,
    #
    # i.e. the closure of the course-rate block above applied to the un-pinned
    # row. Equivalently delta_chi_dot_turn = delta_v_dot / u0, so on an
    # eigenmode it equals lambda * delta_chi — the mode's course excursion
    # times its own rate. This costs nothing (one row of A_full is already
    # M^-1 J's normal row), so it is always emitted, and it lets the 10-state
    # and 9-state blocks report the same quantity computed the same way. When
    # the state set pins v the row IS ``chi_turn_gain_row`` restricted to the
    # retained states.
    #
    # Convention: the trim vector's ``timeder_angle_course`` sense, matching
    # ``chi_turn_gain_row`` — NOT the opposite radial sense of the
    # post-processing helper ``course_vs_yaw_ratio`` (delta_chi = dv/u0).
    # Magnitudes agree, signs do not; report magnitudes or fix one convention.
    #
    # ``chi_dot_gc`` sits in the frozen radial entry of Omega_C and does not
    # move with the fast states (see ``transport_rate_follows_states``), so
    # within this model delta_chi_dot == delta_chi_dot_turn.
    #
    # G_Omega is the FD-measured ``chi_turn_denominator`` when the course-rate
    # block ran, else the exact kinematic u0 = -v_tau. The two agree to ~1e-12
    # for the rigid coupled mass matrix, but NOT under include_added_mass=True
    # — there only the measured one is right, which is what
    # ``chi_turn_denominator_source`` records.
    chi_turn_denominator_full = float(v_kite_trim_axes[0])
    chi_turn_denominator_source = "kinematic"
    if course_rate_state and not chi_turn_closure_singular:
        chi_turn_denominator_full = float(chi_turn_denominator)
        chi_turn_denominator_source = "finite-difference"
    chi_turn_gain_row_full: np.ndarray | None = None
    if abs(chi_turn_denominator_full) > 1e-9 * max(1.0, abs(float(speed_tangential))):
        chi_turn_gain_row_full = (
            -A_full[ALL_STATE_NAMES.index("v"), :] / chi_turn_denominator_full
        )

    def mass_matrix_world_fn(
        droll_deg: float = 0.0, dpitch_deg: float = 0.0, dyaw_deg: float = 0.0
    ) -> np.ndarray:
        """Coupled 6x6 mass matrix (rigid + added mass) at the perturbed
        attitude, in world components — the matrix ``nonlinear_rhs`` solves.
        Exposed so verification can form the non-equilibrium FD artifact
        ``d(M^-1)/dTheta @ rhs_world_at_trim`` exactly."""
        cg_world, inertia_b_world = _cg_and_inertia_b(droll_deg, dpitch_deg, dyaw_deg)
        return _bpoint_mass_matrix(mass, cg_world, inertia_b_world) + _added_mass_world(
            droll_deg, dpitch_deg, dyaw_deg
        )

    def nonlinear_rhs_full(delta_state: np.ndarray) -> dict[str, np.ndarray]:
        """One evaluation of the nonlinear fast subsystem with its internals.

        Returns ``{"xdot", "rhs_world", "accel_world", "mass_matrix_world"}``
        for the perturbed state: the state derivative (see
        :func:`nonlinear_rhs`), the B-point right-hand side ``(force,
        moment)`` in world components, the joint 6-DOF acceleration it
        produces, and the perturbed-attitude mass matrix that links them
        (``rhs_world = mass_matrix_world @ accel_world``). Verification can
        difference ``rhs_world`` through the *trim* mass matrix — the
        documented ``A_full`` assembly on independent evaluations — which is
        free of the two attitude-column FD artifacts of the raw field: the
        constant-residual term ``d(M^-1)/dTheta @ rhs(0)`` and the O(eps^2)
        curvature of ``M(Theta)^-1`` (visible once the strongly anisotropic
        added-mass block rotates with attitude).
        """
        dx = np.asarray(delta_state, dtype=float).reshape(10)
        du, dv, dw, dz, dphi, dtheta, dpsi, dp, dq, dr = dx
        delta_v = du * axes.course + dv * axes.normal + dw * axes.radial
        omega_perturb = dp * axes.course + dq * axes.normal + dr * axes.radial
        droll_deg = float(np.rad2deg(dphi))
        dpitch_deg = float(np.rad2deg(dtheta))
        dyaw_deg = float(np.rad2deg(dpsi))
        f_world, m_world, _ = eval_force_moment(
            delta_v,
            omega_perturb,
            radial_position_offset=float(dz),
            delta_roll_deg=droll_deg,
            delta_pitch_deg=dpitch_deg,
            delta_yaw_deg=dyaw_deg,
        )
        # Joint accelerations from the coupled mass matrix at the perturbed
        # attitude (world components, projected onto the axes basis below).
        rhs_world = np.concatenate([f_world, m_world])
        mass_matrix_world = mass_matrix_world_fn(droll_deg, dpitch_deg, dyaw_deg)
        accel = np.linalg.solve(mass_matrix_world, rhs_world)
        v_dot_axes = R_body @ accel[:3]
        rate_dot_axes = R_body @ accel[3:]
        # Exact attitude kinematics: the rate perturbation (frozen-basis
        # components) equals the rotation-vector rate of the attitude
        # perturbation, mapped to Euler-increment rates by the generator
        # matrix at the perturbed attitude.
        roll_eff, pitch_eff, yaw_eff = _attitude_angles_deg(
            droll_deg, dpitch_deg, dyaw_deg
        )
        euler_rates = np.linalg.solve(
            _attitude_generator_matrix(
                roll_deg=roll_eff, pitch_deg=pitch_eff, yaw_deg=yaw_eff, axes=axes
            ),
            omega_perturb,
        )
        xdot = np.array(
            [
                v_dot_axes[0],  # u_dot (course)
                v_dot_axes[1],  # v_dot (normal)
                v_dot_axes[2],  # w_dot (radial)
                dw,  # z_dot = w
                euler_rates[0],  # phi_dot
                euler_rates[1],  # theta_dot
                euler_rates[2],  # psi_dot
                rate_dot_axes[0],  # p_dot
                rate_dot_axes[1],  # q_dot
                rate_dot_axes[2],  # r_dot
            ],
            dtype=float,
        )
        return {
            "xdot": xdot,
            "rhs_world": rhs_world,
            "accel_world": accel,
            "mass_matrix_world": mass_matrix_world,
        }

    def nonlinear_rhs(delta_state: np.ndarray) -> np.ndarray:
        """Nonlinear fast-subsystem vector field ``xdot = f(x)`` at this trim.

        ``delta_state`` is the perturbation of the 10 states in
        :data:`ALL_STATE_NAMES` order ``(u, v, w, z, phi, theta, psi, p, q,
        r)`` about the trim (angles in rad, rates in rad/s, velocities in m/s,
        ``z`` in m); the return is ``xdot`` in the same order and
        units-per-second.

        This assembles the state derivative *directly* from the governing
        B-point equations: the right-hand side from :func:`eval_force_moment`,
        one joint solve of the coupled 6x6 mass matrix (built at the
        *perturbed* attitude, so the CG offset, I_B, and added-mass tensor
        rotate with the state) for the translational and angular
        accelerations, the kinematic identity ``z_dot = w``, and the exact
        attitude kinematics through the increment-generator matrix evaluated
        at the perturbed attitude. It shares no code with
        :func:`_build_state_space`, so:

        * ``nonlinear_rhs(zeros(10))`` is the trim **equilibrium residual**. A
          trim solved with the gyroscopic couple (``inertia_cg``), the
          centripetal CG-offset force, and the full ``Omega_C`` aerodynamic
          body rates shares this evaluation's physics, so all channels should
          vanish to within the trim solver convergence (a pinned-roll
          bridle-steering trim keeps its steering-line reaction moment), and
        * central-differencing this field is an independent cross-check of
          ``A_full`` — after removing the attitude-column artifacts of the
          attitude-dependent mass matrix; ``nonlinear_rhs_full`` exposes the
          per-evaluation right-hand side for an artifact-free comparison.

        Provided for verification and for time-domain integration of the fast
        subsystem; not used to build any of the returned matrices.
        """
        return nonlinear_rhs_full(delta_state)["xdot"]

    def nonlinear_rhs_aug_full(delta_state: np.ndarray) -> dict[str, np.ndarray]:
        """Position-augmented analogue of :func:`nonlinear_rhs_full`.

        ``delta_state`` is the 12-vector in :data:`AUG_STATE_NAMES` order
        (``x``, ``y`` appended). The right-hand side carries the position
        feedback of the augmented block — tether at the displaced ``r_kite``
        (with the straight-tether fallback) and wind shear at the displaced
        height — and the position kinematics carry the frozen-axes transport
        ``delta_r_dot = delta_v - Omega_C x delta_r``. Like
        :func:`nonlinear_rhs_full` it shares no code with
        :func:`_build_state_space`: ``f(0)`` is the trim equilibrium residual
        and central-differencing it independently cross-checks ``A_aug``.
        Only exposed in the result dict when ``position_states=True``.
        """
        dx = np.asarray(delta_state, dtype=float).reshape(12)
        du, dv, dw, dz, dphi, dtheta, dpsi, dp, dq, dr, dxp, dyp = dx
        delta_v = du * axes.course + dv * axes.normal + dw * axes.radial
        omega_perturb = dp * axes.course + dq * axes.normal + dr * axes.radial
        droll_deg = float(np.rad2deg(dphi))
        dpitch_deg = float(np.rad2deg(dtheta))
        dyaw_deg = float(np.rad2deg(dpsi))
        f_world, m_world, _ = eval_force_moment(
            delta_v,
            omega_perturb,
            radial_position_offset=float(dz),
            delta_roll_deg=droll_deg,
            delta_pitch_deg=dpitch_deg,
            delta_yaw_deg=dyaw_deg,
            position_offset_axes=dxp * axes.course + dyp * axes.normal,
            position_feedback=True,
        )
        rhs_world = np.concatenate([f_world, m_world])
        mass_matrix_world = mass_matrix_world_fn(droll_deg, dpitch_deg, dyaw_deg)
        accel = np.linalg.solve(mass_matrix_world, rhs_world)
        v_dot_axes = R_body @ accel[:3]
        rate_dot_axes = R_body @ accel[3:]
        roll_eff, pitch_eff, yaw_eff = _attitude_angles_deg(
            droll_deg, dpitch_deg, dyaw_deg
        )
        euler_rates = np.linalg.solve(
            _attitude_generator_matrix(
                roll_deg=roll_eff, pitch_deg=pitch_eff, yaw_deg=yaw_eff, axes=axes
            ),
            omega_perturb,
        )
        # Position kinematics with the frame transport: delta_r = (x, y, z)
        # in the (course, normal, radial) component ordering.
        delta_r = np.array([dxp, dyp, dz], dtype=float)
        r_dot = np.array([du, dv, dw], dtype=float) - np.cross(
            np.asarray(omega_c_axes, dtype=float), delta_r
        )
        xdot = np.array(
            [
                v_dot_axes[0],  # u_dot (course)
                v_dot_axes[1],  # v_dot (normal)
                v_dot_axes[2],  # w_dot (radial)
                r_dot[2],  # z_dot = w - (Omega_C x delta_r)_radial
                euler_rates[0],  # phi_dot
                euler_rates[1],  # theta_dot
                euler_rates[2],  # psi_dot
                rate_dot_axes[0],  # p_dot
                rate_dot_axes[1],  # q_dot
                rate_dot_axes[2],  # r_dot
                r_dot[0],  # x_dot = u - (Omega_C x delta_r)_course
                r_dot[1],  # y_dot = v - (Omega_C x delta_r)_normal
            ],
            dtype=float,
        )
        return {
            "xdot": xdot,
            "rhs_world": rhs_world,
            "accel_world": accel,
            "mass_matrix_world": mass_matrix_world,
        }

    def nonlinear_rhs_aug(delta_state: np.ndarray) -> np.ndarray:
        """``xdot = f(x)`` of the position-augmented 12-state fast subsystem
        (see :func:`nonlinear_rhs_aug_full`)."""
        return nonlinear_rhs_aug_full(delta_state)["xdot"]

    def nonlinear_rhs_chi_full(
        delta_state: np.ndarray,
        *,
        tol: float = 1e-10,
        max_iter: int = 12,
    ) -> dict[str, np.ndarray]:
        """Course-rate analogue of :func:`nonlinear_rhs_full` (9 states).

        ``delta_state`` is the 9-vector in :data:`CHI_STATE_NAMES` order
        ``(u, w, z, phi, theta, psi, p, q, r)`` — the canonical set with the
        lateral velocity ``v`` removed, since the course frame is defined by
        the velocity direction and carries no normal velocity component.

        At each evaluation the relative turn rate ``delta_chi_dot_turn`` is
        solved (scalar Newton on the course-rate perturbation) so that the
        frame-relative normal acceleration vanishes,

            [M(Theta)^-1 rhs(dx, dchidot)]_normal = 0,

        which is the paper's ``a_n = -v_tau chi_dot_turn`` written in the
        rotating frame where ``eval_force_moment`` already carries the
        transport force ``-m (Omega_C x v)``. The converged ``dchidot`` feeds
        back into ``Omega_C`` for the remaining rows, so this field shares no
        code with the :data:`A_chi` assembly and central-differencing it is an
        independent cross-check. ``f(0)`` is the trim equilibrium residual.

        Returns ``{"xdot", "rhs_world", "accel_world", "mass_matrix_world",
        "delta_chi_dot_turn", "normal_residual", "n_iter"}``.
        """
        dx = np.asarray(delta_state, dtype=float).reshape(9)
        du, dw, dz, dphi, dtheta, dpsi, dp, dq, dr = dx
        # No normal component: v_n == 0 is the defining property of the frame.
        delta_v = du * axes.course + dw * axes.radial
        omega_perturb = dp * axes.course + dq * axes.normal + dr * axes.radial
        droll_deg = float(np.rad2deg(dphi))
        dpitch_deg = float(np.rad2deg(dtheta))
        dyaw_deg = float(np.rad2deg(dpsi))
        mass_matrix_world = mass_matrix_world_fn(droll_deg, dpitch_deg, dyaw_deg)

        def _evaluate(dchidot: float) -> tuple[np.ndarray, np.ndarray, float]:
            f_world, m_world, _ = eval_force_moment(
                delta_v,
                omega_perturb,
                radial_position_offset=float(dz),
                delta_roll_deg=droll_deg,
                delta_pitch_deg=dpitch_deg,
                delta_yaw_deg=dyaw_deg,
                delta_course_rate=float(dchidot),
            )
            rhs = np.concatenate([f_world, m_world])
            acc = np.linalg.solve(mass_matrix_world, rhs)
            # Normal component of the frame-relative translational acceleration.
            return rhs, acc, float((R_body @ acc[:3])[1])

        # Newton on the scalar residual. The map is close to affine (the turn
        # rate enters through Omega_C linearly plus a mild aerodynamic
        # body-rate term), so the secant slope is refreshed only on the first
        # step and reused unless it degrades.
        dchidot = 0.0
        rhs_world, accel, residual = _evaluate(dchidot)
        residual0 = residual
        step = float(eps_course_rate)
        _, _, residual_probe = _evaluate(step)
        slope = (residual_probe - residual0) / step
        n_iter = 1
        if abs(slope) > 0.0:
            for _ in range(max_iter):
                if abs(residual) <= tol * max(1.0, abs(residual0)):
                    break
                dchidot -= residual / slope
                rhs_world, accel, residual = _evaluate(dchidot)
                n_iter += 1
        v_dot_axes = R_body @ accel[:3]
        rate_dot_axes = R_body @ accel[3:]
        roll_eff, pitch_eff, yaw_eff = _attitude_angles_deg(
            droll_deg, dpitch_deg, dyaw_deg
        )
        euler_rates = np.linalg.solve(
            _attitude_generator_matrix(
                roll_deg=roll_eff, pitch_deg=pitch_eff, yaw_deg=yaw_eff, axes=axes
            ),
            omega_perturb,
        )
        xdot = np.array(
            [
                v_dot_axes[0],  # u_dot (course)
                v_dot_axes[2],  # w_dot (radial)
                dw,  # z_dot = w
                euler_rates[0],  # phi_dot
                euler_rates[1],  # theta_dot
                euler_rates[2],  # psi_dot
                rate_dot_axes[0],  # p_dot
                rate_dot_axes[1],  # q_dot
                rate_dot_axes[2],  # r_dot
            ],
            dtype=float,
        )
        return {
            "xdot": xdot,
            "rhs_world": rhs_world,
            "accel_world": accel,
            "mass_matrix_world": mass_matrix_world,
            "delta_chi_dot_turn": float(dchidot),
            "normal_residual": float(residual),
            "n_iter": int(n_iter),
        }

    def nonlinear_rhs_chi(delta_state: np.ndarray) -> np.ndarray:
        """``xdot = f(x)`` of the 9-state course-rate fast subsystem
        (see :func:`nonlinear_rhs_chi_full`)."""
        return nonlinear_rhs_chi_full(delta_state)["xdot"]

    # ---- Backward-compatible default decoupled blocks -------------------
    # Historical split (u, theta, q | phi, psi, p, r) with per-channel scalar
    # inertia: the translational–rotational mass-matrix coupling is dropped by
    # construction and each row divides by the corresponding mass-matrix
    # diagonal (== m or the I_B diagonal when added mass is off). All
    # velocity-product derivatives (gyroscopic terms included) now arrive
    # through J_full itself — they are part of the differenced right-hand
    # side — so nothing is added analytically here. The full coupled
    # treatment lives in A_full and the selected blocks.
    # J_long keeps the historical (3, 3) shape: rows = [F_x, F_z, M_y]
    # (i.e. the longitudinal force/moment channels), cols = [u, theta, q].
    long_default_state_idx = _state_indices(["u", "theta", "q"])
    long_out_rows = [0, 2, 4]  # F_course, F_radial, M_normal
    j_long = J_full[np.ix_(long_out_rows, long_default_state_idx)]

    a_long = np.zeros((3, 3))
    a_long[0, :] = j_long[0, :] / mass_matrix_axes[0, 0]
    a_long[1, :] = [0.0, 0.0, kinematic_map[1, 1]]
    a_long[2, :] = j_long[2, :] / mass_matrix_axes[4, 4]

    lat_default_state_idx = _state_indices(["phi", "psi", "p", "r"])
    lat_out_rows = [1, 3, 5]  # F_normal, M_course, M_radial
    j_lat = J_full[np.ix_(lat_out_rows, lat_default_state_idx)]

    a_lat = np.zeros((4, 4))
    a_lat[0, :] = [0.0, 0.0, kinematic_map[0, 0], kinematic_map[0, 2]]
    a_lat[1, :] = [0.0, 0.0, kinematic_map[2, 0], kinematic_map[2, 2]]
    a_lat[2, :] = j_lat[1, :] / mass_matrix_axes[3, 3]
    a_lat[3, :] = j_lat[2, :] / mass_matrix_axes[5, 5]

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
        # Full 10-state coupled system (always available for inspection)
        "J_full": J_full,
        "A_full": A_full,
        "eig_full": eig_full,
        "vec_full": vec_full,
        "Tfast_full": _timescales_from_eigs(eig_full),
        "stable_full": bool(np.all(np.real(eig_full) < 0.0)),
        # Nonlinear fast-subsystem vector field xdot = f(delta_state), assembled
        # directly from the governing equations (independent of A_full). f(0) is
        # the trim equilibrium residual; central-differencing it cross-checks
        # A_full. Also usable for time-domain integration of the fast subsystem.
        "nonlinear_rhs": nonlinear_rhs,
        # Turn-rate OUTPUT of the 10-state block: delta_chi_dot_turn =
        # chi_turn_gain_row_full @ delta_state, in the frozen basis and the
        # canonical column order. Not a state — a linear functional, so it
        # slices under a DOF reduction and transforms with inv(T) under a
        # change of basis. ``None`` if the closure denominator vanishes.
        "chi_turn_gain_row_full": chi_turn_gain_row_full,
        "chi_turn_denominator_full": chi_turn_denominator_full,
        "chi_turn_denominator_source": chi_turn_denominator_source,
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
        # Numerical-hygiene diagnostics: warm-started FD solves that still
        # failed to converge (after one cold retry), and the attached-flow
        # state of the linearisation point itself. Consumers should distrust
        # the eigenvalues when solves are unconverged or panels are stalled
        # at trim (NaN margin = no stall information available).
        "n_unconverged_perturbation_solves": int(n_unconverged_solves),
        "perturbation_solves_converged": bool(n_unconverged_solves == 0),
        "gamma_warm_start_used": bool(
            _solve_accepts_gamma and gamma_baseline is not None
        ),
        "stall_margin_min_deg_at_trim": stall_margin_min_deg,
        "n_stalled_panels_at_trim": n_stalled_panels,
        # Course-frame transport rate used for the aero body-rate + inertial terms.
        "omega_c_model": "full" if omega_c_is_full else "radial_only",
        "omega_c_axes": np.asarray(omega_c_axes, dtype=float),
        # Co-rotating-axes representation (paper convention: course axes
        # carried by the body about B). The native states above are frozen-
        # axes; apply T to move to co-rotating components — eigenvalues are
        # invariant. See :func:`corotating_state_transform`.
        "v_kite_trim_axes": np.asarray(v_kite_trim_axes, dtype=float),
        "T_corotating_from_frozen": corotating_state_transform(
            v_kite_trim_axes, omega_c_axes
        ),
        # CG inertia tensor at the trim attitude (historical key).
        "inertia_stability": np.asarray(inertia_stability, dtype=float),
        "inertia_rotated_by_trim": bool(rotate_inertia_by_trim),
        # B-point (tether-attachment) quantities of the coupled assembly.
        "mass": float(mass),
        "cg_offset_world": np.asarray(cg_offset0, dtype=float),
        "cg_offset_trim_world": np.asarray(c_trim_world, dtype=float),
        # ...and in the stability-axes basis the states use, so the
        # CG-referenced position map can be built downstream. See
        # :func:`cg_position_state_transform`.
        "cg_offset_trim_axes": np.asarray(R_body @ c_trim_world, dtype=float),
        "inertia_b_axes": np.asarray(inertia_b_trim_axes, dtype=float),
        "mass_matrix_axes": np.asarray(mass_matrix_axes, dtype=float),
        "kinematic_map": np.asarray(kinematic_map, dtype=float),
        # Strip-theory apparent mass (zeros / "none" when disabled). The
        # matrix is the trim-attitude added-mass block already CONTAINED in
        # ``mass_matrix_axes``; subtract it to recover the rigid-only matrix.
        "include_added_mass": bool(include_added_mass),
        "added_mass_model": "strip_theory" if include_added_mass else "none",
        "added_mass_matrix_axes": np.asarray(added_mass_trim_axes, dtype=float),
        # Warm-started right-hand side at the trim (world components): the
        # residual the FD field carries at zero perturbation — warm/cold
        # anchor mismatch in the force channel, plus any constant reaction
        # the trim did not balance (the steering-line reaction of a
        # pinned-roll bridle-steering trim, or solver tolerance) in the
        # moment channel. A constant reaction contributes nothing to the
        # Jacobian at the balanced point, but a raw finite difference of
        # ``nonlinear_rhs`` picks up ``d(M^-1)/dTheta @ rhs_world_at_trim``;
        # together with ``mass_matrix_world_fn`` verification can subtract
        # that artifact exactly.
        "rhs_world_at_trim": np.concatenate(
            [
                np.asarray(_force0_warm, dtype=float),
                np.asarray(_moment0_warm, dtype=float),
            ]
        ),
        "mass_matrix_world_fn": mass_matrix_world_fn,
        "nonlinear_rhs_full": nonlinear_rhs_full,
        # CG-form evaluator: attitude perturbations about the CG, tether
        # re-solved at the swinging attachment. See awetrim.aerodynamics.cg_eom
        # for the equations, the trim-carryover proof, and the check helpers.
        "cg_eom_eval": cg_eom_eval,
        # Raw B-point force model f(delta_v, omega_perturb, ...) -> (force,
        # moment_B, res), world components, warm-started. Exposed for
        # verification and state-space experiments on channels outside the
        # canonical state set.
        "eval_force_moment_fn": eval_force_moment,
    }

    # ---- Position-augmented outputs (additive, position_states only) -----
    if position_states:
        result.update(
            {
                "J_aug": J_aug,
                "A_aug": A_aug,
                "eig_aug": eig_aug,
                "vec_aug": vec_aug,
                "Tfast_aug": _timescales_from_eigs(eig_aug),
                "stable_aug": bool(np.all(np.real(eig_aug) < 0.0)),
                "state_names_aug": list(AUG_STATE_NAMES),
                # Positions are space-fixed in the frozen axes, so the
                # co-rotating map extends with identity x/y rows.
                "T_corotating_from_frozen_aug": corotating_state_transform(
                    v_kite_trim_axes, omega_c_axes, AUG_STATE_NAMES
                ),
                "tether_position_model_aug": tether_position_model_aug,
                "tether_lateral_feedback": bool(tether_lateral_feedback),
                "eps_position_lateral_used": float(lateral_eps),
                "nonlinear_rhs_aug": nonlinear_rhs_aug,
                "nonlinear_rhs_aug_full": nonlinear_rhs_aug_full,
            }
        )

    # ---- Course-rate outputs (additive, course_rate_state only) ----------
    if course_rate_state:
        result.update(
            {
                "J_course_rate": J_course_rate,
                "A_chi": A_chi,
                "eig_chi": eig_chi,
                "vec_chi": vec_chi,
                "Tfast_chi": _timescales_from_eigs(eig_chi),
                "stable_chi": bool(np.all(np.real(eig_chi) < 0.0)),
                "state_names_chi": list(CHI_STATE_NAMES),
                # delta_chi_dot_turn = chi_turn_gain_row @ delta_state_9, the
                # algebraic variable eliminated by the normal equation. Apply
                # it to an eigenvector for that mode's turn-rate content.
                "chi_turn_gain_row": chi_turn_gain_row,
                # Same closure with v RETAINED: 10 differential states whose v
                # row is v_dot = 0 ("the frame follows the velocity") instead
                # of v pinned to zero. Block upper triangular, so
                # spec(A_chi10) = spec(A_chi) + one structural zero (the
                # neutral constant-sideslip mode); the v COLUMN survives, so
                # the aerodynamic effect of a standing sideslip still reaches
                # the other nine rows through the eigenvectors.
                "A_chi10": A_chi10,
                "eig_chi10": eig_chi10,
                "vec_chi10": vec_chi10,
                "Tfast_chi10": _timescales_from_eigs(eig_chi10),
                "state_names_chi10": list(ALL_STATE_NAMES),
                "chi10_gain_row": chi10_gain_row,
                "chi_turn_denominator": chi_turn_denominator,
                "chi_turn_closure_singular": bool(chi_turn_closure_singular),
                "eps_course_rate": float(eps_course_rate),
                "nonlinear_rhs_chi": nonlinear_rhs_chi,
                "nonlinear_rhs_chi_full": nonlinear_rhs_chi_full,
            }
        )

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
                mass_matrix=mass_matrix_axes,
                kinematic_map=kinematic_map,
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
                mass_matrix=mass_matrix_axes,
                kinematic_map=kinematic_map,
            )
            J_sel_lat, A_sel_lat = _build_state_space(
                J_full,
                sel_lat,
                mass_matrix=mass_matrix_axes,
                kinematic_map=kinematic_map,
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

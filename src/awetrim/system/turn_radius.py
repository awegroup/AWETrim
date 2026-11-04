"""The tightest turn a kite holds on each course at its steering limit.

A figure-eight seed is only flyable if every turn is at least as wide as the
kite can hold at its largest steering. That radius depends strongly on the
COURSE: heading down, gravity speeds the kite up and it needs about twice the
radius it needs heading up (LEI-V3 ROM, 6 m/s at 6 m: 17 m up, 34-37 m down).
:func:`steering_limited_turn_radius` tabulates it from quasi-steady states of
the system model: per course, the state at zero steering (pre-stall root),
then the steering ramped to the limit in small steps (continuation), each
direction separately; the radius is ``v_tau / |chi_dot_turn|``
(``kinematics.parametrized_patterns.heading_rate_turn``). A course without a
pre-stall steady state at the limit is UNFLYABLE (radius NaN).
"""
import contextlib
import io
import logging
from dataclasses import dataclass

import numpy as np

from awetrim.kinematics.parametrized_patterns import heading_rate_turn
from awetrim.system.state import State
from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

logger = logging.getLogger(__name__)

#: Seeds of the zero-steering solve (tangential speed [m/s], tension [N]),
#: tried in order until one converges below the stall.
SEED_GUESSES = tuple(
    (v, f) for v in (20.0, 30.0, 12.0, 45.0, 60.0) for f in (4000.0, 1500.0, 10000.0)
)


@dataclass
class TurnRadiusTable:
    """Steering-limited turn radius per course, for each turn direction.

    ``radius_positive[i]`` is the radius [m] of the turn with
    ``chi_dot_turn > 0`` (geodesic curvature ``kappa_g > 0``) on course
    ``courses[i]`` [rad, 0 = flying up], ``radius_negative`` the other way;
    NaN = no steady turn there. ``input_steering_max`` is the |u_s| used.
    """

    courses: np.ndarray
    radius_positive: np.ndarray
    radius_negative: np.ndarray
    input_steering_max: float
    distance_radial: float
    angle_elevation: float
    input_depower: float

    def radius(self, course, curvature_sign):
        """Radius [m] for turns on ``course`` [rad] curving with
        ``curvature_sign`` (+1/-1), periodic linear interpolation in the
        course; inf where the kite has no steady turn (a NaN neighbour)."""
        course = np.mod(np.asarray(course, dtype=float), 2.0 * np.pi)
        sign = np.broadcast_to(np.sign(curvature_sign), course.shape)
        x = np.r_[self.courses, self.courses[0] + 2.0 * np.pi]
        out = np.empty(course.shape)
        for table, which in ((self.radius_positive, sign > 0), (self.radius_negative, sign <= 0)):
            y = np.r_[table, table[0]]
            y = np.where(np.isfinite(y), y, np.inf)
            i = np.clip(np.searchsorted(x, course[which], side="right") - 1, 0, len(x) - 2)
            w = (course[which] - x[i]) / (x[i + 1] - x[i])
            lo, hi = y[i], y[i + 1]
            both = np.isfinite(lo) & np.isfinite(hi)
            out[which] = np.where(both, lo + w * (hi - lo), np.inf)
        return out

    def summary(self):
        both = np.r_[self.radius_positive, self.radius_negative]
        return {
            "radius_min_m": float(np.nanmin(both)) if np.isfinite(both).any() else np.nan,
            "radius_max_m": float(np.nanmax(both)) if np.isfinite(both).any() else np.nan,
            "unflyable": int(np.isnan(both).sum()),
            "states": int(both.size),
        }


def steering_limit(system_model):
    """|u_s| the optimizer may use: the hardware box after the ROM clip."""
    box = (getattr(system_model, "hardware_limits", None) or {}).get(
        "input_steering", DEFAULT_OPTI_LIMITS["input_steering"]
    )
    return float(min(abs(box[0]), abs(box[1])))


def _solve(system_model, state, winch=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return system_model.solve_quasi_steady(state, winch=winch)


def _pre_stall(system_model, solved):
    stall = (getattr(system_model.kite, "aero_params", None) or {}).get("angle_of_attack_stall")
    return solved is not None and (stall is None or solved.angle_of_attack < stall)


def steering_limited_turn_radius(
    system_model,
    *,
    distance_radial,
    angle_elevation,
    input_depower,
    input_steering_max=None,
    speed_radial=0.0,
    winch=None,
    angle_azimuth=0.0,
    n_courses=24,
    continuation_steps=8,
):
    """:class:`TurnRadiusTable` of ``system_model`` at one point of the window.

    ``distance_radial`` [m], ``angle_elevation``/``angle_azimuth`` [rad],
    ``input_depower`` in the ROM's unit. The reel speed is either fixed
    (``speed_radial`` [m/s]) or, with a ``winch`` (``system.winch.Winch`` of
    the phase's radial parameters), solved from its force law with each
    state -- use the winch: a fixed slow reel at high wind is an overpowered
    state (tens of kN) whose pre-stall root the solve does not find, so the
    course reads unflyable. ``input_steering_max`` defaults to
    :func:`steering_limit`. The wind and air density are the model's.
    """
    us_max = steering_limit(system_model) if input_steering_max is None else float(input_steering_max)
    courses = np.linspace(0.0, 2.0 * np.pi, int(n_courses), endpoint=False)
    radius = {+1: np.full(courses.size, np.nan), -1: np.full(courses.size, np.nan)}

    def state(chi, us, v, chi_dot, tension, v_r=speed_radial):
        return State(
            distance_radial=distance_radial, angle_elevation=angle_elevation,
            angle_azimuth=angle_azimuth, angle_course=chi, speed_radial=v_r,
            timeder_speed_radial=0.0, timeder_speed_tangential=0.0,
            input_depower=input_depower, input_steering=us, speed_tangential=v,
            timeder_angle_course=chi_dot, tension_tether_ground=tension,
        )

    for i, chi in enumerate(courses):
        start = None
        for v, tension in SEED_GUESSES:
            solved = _solve(system_model, state(chi, 0.0, v, 0.0, tension), winch)
            if _pre_stall(system_model, solved):
                start = solved
                break
        if start is None:
            continue
        for direction in (+1.0, -1.0):
            solved = start
            for us in np.linspace(0.0, direction * us_max, int(continuation_steps) + 1)[1:]:
                solved = _solve(system_model, state(
                    chi, us, solved.speed_tangential, solved.timeder_angle_course,
                    solved.tension_tether_ground, solved.speed_radial,
                ), winch)
                if not _pre_stall(system_model, solved):
                    solved = None
                    break
            if solved is None:
                continue
            turn = heading_rate_turn(
                solved.timeder_angle_course, solved.speed_tangential, distance_radial,
                angle_elevation, chi,
            )
            if abs(turn) > 1e-9:
                # Where gravity dominates both steering directions may turn the
                # same way: keep the tighter one; the other way stays NaN.
                table = radius[int(np.sign(turn))]
                table[i] = np.fmin(table[i], solved.speed_tangential / abs(turn))

    return TurnRadiusTable(
        courses=courses, radius_positive=radius[+1], radius_negative=radius[-1],
        input_steering_max=us_max, distance_radial=float(distance_radial),
        angle_elevation=float(angle_elevation), input_depower=float(input_depower),
    )

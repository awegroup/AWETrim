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

"""Pydantic request/response schemas for the reelout optimization server.

Units and frames (documented for external clients, e.g. a Julia kite
simulator): angles in radians, angle rates in rad/s, lengths in metres,
speeds in m/s, forces in newtons, times in seconds. The trajectory lives in
the wind-aligned spherical ground frame: ``distance_radial`` (r) is the
tether-sphere radius, ``azimuth`` (phi) rotates about the vertical axis with
0 pointing downwind (+x, the direction the wind blows towards), and
``elevation`` (beta) is measured from the horizontal plane.

The client-facing co-simulation structs are the exception: the trajectory
(``TrajectoryAngles``) and the wind direction (``InflowConditions``) are in
DEGREES, matching the Julia/MATLAB clients.
"""

import math
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from awetrim.environment.wind_profiles import (
    DEFAULT_ALPHA,
    DEFAULT_Z0,
    REFERENCE_HEIGHT,
    wind_kwargs_from_inflow,
)


# ---------------------------------------------------------------------------
# Inflow conditions — the client-facing wind description
# ---------------------------------------------------------------------------
class InflowConditions(BaseModel):
    """Wind seen by the kite, as sent by the simulator (shared struct).

    Mirrors the ``InflowConditions`` struct of the Julia/MATLAB/Python
    clients: a reference speed at 6 m plus a profile law. The profile laws
    are those of AtmosphericModels.jl; laws 4-6 ignore ``wind_speed``,
    ``alpha`` and ``z0`` and are fitted to ``heights``/``speeds`` instead.
    """

    model_config = ConfigDict(extra="forbid")

    wind_speed: float = Field(
        gt=0, description=f"Wind speed [m/s] at {REFERENCE_HEIGHT:g} m height"
    )
    wind_direction: float = Field(
        default=0.0,
        description="Direction the wind comes FROM [deg], 0 = North, "
        "90 = East. Only orients the result in the world frame — the path "
        "is optimized in the wind-aligned frame (azimuth 0 = downwind).",
    )
    profile_law: Literal[0, 1, 2, 3, 4, 5, 6] = Field(
        description="0=CONST, 1=EXP, 2=LOG, 3=EXPLOG, 4=CUSTOM_LOG, "
        "5=CUSTOM_EXP, 6=CUSTOM_JET; the CUSTOM_* laws are least-squares "
        "fits of the heights/speeds table",
    )
    alpha: float = Field(
        default=DEFAULT_ALPHA,
        description="Exponent of the power-law profile (EXP, EXPLOG)",
    )
    z0: float = Field(
        default=DEFAULT_Z0,
        gt=0,
        description="Surface roughness [m] (LOG, EXPLOG)",
    )
    turbulence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0 = no turbulence, 1 = full turbulence. Accepted and "
        "echoed, but NOT used: the optimizer is deterministic and works on "
        "the mean profile.",
    )
    heights: Optional[List[float]] = Field(
        default=None,
        description=f"Sample heights [m] for the CUSTOM_* laws, "
        f"default [{REFERENCE_HEIGHT:g}]",
    )
    speeds: Optional[List[float]] = Field(
        default=None,
        description="Wind speeds [m/s] at heights, default [wind_speed]",
    )

    @model_validator(mode="after")
    def _check_samples(self):
        if (self.heights is None) != (self.speeds is None):
            raise ValueError("heights and speeds must be given together")
        if self.heights is not None:
            if len(self.heights) != len(self.speeds):
                raise ValueError("heights and speeds must have the same length")
            if any(h <= 0 for h in self.heights):
                raise ValueError("heights must be positive")
            if any(v < 0 for v in self.speeds):
                raise ValueError("speeds must be non-negative")
        # Fail fast on a CUSTOM_* law without enough samples to fit, and on
        # samples the fit cannot handle (rather than deep inside the solver).
        wind_kwargs_from_inflow(self.model_dump())
        return self


def wind_kwargs_from_inflow_schema(inflow: InflowConditions) -> dict:
    """Translate :class:`InflowConditions` into ``create_wind_model`` kwargs."""
    return wind_kwargs_from_inflow(inflow.model_dump())


# ---------------------------------------------------------------------------
# Initial guess — mirrors make_bspline_path_parameters_from_named_curve kwargs
# ---------------------------------------------------------------------------
class InitialGuess(BaseModel):
    curve_type: Literal["lissajous", "lemniscate", "helix"] = Field(
        default="lissajous",
        description="Shape family of the starting curve: lissajous/lemniscate "
        "= figure-eight, helix = circular loops",
    )
    M: int = Field(
        default=10,
        ge=4,
        description="Number of B-spline control points (more = more shape "
        "freedom for the optimizer, slower solve)",
    )
    n_fit: int = Field(
        default=400, ge=10, description="Samples used to fit the spline"
    )
    s_init: float = Field(default=0.0, description="Path-parameter start")
    s_final: float = Field(
        default=2.0 * math.pi, description="Path-parameter end (2*pi = one figure)"
    )
    az_amp0: float = Field(
        default=0.3, description="Azimuth half-width of the figure [rad]"
    )
    beta0: float = Field(
        default=0.35, description="Mean elevation of the figure [rad]"
    )
    beta_amp0: float = Field(
        default=0.12, description="Elevation half-height of the figure [rad]"
    )
    downloops: bool = Field(
        default=True,
        description="True = kite turns downward through the loops "
        "(downloop), False = upward (uploop)",
    )


# ---------------------------------------------------------------------------
# Co-simulation client structs (mirrors the Julia-side InitParams/StepParams)
# ---------------------------------------------------------------------------
class WinchParams(BaseModel):
    """Ground-station winch controller: v_set = k_v * sqrt(force)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["reelout", "reelin"] = Field(
        default="reelout",
        description="Phase to optimize; only 'reelout' is supported for now",
    )
    k_v: float = Field(
        gt=0, description="Winch law gain: v_set = k_v * sqrt(force)"
    )
    f_min: float = Field(ge=0, description="Minimum winch force [N]")
    f_max: float = Field(gt=0, description="Maximum winch force [N]")
    # Past the force limit the controller holds F = f_max while the reel-out
    # speed keeps rising, until the winch's power limit P_max = f_max * v_max.
    # So the speed cap is a genuine input, not derivable from k_v/f_max: give
    # either v_max or p_max (v_max = p_max / f_max). Omitted -> the optimizer's
    # default reel-speed bound (10 m/s) applies.
    v_max: Optional[float] = Field(
        default=None, gt=0, description="Maximum reel-out speed [m/s]"
    )
    p_max: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum winch power [W]; alternative to v_max "
        "(v_max = p_max / f_max)",
    )
    # Treat the winch gain as a design variable instead of a given. The
    # returned path then assumes the OPTIMIZED k_v, which the replies echo back
    # in this same struct -- exactly like depower, the path is NOT flyable with
    # the k_v that was sent in.
    optimize_k_v: bool = Field(
        default=False,
        description="Let the optimizer retune the winch gain k_v. The reply "
        "echoes the optimized value; fly that one, not the value sent in.",
    )
    k_v_bounds: Optional[List[float]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="[k_v_min, k_v_max] bracket for the optimized gain; "
        "defaults to a factor 2 either side of k_v. Ignored unless "
        "optimize_k_v is true.",
    )
    # Corner sharpness of the two soft saturations applied to the tension curve.
    # Carried on the wire so a client that INVERTS this curve to command a reel-out
    # speed cannot silently drift from the curve the path was planned against: the
    # two sides used to agree only because both happened to say 1e-3.
    softplus_beta: Optional[float] = Field(
        default=None,
        gt=0,
        description="Corner sharpness of the UPPER force limit [1/N]; larger is "
        "sharper, and the transition spans a tension band of order 1/beta. "
        "Unset keeps the server default (1e-3).",
    )
    softminus_beta: Optional[float] = Field(
        default=None,
        gt=0,
        description="Corner sharpness of the LOWER force limit [1/N]; larger is "
        "sharper. Unset keeps the server default (1e-3). Watch this one: the "
        "effective floor is softplus(beta*f_min)/beta, so a 1/beta larger than "
        "f_min itself dominates the limit it smooths -- at 1e-3 an f_min of 350 N "
        "gives an 884 N floor, 2.5x the value requested.",
    )
    # Reel-in-capable blend, mirroring WinchControllers.jl's calc_vro_soft
    # (Winch.tension_curve computes force from speed; that one inverts it).
    # 0 leaves the winch law exactly as above (symmetric, unphysical for
    # negative speed); 1 replaces it below f_min with a straight line through
    # (0, f_min) and (v_reel_in, 0), handed to the quadratic law above by a
    # smooth maximum. Independent of `mode`: it makes the tension curve
    # itself valid for a momentary negative speed_radial within whichever
    # phase is being optimized, not a reel-in trajectory phase in its own
    # right (mode = "reelin" is still rejected server-side).
    use_awe_trim: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Blend factor in [0, 1] towards a reel-in-capable winch "
        "law. 0 (default) is the plain law above; 1 replaces it below f_min "
        "with a straight reel-in line, smoothly handed to the quadratic law.",
    )
    v_reel_in: Optional[float] = Field(
        default=None,
        lt=0,
        description="use_awe_trim only: reel-in speed [m/s] at zero force. "
        "Unset keeps the server default (-2.0). Must be negative.",
    )
    reel_in_beta: Optional[float] = Field(
        default=None,
        gt=0,
        description="use_awe_trim only: sharpness of the smooth handover "
        "between the reel-in line and the quadratic law [s/m]. Unset keeps "
        "the server default (20.0). Unlike softminus_beta above, this has no "
        "matching sharpness requirement against f_min/k_v -- the forward "
        "law's quadratic term has zero, not infinite, slope at its own "
        "zero, so soft_max stays well-behaved for any positive value.",
    )

    @model_validator(mode="after")
    def _check_forces(self):
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")
        if self.v_max is not None and self.p_max is not None:
            raise ValueError("give either v_max or p_max, not both")
        if self.k_v_bounds is not None:
            lo, hi = self.k_v_bounds
            if lo <= 0 or hi <= 0:
                raise ValueError("k_v_bounds must be positive")
            if lo >= hi:
                raise ValueError("k_v_bounds must be [min, max] with min < max")
            if not lo <= self.k_v <= hi:
                raise ValueError("k_v must lie inside k_v_bounds")
        return self

    def reel_speed_limit(self) -> Optional[float]:
        """v_max [m/s] from whichever limit was given (None = default bound)."""
        if self.v_max is not None:
            return self.v_max
        if self.p_max is not None:
            return self.p_max / self.f_max
        return None


class DepowerSpec(BaseModel):
    """How the depower input l_dp is handled during the optimization.

    Depower changes the kite's angle of attack, so the optimized path is only
    flyable at the depower setting it was optimized for. Clients that cannot
    command depower should send ``mode="fixed"`` with the value their own
    simulator flies; clients that can should read the value back from the
    ``depower`` field of the /init and /step replies and apply it.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed", "optimize", "profile"] = Field(
        default="optimize",
        description="'fixed': l_dp stays at value (the client's setting is "
        "honoured). 'optimize': one scalar l_dp is optimized (default). "
        "'profile': l_dp is optimized per node, giving a depower time-profile.",
    )
    value: Optional[float] = Field(
        default=None,
        description="Depower setting: the FIXED value in 'fixed' mode, the "
        "starting value in 'optimize'/'profile' mode. Defaults to the "
        "input_depower knob, else the kite cycle config's value.",
    )


class DepowerReply(BaseModel):
    """The depower the returned trajectory was optimized for — fly this.

    In 'profile' mode ``profile`` holds the per-node values, index-aligned with
    the reply trajectory's azimuth/elevation; ``value`` is then its mean.
    """

    mode: Literal["fixed", "optimize", "profile"]
    value: float = Field(description="Scalar depower setting l_dp")
    profile: Optional[List[float]] = Field(
        default=None,
        description="Per-node depower, aligned with the reply trajectory "
        "(profile mode only; null otherwise)",
    )


class TrajectoryAngles(BaseModel):
    """Flight path as azimuth/elevation samples in DEGREES.

    Periodic: the last point may repeat the first (it is dropped before
    fitting). Responses are always closed (last point equals the first).
    """

    azimuth: List[float] = Field(min_length=8, description="[deg]")
    elevation: List[float] = Field(min_length=8, description="[deg]")

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.azimuth) != len(self.elevation):
            raise ValueError("azimuth and elevation must have the same length")
        return self


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class PatternLimits(BaseModel):
    """Optional box on where the optimized pattern may go, in DEGREES.

    The pattern is a periodic cubic B-spline in azimuth/elevation; bounding
    its control coefficients bounds the whole continuous curve (convex-hull
    property), so these limits hold everywhere along the path, not only at the
    nodes. Use them to keep the optimizer out of shapes your controller or
    simulator should not fly -- e.g. an ``elevation_max`` ceiling against the
    run-away-to-zenith basin, or an ``azimuth_amplitude_min`` floor against
    the zero-width collapse of a cold start. Every field is optional; an
    omitted field keeps the optimizer's default for it (|azimuth| <= 45.8 deg,
    0.6 <= elevation <= 51.6 deg, no amplitude floor). On /step the struct
    replaces the session's limits as a whole (send ``{}`` to clear them).
    """

    model_config = ConfigDict(extra="forbid")

    azimuth_max: Optional[float] = Field(
        default=None,
        gt=0,
        le=90,
        description="|azimuth| of the path stays <= this [deg]",
    )
    elevation_min: Optional[float] = Field(
        default=None,
        ge=0,
        lt=90,
        description="elevation of the path stays >= this [deg]",
    )
    elevation_max: Optional[float] = Field(
        default=None,
        gt=0,
        le=90,
        description="elevation of the path stays <= this [deg]",
    )
    azimuth_amplitude_min: Optional[float] = Field(
        default=None,
        ge=0,
        le=90,
        description="The figure's azimuth half-width stays >= this [deg] "
        "(one smooth constraint: mean over the path of azimuth^2 >= value^2/2, "
        "which a figure-eight or helix of half-width A satisfies with A >= "
        "value). Guards the degenerate zero-width collapse. 0/omitted = off.",
    )
    elevation_amplitude_max: Optional[float] = Field(
        default=None,
        ge=0,
        le=90,
        description="The figure's elevation half-span stays <= this [deg] "
        "(one smooth constraint: mean over the path of "
        "(elevation - mean)^2 <= value^2/2, which a figure-eight or helix of "
        "half-span B satisfies with B <= value). Caps how TALL the pattern is "
        "where ``elevation_max`` only caps where it may sit. 0/omitted = off.",
    )

    @model_validator(mode="after")
    def _check_elevation_band(self):
        if (
            self.elevation_min is not None
            and self.elevation_max is not None
            and self.elevation_max <= self.elevation_min
        ):
            raise ValueError("elevation_max must be greater than elevation_min")
        return self


class InitRequest(BaseModel):
    # Unknown keys are rejected: a client still sending a removed name (e.g.
    # the old "distance_radial") gets a 422 instead of a silently ignored field.
    model_config = ConfigDict(extra="forbid")

    inflow_conditions: InflowConditions = Field(
        description="Wind seen by the kite (shared client struct); required, "
        "the optimizer cannot work without wind",
    )
    name: str = Field(
        default="reelout-optimization", description="Name of the simulation"
    )
    winch_params: Optional[WinchParams] = Field(
        default=None,
        description="Ground-station winch law; mapped onto the optimizer's "
        "radial force model so the optimized pattern matches the client's "
        "winch behavior",
    )
    trajectory: Optional[TrajectoryAngles] = Field(
        default=None,
        description="Starting flight path in degrees; fitted to the pattern "
        "B-spline. Takes precedence over initial_guess.",
    )
    initial_guess: Optional[InitialGuess] = None
    length: Optional[float] = Field(
        default=None,
        gt=0,
        description="Initial tether length / pattern sphere radius r0 [m]; "
        "defaults to the value in the kite's cycle config",
    )
    system_config_path: Optional[str] = None
    cycle_config_path: Optional[str] = None
    optimization_params: List[str] = Field(
        default_factory=lambda: ["C_phi", "C_beta", "input_depower"]
    )
    target: Literal["power", "energy"] = "power"
    n_points: int = Field(default=100, ge=10, le=1000)
    # The InitParams solver knobs: sent flat by the clients, merged into the
    # cycle config's sim_parameters by the session. They are the only solver
    # knobs the API exposes — there is no generic overrides dict.
    input_depower: Optional[float] = Field(
        default=None,
        description="Depower setting l_dp: the starting value when depower is "
        "optimized (the default), the fixed value otherwise. Prefer the "
        "``depower`` struct, which also says whether it is optimized.",
    )
    reg_weight: Optional[float] = Field(
        default=None, description="Regularization weight"
    )
    detect_simple_bounds: Optional[bool] = Field(
        default=None, description="Solver flag"
    )
    depower: Optional[DepowerSpec] = Field(
        default=None,
        description="Depower handling {mode, value}. Takes precedence over "
        "input_depower / optimization_params when given; when omitted the "
        "mode is derived from optimization_params (input_depower listed = "
        "optimize) and the value from input_depower.",
    )
    min_turn_radius: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum physical turn radius [m] the optimized path must "
        "respect everywhere (geodesic radius on the tether sphere, r/|kappa|, "
        "at the tether length at which each part of the path is flown). "
        "Client-defined, e.g. 1/(c1*u_s_max) for a kite steered by "
        "psi_dot = c1*v_a*u_s. Omitted/0 = no constraint (only the kite "
        "model's own steering limits apply). Enforced densely along the path, "
        "so it also rules out cusps and near-degenerate tiny loops.",
    )
    pattern_limits: Optional[PatternLimits] = Field(
        default=None,
        description="Optional box on the path's azimuth/elevation range and "
        "an azimuth-amplitude floor, in degrees; omitted = the optimizer's "
        "defaults (plus whatever the cycle config sets).",
    )


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inflow_conditions: Optional[InflowConditions] = Field(
        default=None,
        description="Updated wind for this re-optimization; the inflow of "
        "/init keeps being used when it is omitted",
    )
    winch_params: Optional[WinchParams] = Field(
        default=None, description="Updated winch law for this re-optimization"
    )
    trajectory: Optional[TrajectoryAngles] = Field(
        default=None,
        description="Current flight path in degrees, used as the starting "
        "guess for the re-optimization",
    )
    length: Optional[float] = Field(
        default=None,
        gt=0,
        description="Current tether length [m]; re-anchors the pattern radius r0",
    )
    depower: Optional[DepowerSpec] = Field(
        default=None,
        description="Updated depower handling for this and later steps; "
        "omit to keep the mode and value set at /init",
    )
    min_turn_radius: Optional[float] = Field(
        default=None,
        ge=0,
        description="Updated minimum turn radius [m] for this and later steps; "
        "omit to keep the value set at /init, send 0 to remove the constraint",
    )
    pattern_limits: Optional[PatternLimits] = Field(
        default=None,
        description="Updated pattern limits for this and later steps; omit to "
        "keep the current ones, send {} to clear them",
    )
    max_iter: Optional[int] = Field(default=None, ge=1)
    wait: bool = Field(
        default=True,
        description="True (default): block until the solve finishes and "
        "return the optimized trajectory in the reply. False: return 202 "
        "immediately and poll /status.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class SolveMetrics(BaseModel):
    energy_J: float
    total_time_s: float
    avg_power_W: float
    turn_radius_min_m: Optional[float] = Field(
        default=None,
        description="Tightest physical turn radius [m] of the returned path "
        "(geodesic radius r/|kappa| on the tether sphere at the length where "
        "that part is flown, from the optimizer's own curvature expression, "
        "sampled densely along the path); >= min_turn_radius when that "
        "constraint is on. Null if it could not be evaluated.",
    )


class StatusResponse(BaseModel):
    state: Literal["uninitialized", "ready", "solving", "converged", "failed"]
    step_count: int = 0
    successful_steps: int = 0
    last_error: Optional[str] = None
    metrics: Optional[SolveMetrics] = None
    inflow_conditions: Optional[InflowConditions] = None
    wind: Optional[dict] = Field(
        default=None,
        description="Read-only diagnostics: the Wind model the optimizer "
        "built from inflow_conditions",
    )
    n_points: Optional[int] = None
    last_step_started: Optional[datetime] = None
    last_step_finished: Optional[datetime] = None
    session_id: str = "default"


class StepAccepted(BaseModel):
    state: Literal["solving"] = "solving"
    step_index: int


class InitReply(BaseModel):
    """Reply of POST /init — contains the InitParams struct (name, length,
    winch_params, inflow_conditions, trajectory + the solver knobs)
    plus server state. The trajectory is the fitted starting path (closed:
    last point equals the first, degrees); inflow_conditions echoes the
    accepted inflow; ``depower`` echoes the mode and starting value (no solve
    has run yet) and ``min_turn_radius`` the limit the optimizer will enforce."""

    name: str
    length: Optional[float] = Field(
        default=None, description="Accepted tether length / sphere radius [m]"
    )
    winch_params: Optional[WinchParams] = None
    inflow_conditions: Optional[InflowConditions] = None
    trajectory: TrajectoryAngles
    input_depower: Optional[float] = None
    reg_weight: Optional[float] = None
    detect_simple_bounds: Optional[bool] = None
    depower: Optional[DepowerReply] = None
    min_turn_radius: Optional[float] = Field(
        default=None,
        description="Minimum turn radius [m] the optimizer will enforce "
        "(null = unconstrained)",
    )
    pattern_limits: Optional[PatternLimits] = Field(
        default=None,
        description="Pattern limits in force [deg] (null = optimizer defaults)",
    )
    state: str
    n_points: Optional[int] = None
    session_id: str = "default"


class StepReply(BaseModel):
    """Reply of POST /step with wait=true — contains the StepParams struct
    (length, winch_params, trajectory, depower) plus solve metadata. The
    trajectory is the OPTIMIZED path (closed, degrees) and ``depower`` is the
    setting it was optimized for: fly BOTH, or the reported metrics are not
    achievable. ``min_turn_radius`` is the limit the path was optimized under
    (null = unconstrained); metrics.turn_radius_min_m is what it achieves."""

    length: Optional[float] = Field(
        default=None, description="Tether length the pattern is anchored to [m]"
    )
    winch_params: Optional[WinchParams] = None
    trajectory: TrajectoryAngles
    depower: Optional[DepowerReply] = None
    min_turn_radius: Optional[float] = None
    pattern_limits: Optional[PatternLimits] = Field(
        default=None,
        description="Pattern limits the path was optimized under [deg] "
        "(null = optimizer defaults)",
    )
    state: str
    step_index: int
    metrics: SolveMetrics


class SplineBlock(BaseModel):
    spline_type: str = "periodic"
    M: int
    C_phi: List[float]
    C_beta: List[float]
    r0: float
    s_init: float
    s_final: float
    downloops: bool = True


class TrajectoryResponse(BaseModel):
    step_index: int
    n_points: int
    # Dense per-node guidance table. Columns: t, s, azimuth, elevation,
    # azimuth_dot, elevation_dot, distance_radial, speed_radial, s_dot,
    # tension_tether_ground, input_steering, input_depower, turn_radius. The
    # input_depower column is always present: constant in fixed/optimize
    # mode, the per-node profile in profile mode. turn_radius [m] is the
    # physical turn radius of the path at each node (r/|kappa|, geodesic on
    # the tether sphere at that node's distance_radial).
    table: Dict[str, List[Optional[float]]]
    spline: SplineBlock
    metrics: SolveMetrics
    optimized_parameters: dict
    timeseries: Optional[Dict[str, List[Optional[float]]]] = None
    energy_metrics: Optional[dict] = None

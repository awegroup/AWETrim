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
"""

import math
from datetime import datetime
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Wind profile (discriminated union on model_type)
# ---------------------------------------------------------------------------
class WindLogarithmic(BaseModel):
    model_type: Literal["logarithmic"]
    U_ref: float = Field(gt=0, description="Wind speed [m/s] at z_ref")
    z_ref: float = Field(default=100.0, gt=0, description="Reference height [m]")
    z0: float = Field(default=0.03, gt=0, description="Roughness length [m]")
    direction_wind: float = Field(
        default=0.0, description="Wind direction [rad], 0 = blowing along +x"
    )


class WindUniform(BaseModel):
    model_type: Literal["uniform"]
    U_ref: float = Field(gt=0, description="Wind speed [m/s], constant with height")
    direction_wind: float = 0.0


class WindTabulated(BaseModel):
    model_type: Literal["tabulated"]
    heights: List[float] = Field(min_length=2, description="Sample heights [m]")
    speeds: List[float] = Field(description="Wind speeds [m/s] at heights")
    direction_wind: float = 0.0

    @model_validator(mode="after")
    def _check_table(self):
        if len(self.heights) != len(self.speeds):
            raise ValueError("heights and speeds must have the same length")
        if any(h2 <= h1 for h1, h2 in zip(self.heights, self.heights[1:])):
            raise ValueError("heights must be strictly increasing")
        if self.heights[0] <= 0:
            raise ValueError("heights must be positive")
        if any(v < 0 for v in self.speeds):
            raise ValueError("speeds must be non-negative")
        return self


WindProfile = Annotated[
    Union[WindLogarithmic, WindUniform, WindTabulated],
    Field(discriminator="model_type"),
]


def wind_kwargs_from_schema(wind: Union[WindLogarithmic, WindUniform, WindTabulated]) -> dict:
    """Translate a wind schema into ``create_wind_model`` keyword arguments."""
    if wind.model_type == "logarithmic":
        return dict(
            model_type="logarithmic",
            U_ref=wind.U_ref,
            z_ref=wind.z_ref,
            z0=wind.z0,
            direction_wind=wind.direction_wind,
        )
    if wind.model_type == "uniform":
        return dict(
            model_type="uniform",
            U_ref=wind.U_ref,
            direction_wind=wind.direction_wind,
        )
    return dict(
        model_type="tabulated",
        heights=wind.heights,
        speeds=wind.speeds,
        direction_wind=wind.direction_wind,
    )


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

    mode: Literal["reelout", "reelin"] = Field(
        default="reelout",
        description="Phase to optimize; only 'reelout' is supported for now",
    )
    k_v: float = Field(
        gt=0, description="Winch law gain: v_set = k_v * sqrt(force)"
    )
    v_max: float = Field(gt=0, description="Maximum winch speed [m/s]")
    f_min: float = Field(ge=0, description="Minimum winch force [N]")
    f_max: float = Field(gt=0, description="Maximum winch force [N]")

    @model_validator(mode="after")
    def _check_forces(self):
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")
        return self


class DepowerSpec(BaseModel):
    """How the depower input l_dp is handled during the optimization.

    Depower changes the kite's angle of attack, so the optimized path is only
    flyable at the depower setting it was optimized for. Clients that cannot
    command depower should send ``mode="fixed"`` with the value their own
    simulator flies; clients that can should read the value back from the
    ``depower`` field of the /init and /step replies and apply it.
    """

    mode: Literal["fixed", "optimize", "profile"] = Field(
        default="optimize",
        description="'fixed': l_dp stays at value (the client's setting is "
        "honoured). 'optimize': one scalar l_dp is optimized (default). "
        "'profile': l_dp is optimized per node, giving a depower time-profile.",
    )
    value: Optional[float] = Field(
        default=None,
        description="Depower setting: the FIXED value in 'fixed' mode, the "
        "starting value in 'optimize'/'profile' mode. Defaults to the kite "
        "cycle config's value.",
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
class InitRequest(BaseModel):
    wind: WindProfile
    name: str = Field(
        default="reelout-optimization", description="Name of the simulation"
    )
    max_time: Optional[float] = Field(
        default=None,
        gt=0,
        description="Client-side simulation horizon [s]; stored and echoed",
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
    depower: Optional[DepowerSpec] = Field(
        default=None,
        description="Depower handling. Takes precedence over the legacy "
        "combination of optimization_params/sim_parameters.input_depower; "
        "when omitted those still apply unchanged.",
    )
    distance_radial: Optional[float] = Field(
        default=None,
        gt=0,
        description="Initial tether length / pattern sphere radius r0 [m]; "
        "defaults to the value in the kite's cycle config",
    )
    min_turn_radius: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum physical turn radius [m] the optimized path must "
        "respect everywhere (geodesic radius on the tether sphere, r/|kappa|, "
        "evaluated at the tether length at which each part of the path is "
        "flown). Client-defined, e.g. 1/(c1*u_s_max) for a kite steered by "
        "psi_dot = c1*v_a*u_s. Omitted/0 = no constraint (only the kite "
        "model's own steering limits apply). Enforced densely along the "
        "path, so it also rules out cusps and degenerate collapsed shapes.",
    )
    system_config_path: Optional[str] = None
    cycle_config_path: Optional[str] = None
    optimization_params: List[str] = Field(
        default_factory=lambda: ["C_phi", "C_beta", "input_depower"]
    )
    target: Literal["power", "energy"] = "power"
    n_points: int = Field(default=100, ge=10, le=1000)
    sim_parameters: Dict = Field(
        default_factory=dict,
        description="Overrides merged into the pattern config sim_parameters "
        "(e.g. input_depower, reg_weight, detect_simple_bounds)",
    )


class StepRequest(BaseModel):
    wind: Optional[WindProfile] = None
    winch_params: Optional[WinchParams] = Field(
        default=None, description="Updated winch law for this re-optimization"
    )
    trajectory: Optional[TrajectoryAngles] = Field(
        default=None,
        description="Current flight path in degrees, used as the starting "
        "guess for the re-optimization",
    )
    depower: Optional[DepowerSpec] = Field(
        default=None,
        description="Updated depower handling for this and later steps; "
        "omit to keep the mode and value set at /init",
    )
    distance_radial: Optional[float] = Field(
        default=None,
        gt=0,
        description="Current tether length [m]; re-anchors the pattern radius r0",
    )
    min_turn_radius: Optional[float] = Field(
        default=None,
        ge=0,
        description="Updated minimum turn radius [m] for this and later steps; "
        "omit to keep the value set at /init, send 0 to remove the constraint",
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
    wind: Optional[dict] = None
    n_points: Optional[int] = None
    last_step_started: Optional[datetime] = None
    last_step_finished: Optional[datetime] = None
    session_id: str = "default"


class StepAccepted(BaseModel):
    state: Literal["solving"] = "solving"
    step_index: int


class InitReply(BaseModel):
    """Reply of POST /init — contains the InitParams struct (name, max_time,
    winch_params, trajectory, depower) plus server state. The trajectory is the
    fitted starting path (closed: last point equals the first, degrees) and
    ``depower`` echoes the mode and starting value (no solve has run yet)."""

    name: str
    max_time: Optional[float] = None
    winch_params: Optional[WinchParams] = None
    trajectory: TrajectoryAngles
    depower: Optional[DepowerReply] = None
    min_turn_radius: Optional[float] = Field(
        default=None,
        description="Minimum turn radius [m] the optimizer will enforce "
        "(null = unconstrained)",
    )
    state: str
    n_points: Optional[int] = None
    session_id: str = "default"


class StepReply(BaseModel):
    """Reply of POST /step with wait=true — contains the StepParams struct
    (winch_params, trajectory, depower) plus solve metadata. The trajectory is
    the OPTIMIZED path (closed, degrees) and ``depower`` is the setting it was
    optimized for: fly BOTH, or the reported metrics are not achievable."""

    winch_params: Optional[WinchParams] = None
    trajectory: TrajectoryAngles
    depower: Optional[DepowerReply] = None
    min_turn_radius: Optional[float] = Field(
        default=None,
        description="Minimum turn radius [m] the returned path was optimized "
        "under (null = unconstrained); metrics.turn_radius_min_m is what the "
        "path actually achieves",
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

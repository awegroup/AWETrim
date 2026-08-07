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

    @model_validator(mode="after")
    def _check_forces(self):
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")
        return self


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
        default=None, description="Depower setting u_p"
    )
    reg_weight: Optional[float] = Field(
        default=None, description="Regularization weight"
    )
    detect_simple_bounds: Optional[bool] = Field(
        default=None, description="Solver flag"
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
    """Reply of POST /init — contains the InitParams struct (name, max_time,
    length, winch_params, inflow_conditions, trajectory + the solver knobs)
    plus server state. The trajectory is the fitted starting path (closed:
    last point equals the first, degrees); inflow_conditions echoes the
    accepted inflow."""

    name: str
    max_time: Optional[float] = None
    length: Optional[float] = Field(
        default=None, description="Accepted tether length / sphere radius [m]"
    )
    winch_params: Optional[WinchParams] = None
    inflow_conditions: Optional[InflowConditions] = None
    trajectory: TrajectoryAngles
    input_depower: Optional[float] = None
    reg_weight: Optional[float] = None
    detect_simple_bounds: Optional[bool] = None
    state: str
    n_points: Optional[int] = None
    session_id: str = "default"


class StepReply(BaseModel):
    """Reply of POST /step with wait=true — contains the StepParams struct
    (length, winch_params, trajectory) plus solve metadata. The trajectory is
    the OPTIMIZED path (closed, degrees)."""

    length: Optional[float] = Field(
        default=None, description="Tether length the pattern is anchored to [m]"
    )
    winch_params: Optional[WinchParams] = None
    trajectory: TrajectoryAngles
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
    # tension_tether_ground, input_steering, input_depower.
    table: Dict[str, List[Optional[float]]]
    spline: SplineBlock
    metrics: SolveMetrics
    optimized_parameters: dict
    timeseries: Optional[Dict[str, List[Optional[float]]]] = None
    energy_metrics: Optional[dict] = None

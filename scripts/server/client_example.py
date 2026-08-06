"""Python client for the AWETrim reelout flight-path optimizer (REST).

Mirrors client_example.jl / client_example.m: the same shared structs
(InitParams / StepParams), the same blocking call flow.

Dependencies: httpx (pip install httpx) — no awetrim import needed; this file
is self-contained and can be copied into any project.

Contract: POST /init receives and replies InitParams; POST /step receives and
replies StepParams (blocking — the reply contains the OPTIMIZED trajectory).
Reply trajectories are CLOSED (last point == first) with the same number of
points you sent. Angles in DEGREES; the tether length travels as a sibling
JSON field next to the structs.

Winch coupling: the optimizer maps v_set = k_v*sqrt(force) onto its radial
force model, so the optimized path assumes exactly your winch behavior.

Infeasibility: an impossible request returns HTTP 422 with the solver
message; the previous trajectory remains available under GET /trajectory.
"""

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

import httpx

# ---------------------------------------------------------------------------
# The shared structs (as agreed)
# ---------------------------------------------------------------------------
@dataclass
class InflowConditions:
    wind_speed: float                   # in m/s at 6 m height
    wind_direction: float               # in degrees, 0 = North, 90 = East
    profile_law: int                    # 0=CONST, 1=EXP, 2=LOG, 3=EXPLOG, 4=CUSTOM_LOG, 5=CUSTOM_EXP, 6=CUSTOM_JET
    # the custom profiles are fitted using the heights and speeds given in the heights and speeds fields
    # CUSTOM_JET: u(z) = u_bg(z) + U_J * exp(-(z - z_c)^2 / (2*sigma^2))
    # the following fields are optional; leave them at None to get the server
    # defaults given in the comments
    alpha: Optional[float] = None       # exponent of the wind profile law, default: 0.08163
    z0: Optional[float] = None          # surface roughness [m], default: 0.0002
    turbulence: Optional[float] = None  # in [0, 1], 0 = no turbulence, 1 = full turbulence, default: 0.0
    heights: Optional[List[float]] = None  # heights at which the wind speed is given, default: [6.0]
    speeds: Optional[List[float]] = None   # wind speeds at the given heights, default: [wind_speed]


@dataclass
class WinchParams:
    mode: str        # "reelout" ("reelin" not supported yet)
    k_v: float       # v_set = k_v * sqrt(force)
    v_max: float     # maximum winch speed [m/s]
    f_min: float     # minimum winch force [N]
    f_max: float     # maximum winch force [N]


@dataclass
class Trajectory:
    azimuth: List[float]    # [deg], azimuth 0 = downwind
    elevation: List[float]  # [deg], from the ground plane


@dataclass
class InitParams:
    name: str
    max_time: float
    winch_params: WinchParams
    inflow_conditions: InflowConditions
    trajectory: Trajectory


@dataclass
class StepParams:
    winch_params: WinchParams
    trajectory: Trajectory


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------
BASE = "http://127.0.0.1:8000"
_http = httpx.Client(base_url=BASE, timeout=600.0)  # blocking solves


def _payload(params) -> dict:
    """dataclass -> JSON dict, dropping unset optional fields so the server
    applies its documented defaults."""
    return {
        key: value
        for key, value in asdict(params).items()
        if value is not None
    }


def _post(path: str, payload: dict) -> dict:
    response = _http.post(path, json=payload)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(f"POST {path} -> {response.status_code}: {detail}")
    return response.json()


def _init_params(reply: dict) -> InitParams:
    return InitParams(
        name=reply["name"],
        max_time=reply["max_time"],
        winch_params=WinchParams(**reply["winch_params"]),
        inflow_conditions=InflowConditions(**reply["inflow_conditions"]),
        trajectory=Trajectory(**reply["trajectory"]),
    )


def init(
    params: InitParams,
    *,
    distance_radial: Optional[float] = None,
    **extra,
) -> InitParams:
    """Send InitParams (+ tether length, solver knobs) -> InitParams."""
    payload = _payload(params)
    payload["inflow_conditions"] = _payload(params.inflow_conditions)
    if distance_radial is not None:
        payload["distance_radial"] = distance_radial
    payload.update(extra)  # e.g. sim_parameters={...}
    return _init_params(_post("/init", payload))


def step(
    params: StepParams,
    *,
    inflow_conditions: Optional[InflowConditions] = None,
    distance_radial: Optional[float] = None,
) -> StepParams:
    """Send StepParams (+ optional new inflow / tether length) -> receive
    StepParams with the OPTIMIZED trajectory. Blocks ~10 s - 2 min."""
    payload = _payload(params)
    if inflow_conditions is not None:
        payload["inflow_conditions"] = _payload(inflow_conditions)
    if distance_radial is not None:
        payload["distance_radial"] = distance_radial
    reply = _post("/step", payload)
    return StepParams(
        winch_params=WinchParams(**reply["winch_params"]),
        trajectory=Trajectory(**reply["trajectory"]),
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A starting figure-eight, 1000 points, repetitive (last == first)
    n = 1000
    s = [2.0 * math.pi * i / (n - 1) for i in range(n)]
    guess = Trajectory(
        azimuth=[20.0 * math.sin(si) for si in s],
        elevation=[22.0 + 8.0 * math.sin(2.0 * si) for si in s],
    )

    # k_v example: v = k_v*sqrt(F) -> at 8400 N this winch reels ~10 m/s.
    # Too-stiff values (e.g. 0.02 -> only ~1.8 m/s at max force) make the
    # optimization infeasible and the server replies 422.
    winch = WinchParams(mode="reelout", k_v=0.11, v_max=8.0,
                        f_min=1000.0, f_max=8400.0)

    # 5.2 m/s at 6 m height, wind from the west, logarithmic profile
    # (~8 m/s at 100 m with this roughness).
    inflow = InflowConditions(wind_speed=5.2, wind_direction=270.0,
                              profile_law=2, z0=0.03)

    reply = init(
        InitParams("python-sim-1", 600.0, winch, inflow, guess),
        distance_radial=200.0,
        sim_parameters={"input_depower": 1.6, "reg_weight": 1.0,
                        "detect_simple_bounds": True},
    )
    print(f"init ok: {reply.name}, "
          f"starting path {len(reply.trajectory.azimuth)} pts")

    # First optimization (blocking; reply contains the optimized path)
    result = step(StepParams(winch, reply.trajectory))
    elevation = result.trajectory.elevation
    print(f"optimized: {len(result.trajectory.azimuth)} pts, "
          f"elevation {min(elevation):.1f} - {max(elevation):.1f} deg")

    # ... fly the kite along result.trajectory ...

    # Later, refresh with the current conditions from your simulation. A
    # measured profile is sent as a CUSTOM_* law: the heights/speeds samples
    # are fitted (here: log law + Gaussian jet).
    result = step(
        StepParams(winch, result.trajectory),
        inflow_conditions=InflowConditions(
            wind_speed=8.4, wind_direction=265.0, profile_law=6,
            heights=[10.0, 50.0, 100.0, 200.0, 300.0],
            speeds=[5.5, 7.4, 8.0, 9.3, 8.6],
        ),
        distance_radial=220.0,  # current tether length
    )
    print("refreshed trajectory received")

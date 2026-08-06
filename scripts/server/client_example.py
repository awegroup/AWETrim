"""Python client for the AWETrim reelout flight-path optimizer (REST).

Mirrors client_example.jl / client_example.m: the same shared structs
(InitParams / StepParams), the same blocking call flow.

Dependencies: httpx (pip install httpx) — no awetrim import needed; this file
is self-contained and can be copied into any project.

Contract: POST /init receives and replies InitParams; POST /step receives and
replies StepParams (blocking — the reply contains the OPTIMIZED trajectory).
Reply trajectories are CLOSED (last point == first) with the same number of
points you sent. Angles in DEGREES; wind and tether length travel as sibling
JSON fields next to the structs.

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


def _post(path: str, payload: dict) -> dict:
    response = _http.post(path, json=payload)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(f"POST {path} -> {response.status_code}: {detail}")
    return response.json()


def init(
    params: InitParams,
    *,
    wind: dict,
    distance_radial: Optional[float] = None,
    **extra,
) -> InitParams:
    """Send InitParams (+ wind, tether length, solver knobs) -> InitParams."""
    payload = asdict(params)
    payload["wind"] = wind  # required by the optimizer
    if distance_radial is not None:
        payload["distance_radial"] = distance_radial
    payload.update(extra)  # e.g. sim_parameters={...}
    reply = _post("/init", payload)
    return InitParams(
        name=reply["name"],
        max_time=reply["max_time"],
        winch_params=WinchParams(**reply["winch_params"]),
        trajectory=Trajectory(**reply["trajectory"]),
    )


def step(
    params: StepParams,
    *,
    wind: Optional[dict] = None,
    distance_radial: Optional[float] = None,
) -> StepParams:
    """Send StepParams (+ optional wind / tether length) -> receive
    StepParams with the OPTIMIZED trajectory. Blocks ~10 s - 2 min."""
    payload = asdict(params)
    if wind is not None:
        payload["wind"] = wind
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

    reply = init(
        InitParams("python-sim-1", 600.0, winch, guess),
        wind={"model_type": "logarithmic", "U_ref": 8.0,
              "z_ref": 100.0, "z0": 0.03},
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

    # Later, refresh with the current conditions from your simulation:
    result = step(
        StepParams(winch, result.trajectory),
        wind={"model_type": "tabulated",
              "heights": [10.0, 100.0, 300.0],
              "speeds": [5.5, 8.0, 9.3]},
        distance_radial=220.0,  # current tether length
    )
    print("refreshed trajectory received")

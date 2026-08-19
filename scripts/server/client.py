"""Minimal Python client for the AWETrim reelout optimization server.

Wraps the /init, /step, /status, /trajectory endpoints so callers never deal
with HTTP or polling directly:

    from client import ReeloutClient

    client = ReeloutClient("http://127.0.0.1:8000")
    client.init(wind={"model_type": "logarithmic", "U_ref": 8.0})
    trajectory = client.optimize()                      # step + wait, one call
    ...
    trajectory = client.optimize(                       # later: refresh
        wind={"model_type": "tabulated",
              "heights": [10, 100, 300], "speeds": [5.5, 8.0, 9.3]},
        distance_radial=220.0,
    )

Requires only httpx (installed with the [dev] or [server] extras); runnable
from anywhere, no awetrim import needed.
"""

import time
from typing import Optional

import httpx


class SolveFailedError(RuntimeError):
    """The optimization step ended in the 'failed' state."""


class ReeloutClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    # -- thin endpoint wrappers -------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health")

    def status(self) -> dict:
        return self._request("GET", "/status")

    def init(self, wind: dict, **options) -> dict:
        """POST /init. `options` maps to the InitRequest fields, e.g.
        initial_guess=..., n_points=..., sim_parameters={...},
        depower={"mode": "fixed"|"optimize"|"profile", "value": 1.6},
        min_turn_radius=11.35 (metres; the path will not turn tighter)."""
        return self._request("POST", "/init", json={"wind": wind, **options})

    def step(
        self,
        wind: Optional[dict] = None,
        distance_radial: Optional[float] = None,
        max_iter: Optional[int] = None,
        depower: Optional[dict] = None,
        min_turn_radius: Optional[float] = None,
    ) -> dict:
        """POST /step (non-blocking). Returns the 202 acknowledgement.
        ``min_turn_radius`` [m] updates the path's minimum turn radius
        (0 removes the constraint); omit to keep the /init setting."""
        payload = {"wait": False}  # keep this client's step/wait split
        if wind is not None:
            payload["wind"] = wind
        if distance_radial is not None:
            payload["distance_radial"] = distance_radial
        if max_iter is not None:
            payload["max_iter"] = max_iter
        if depower is not None:
            payload["depower"] = depower
        if min_turn_radius is not None:
            payload["min_turn_radius"] = min_turn_radius
        return self._request("POST", "/step", json=payload)

    def trajectory(self, resimulate: bool = False) -> dict:
        return self._request(
            "GET", "/trajectory", params={"resimulate": resimulate}
        )

    def reset(self) -> dict:
        return self._request("POST", "/reset")

    # -- conveniences -------------------------------------------------------
    def wait(self, timeout: float = 600.0, poll_interval: float = 1.0) -> dict:
        """Block until the running solve finishes; returns the final status."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status()
            if status["state"] in ("converged", "failed"):
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"solve still running after {timeout:.0f}s")

    def optimize(self, timeout: float = 600.0, **step_kwargs) -> dict:
        """step + wait + trajectory in one blocking call.

        Returns the TrajectoryResponse dict. Its ``table["input_depower"]``
        column and ``optimized_parameters["input_depower"]`` give the depower
        the path was optimized for — fly it, or the reported power is not
        achievable. Raises SolveFailedError if the solve fails (the previous
        trajectory, if any, remains available via .trajectory()).
        """
        self.step(**step_kwargs)
        status = self.wait(timeout=timeout)
        if status["state"] == "failed":
            raise SolveFailedError(status["last_error"])
        return self.trajectory()

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            raise RuntimeError(f"{method} {path} -> {response.status_code}: {detail}")
        return response.json()


if __name__ == "__main__":
    # Demo: full cycle against a locally running server.
    client = ReeloutClient()
    client.health()

    print("initializing session ...")
    client.init(
        wind={"model_type": "logarithmic", "U_ref": 8.0, "z_ref": 100.0, "z0": 0.03},
        initial_guess={"curve_type": "lissajous"},
        depower={"mode": "optimize", "value": 1.6},
        sim_parameters={"reg_weight": 1.0, "detect_simple_bounds": True},
    )

    print("optimizing (cold start) ...")
    trajectory = client.optimize()
    metrics = trajectory["metrics"]
    print(f"  step {trajectory['step_index']}: "
          f"{metrics['avg_power_W'] / 1e3:.2f} kW average power, "
          f"{metrics['total_time_s']:.1f} s pattern")
    print(f"  fly it at depower "
          f"{trajectory['optimized_parameters']['input_depower']:.4f}")

    print("re-optimizing with depower pinned at the client's own setting ...")
    trajectory = client.optimize(depower={"mode": "fixed", "value": 1.6})
    print(f"  {trajectory['metrics']['avg_power_W'] / 1e3:.2f} kW at "
          f"l_dp = {trajectory['table']['input_depower'][0]:.4f} "
          f"(what a client that cannot command depower actually gets)")

    print("refreshing with new wind + tether length (warm start) ...")
    trajectory = client.optimize(
        wind={"model_type": "tabulated",
              "heights": [10.0, 100.0, 300.0], "speeds": [5.5, 8.0, 9.3]},
        distance_radial=220.0,
    )
    metrics = trajectory["metrics"]
    print(f"  step {trajectory['step_index']}: "
          f"{metrics['avg_power_W'] / 1e3:.2f} kW average power, "
          f"r0 = {trajectory['spline']['r0']:.0f} m")

    table = trajectory["table"]
    print(f"  guidance table: {len(table['t'])} nodes, columns {sorted(table)}")

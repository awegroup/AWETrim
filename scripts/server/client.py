"""Minimal Python client for the AWETrim reelout optimization server.

Wraps the /init, /step, /status, /trajectory endpoints so callers never deal
with HTTP or polling directly:

    from client import ReeloutClient

    client = ReeloutClient("http://127.0.0.1:8000")
    client.init(inflow_conditions={"wind_speed": 5.2, "profile_law": 2})
    trajectory = client.optimize()                      # step + wait, one call
    ...
    trajectory = client.optimize(                       # later: refresh
        inflow_conditions={"wind_speed": 8.4, "wind_direction": 265.0,
                           "profile_law": 6,
                           "heights": [10, 50, 100, 200, 300],
                           "speeds": [5.5, 7.4, 8.0, 9.3, 8.6]},
        length=220.0,
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

    def init(self, inflow_conditions: dict, **options) -> dict:
        """POST /init. `options` maps to the InitRequest fields, e.g.
        initial_guess=..., n_points=..., length=..., input_depower=..."""
        return self._request(
            "POST", "/init",
            json={"inflow_conditions": inflow_conditions, **options},
        )

    def step(
        self,
        inflow_conditions: Optional[dict] = None,
        length: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> dict:
        """POST /step (non-blocking). Returns the 202 acknowledgement."""
        payload = {"wait": False}  # keep this client's step/wait split
        if inflow_conditions is not None:
            payload["inflow_conditions"] = inflow_conditions
        if length is not None:
            payload["length"] = length
        if max_iter is not None:
            payload["max_iter"] = max_iter
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

        Returns the TrajectoryResponse dict. Raises SolveFailedError if the
        solve fails (the previous trajectory, if any, remains available via
        .trajectory()).
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
        # 5.2 m/s at 6 m height, wind from the west, logarithmic profile
        inflow_conditions={"wind_speed": 5.2, "wind_direction": 270.0,
                           "profile_law": 2, "z0": 0.03},
        initial_guess={"curve_type": "lissajous"},
        input_depower=1.6, reg_weight=1.0, detect_simple_bounds=True,
    )

    print("optimizing (cold start) ...")
    trajectory = client.optimize()
    metrics = trajectory["metrics"]
    print(f"  step {trajectory['step_index']}: "
          f"{metrics['avg_power_W'] / 1e3:.2f} kW average power, "
          f"{metrics['total_time_s']:.1f} s pattern")

    print("refreshing with new wind + tether length (warm start) ...")
    trajectory = client.optimize(
        # a measured profile: CUSTOM_JET fits the heights/speeds samples
        inflow_conditions={"wind_speed": 8.4, "wind_direction": 265.0,
                           "profile_law": 6,
                           "heights": [10.0, 50.0, 100.0, 200.0, 300.0],
                           "speeds": [5.5, 7.4, 8.0, 9.3, 8.6]},
        length=220.0,
    )
    metrics = trajectory["metrics"]
    print(f"  step {trajectory['step_index']}: "
          f"{metrics['avg_power_W'] / 1e3:.2f} kW average power, "
          f"r0 = {trajectory['spline']['r0']:.0f} m")

    table = trajectory["table"]
    print(f"  guidance table: {len(table['t'])} nodes, columns {sorted(table)}")

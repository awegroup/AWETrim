"""HTTP-level tests for the reelout optimization server (no IPOPT)."""

import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import awetrim.server.session as session_mod
from awetrim.server.app import create_app
from awetrim.server.session import ReeloutSession

N_NODES = 12  # must satisfy the InitRequest n_points >= 10 bound
R0 = 200.0


def _fake_result():
    n = N_NODES
    return SimpleNamespace(
        optimized_trajectory={
            "s": np.linspace(0.0, 2.0 * math.pi * 0.95, n),
            "s_dot": np.full(n, 0.8),
            "input_steering": np.zeros(n),
            "speed_radial": np.full(n, 1.5),
            "distance_radial": np.linspace(R0, R0 + 15.0, n),
            "tension_tether_ground": np.full(n, 5000.0),
            "input_depower": np.full(n, 1.6),
        },
        energy_objective=1.0e6,
        total_time=25.0,
        solution=SimpleNamespace(value=lambda expr: expr),
    )


class StubPhase:
    latest = None

    def __init__(self, *, system_model, pattern_config, start_state):
        self.system_model = system_model
        self.pattern_config = pattern_config
        self.start_state = start_state
        self.solve_started = threading.Event()
        self.release = threading.Event()
        self.results = []
        StubPhase.latest = self

    def run_simulation_opti(self, **kwargs):
        self.solve_started.set()
        assert self.release.wait(timeout=5.0)
        self.release.clear()
        self.solve_started.clear()
        return self.results.pop(0)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    system_yaml = tmp_path / "system.yaml"
    cycle_yaml = tmp_path / "cycle.yaml"
    system_yaml.write_text("stub: true\n")
    cycle_yaml.write_text("stub: true\n")

    config = {
        "pattern_type": "spline_periodic",
        "path_parameters": {
            "r0": R0,
            "M": 4,
            "C_phi": [0.3, 0.0, -0.3, 0.0],
            "C_beta": [0.35, 0.45, 0.35, 0.25],
            "s_init": 0.0,
            "s_final": 2.0 * math.pi,
            "downloops": True,
        },
        "radial_parameters": {},
        "sim_parameters": {},
    }
    monkeypatch.setattr(
        session_mod,
        "create_system_model_from_yaml",
        lambda yaml_path: SimpleNamespace(wind=None),
    )
    monkeypatch.setattr(
        session_mod,
        "load_cycle_config_from_yaml",
        lambda path: (dict(config), None),
    )
    monkeypatch.setattr(session_mod, "Phase", StubPhase)

    test_client = TestClient(create_app(session_factory=ReeloutSession))
    test_client.init_payload = {
        "wind": {"model_type": "logarithmic", "U_ref": 8.0},
        "system_config_path": str(system_yaml),
        "cycle_config_path": str(cycle_yaml),
        "n_points": N_NODES,
    }
    return test_client


def _poll_state(client, target, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/status").json()["state"]
        if state == target:
            return state
        time.sleep(0.01)
    return client.get("/status").json()["state"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_endpoints_before_init(client):
    assert client.get("/status").json()["state"] == "uninitialized"
    assert client.post("/step", json={}).status_code == 409
    assert client.get("/trajectory").status_code == 404


def test_init_validation_errors(client):
    payload = dict(client.init_payload)
    payload["wind"] = {"model_type": "gaussian", "U_ref": 8.0}
    assert client.post("/init", json=payload).status_code == 422

    payload = dict(client.init_payload)
    payload["system_config_path"] = "does/not/exist.yaml"
    response = client.post("/init", json=payload)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]


def test_full_flow(client):
    # init: reply contains the InitParams struct incl. the starting path
    response = client.post("/init", json=client.init_payload)
    assert response.status_code == 200
    reply = response.json()
    assert reply["state"] == "ready"
    assert reply["name"] == "reelout-optimization"
    trajectory = reply["trajectory"]
    # no client trajectory sent -> reply resolution follows n_points
    assert len(trajectory["azimuth"]) == len(trajectory["elevation"]) == N_NODES
    # periodic: closed curve
    assert trajectory["azimuth"][0] == pytest.approx(trajectory["azimuth"][-1], abs=1e-9)

    # async step (wait=false) -> 202, then busy while solving
    response = client.post("/step", json={"wait": False})
    assert response.status_code == 202
    assert response.json() == {"state": "solving", "step_index": 1}
    phase = StubPhase.latest
    assert phase.solve_started.wait(timeout=5.0)
    assert client.post("/step", json={}).status_code == 409
    assert client.post("/reset").status_code == 409
    # trajectory not yet available (no prior success)
    assert client.get("/trajectory").status_code == 404

    # release the solve -> converged
    phase.results.append(_fake_result())
    phase.release.set()
    assert _poll_state(client, "converged") == "converged"
    status = client.get("/status").json()
    assert status["successful_steps"] == 1
    assert status["metrics"]["avg_power_W"] == pytest.approx(1.0e6 / 25.0)

    # trajectory: guidance table with n_points rows per column
    trajectory = client.get("/trajectory").json()
    assert trajectory["step_index"] == 1
    assert trajectory["n_points"] == N_NODES
    for column in ("t", "azimuth", "elevation", "azimuth_dot",
                   "elevation_dot", "distance_radial", "speed_radial"):
        values = trajectory["table"][column]
        assert len(values) == N_NODES
        assert all(isinstance(v, float) for v in values)
    assert trajectory["spline"]["M"] == 4

    # second step with new wind + tether length
    response = client.post(
        "/step",
        json={
            "wind": {
                "model_type": "tabulated",
                "heights": [10.0, 300.0],
                "speeds": [5.0, 9.0],
            },
            "distance_radial": R0 + 20.0,
            "wait": False,
        },
    )
    assert response.status_code == 202
    phase = StubPhase.latest
    assert phase.solve_started.wait(timeout=5.0)
    # old trajectory still served while solving
    assert client.get("/trajectory").status_code == 200
    # ... but the (slow) re-simulation variant is rejected mid-solve
    assert client.get("/trajectory", params={"resimulate": True}).status_code == 409

    phase.results.append(None)  # this solve fails
    phase.release.set()
    assert _poll_state(client, "failed") == "failed"
    status = client.get("/status").json()
    assert "did not converge" in status["last_error"]
    # last good trajectory survives the failure
    assert client.get("/trajectory").json()["step_index"] == 1

    # reset
    assert client.post("/reset").status_code == 200
    assert client.get("/status").json()["state"] == "uninitialized"


def _closed_figure_eight_deg(n=200):
    s = np.linspace(0.0, 2.0 * math.pi, n, endpoint=True)  # closed input
    azimuth = np.degrees(0.3 * np.sin(s))
    elevation = np.degrees(0.35 + 0.12 * np.sin(2.0 * s))
    return {"azimuth": azimuth.tolist(), "elevation": elevation.tolist()}


def test_cosim_contract_blocking_step(client):
    """init/step replies contain the InitParams/StepParams structs."""
    winch = {"mode": "reelout", "k_v": 0.02, "v_max": 8.0,
             "f_min": 1000.0, "f_max": 8400.0}
    payload = dict(client.init_payload)
    payload.update(
        name="uwe-sim-1",
        max_time=600.0,
        winch_params=winch,
        trajectory=_closed_figure_eight_deg(),
    )
    reply = client.post("/init", json=payload)
    assert reply.status_code == 200
    reply = reply.json()
    assert reply["name"] == "uwe-sim-1"
    assert reply["max_time"] == 600.0
    assert reply["winch_params"]["k_v"] == 0.02
    # starting path echoed at the client's resolution, closed
    assert len(reply["trajectory"]["azimuth"]) == 200
    # fitted path reproduces the sent figure-eight
    # (loose tolerance: the fixture pattern has only M=4 control points)
    assert max(reply["trajectory"]["azimuth"]) == pytest.approx(
        math.degrees(0.3), rel=0.15
    )

    # blocking step: pre-release the stub solve so the request thread
    # does not deadlock, then expect a StepParams-shaped 200 reply
    phase = StubPhase.latest
    phase.results.append(_fake_result())
    phase.release.set()
    reply = client.post("/step", json={})
    assert reply.status_code == 200
    reply = reply.json()
    assert reply["state"] == "converged"
    assert reply["step_index"] == 1
    assert reply["winch_params"]["f_max"] == 8400.0
    assert len(reply["trajectory"]["azimuth"]) == 200
    assert reply["metrics"]["avg_power_W"] == pytest.approx(1.0e6 / 25.0)

    # infeasible blocking step -> 422, previous state kept queryable
    phase.results.append(None)
    phase.release.set()
    reply = client.post("/step", json={})
    assert reply.status_code == 422
    assert "not converged" in reply.json()["detail"]
    assert client.get("/trajectory").status_code == 200


def test_reelin_mode_rejected(client):
    payload = dict(client.init_payload)
    payload["winch_params"] = {"mode": "reelin", "k_v": 0.02, "v_max": 8.0,
                               "f_min": 1000.0, "f_max": 8400.0}
    reply = client.post("/init", json=payload)
    assert reply.status_code == 400
    assert "reelout" in reply.json()["detail"]


def test_depower_round_trips_through_init_and_step(client):
    payload = dict(client.init_payload)
    payload["depower"] = {"mode": "fixed", "value": 1.45}
    reply = client.post("/init", json=payload)
    assert reply.status_code == 200
    assert reply.json()["depower"] == {
        "mode": "fixed", "value": 1.45, "profile": None
    }

    phase = StubPhase.latest
    phase.results.append(_fake_result())
    phase.release.set()
    step = client.post("/step", json={}).json()
    assert step["depower"]["mode"] == "fixed"
    assert step["depower"]["value"] == pytest.approx(1.45)

    # /step may switch the mode mid-session
    phase.results.append(_fake_result())
    phase.release.set()
    step = client.post(
        "/step", json={"depower": {"mode": "optimize", "value": 1.6}}
    ).json()
    assert step["depower"]["mode"] == "optimize"


def test_depower_omitted_keeps_the_legacy_behaviour(client):
    payload = dict(client.init_payload)
    payload["sim_parameters"] = {"input_depower": 1.6}
    reply = client.post("/init", json=payload)
    assert reply.status_code == 200
    # default optimization_params include input_depower -> derived mode
    assert reply.json()["depower"] == {
        "mode": "optimize", "value": 1.6, "profile": None
    }


def test_unknown_depower_mode_is_rejected(client):
    payload = dict(client.init_payload)
    payload["depower"] = {"mode": "auto", "value": 1.6}
    assert client.post("/init", json=payload).status_code == 422

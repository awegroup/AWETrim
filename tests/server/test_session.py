"""State-machine tests for ReeloutSession with a stubbed Phase (no IPOPT)."""

import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic")

import awetrim.server.session as session_mod
from awetrim.server.session import (
    NoTrajectoryError,
    ReeloutSession,
    SessionBusyError,
    SessionNotInitializedError,
    SessionState,
)

N_NODES = 8
R0 = 200.0


def _fake_trajectory(n=N_NODES, r0=R0):
    return {
        "s": np.linspace(0.0, 2.0 * math.pi * 0.95, n),
        "s_dot": np.full(n, 0.8),
        "input_steering": np.zeros(n),
        "speed_radial": np.full(n, 1.5),
        "distance_radial": np.linspace(r0, r0 + 15.0, n),
        "tension_tether_ground": np.full(n, 5000.0),
        "input_depower": np.full(n, 1.6),
    }


def _fake_result():
    return SimpleNamespace(
        optimized_trajectory=_fake_trajectory(),
        energy_objective=1.0e6,
        total_time=25.0,
        solution=SimpleNamespace(value=lambda expr: expr),
    )


class StubPhase:
    """Blocking, scriptable stand-in for awetrim.timeseries.phase.Phase."""

    def __init__(self, *, system_model, pattern_config, start_state):
        self.system_model = system_model
        self.pattern_config = pattern_config
        self.start_state = start_state
        self.solve_started = threading.Event()
        self.release = threading.Event()
        self.results = []  # one entry per expected solve; Exception -> raised
        self.calls = []

    def run_simulation_opti(self, **kwargs):
        self.calls.append(kwargs)
        self.solve_started.set()
        assert self.release.wait(timeout=5.0), "test never released the solve"
        self.release.clear()
        self.solve_started.clear()
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _base_config():
    return {
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


@pytest.fixture()
def patched_session(monkeypatch, tmp_path):
    """ReeloutSession with stubbed model/config/Phase builders."""
    system_yaml = tmp_path / "system.yaml"
    cycle_yaml = tmp_path / "cycle.yaml"
    system_yaml.write_text("stub: true\n")
    cycle_yaml.write_text("stub: true\n")

    monkeypatch.setattr(
        session_mod,
        "create_system_model_from_yaml",
        lambda yaml_path: SimpleNamespace(wind=None),
    )
    monkeypatch.setattr(
        session_mod,
        "load_cycle_config_from_yaml",
        lambda path: (_base_config(), None),
    )
    monkeypatch.setattr(session_mod, "Phase", StubPhase)

    sess = ReeloutSession()
    init_config = {
        "wind": {"model_type": "uniform", "U_ref": 8.0, "direction_wind": 0.0},
        "initial_guess": None,
        "system_config_path": str(system_yaml),
        "cycle_config_path": str(cycle_yaml),
        "optimization_params": ["C_phi", "C_beta", "input_depower"],
        "target": "power",
        "n_points": N_NODES,
        "sim_parameters": {"input_depower": 1.6},
    }
    return sess, init_config


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _run_step_to_success(sess, **step_kwargs):
    phase = sess.phase
    phase.results.append(_fake_result())
    sess.step(**step_kwargs)
    assert phase.solve_started.wait(timeout=5.0)
    phase.release.set()
    assert _wait_until(lambda: sess.state == SessionState.CONVERGED)


def test_step_before_init_raises():
    sess = ReeloutSession()
    with pytest.raises(SessionNotInitializedError):
        sess.step()
    with pytest.raises(NoTrajectoryError):
        sess.trajectory()


def test_init_builds_phase_and_ready_state(patched_session):
    sess, config = patched_session
    sess.init(config)
    assert sess.state == SessionState.READY
    assert sess.phase.start_state["distance_radial"] == R0
    sim = sess.phase.pattern_config["sim_parameters"]
    assert sim["n_points"] == N_NODES
    assert sim["input_depower"] == 1.6
    assert sim["expand_nlp"] is True  # uniform wind -> SX expansion allowed
    # Wind model actually attached and numeric
    assert sess.system_model.wind is not None
    status = sess.status()
    assert status["state"] == "ready"
    assert status["wind"]["model_type"] == "uniform"


def test_step_success_produces_guidance_record(patched_session):
    sess, config = patched_session
    sess.init(config)
    phase = sess.phase
    phase.results.append(_fake_result())

    step_index = sess.step()
    assert step_index == 1
    assert phase.solve_started.wait(timeout=5.0)
    assert sess.state == SessionState.SOLVING
    with pytest.raises(SessionBusyError):
        sess.step()
    # /trajectory has nothing yet (no prior success)
    with pytest.raises(NoTrajectoryError):
        sess.trajectory()

    phase.release.set()
    assert _wait_until(lambda: sess.state == SessionState.CONVERGED)
    assert sess.successful_steps == 1
    assert phase.calls[0]["warm_start"] is False

    traj = sess.trajectory()
    table = traj["table"]
    for column in (
        "t",
        "s",
        "azimuth",
        "elevation",
        "azimuth_dot",
        "elevation_dot",
        "distance_radial",
        "speed_radial",
        "s_dot",
        "tension_tether_ground",
        "input_steering",
        "input_depower",
    ):
        assert column in table, f"missing column {column}"
        assert len(table[column]) == N_NODES
    assert table["t"][0] == 0.0
    assert all(t2 > t1 for t1, t2 in zip(table["t"], table["t"][1:]))
    assert all(v is not None for v in table["azimuth_dot"])
    assert traj["spline"]["M"] == 4
    assert len(traj["spline"]["C_phi"]) == 4
    assert traj["metrics"]["energy_J"] == pytest.approx(1.0e6)
    assert traj["metrics"]["avg_power_W"] == pytest.approx(1.0e6 / 25.0)


def test_second_step_warm_starts_and_updates_radius(patched_session):
    sess, config = patched_session
    sess.init(config)
    _run_step_to_success(sess)

    phase = sess.phase
    phase._warm_start_trajectory = {
        "distance_radial": np.full(N_NODES, R0),
        "s_dot": np.full(N_NODES, 0.8),
    }
    phase.results.append(_fake_result())
    sess.step(
        wind={"model_type": "tabulated", "heights": [10.0, 300.0],
              "speeds": [5.0, 9.0], "direction_wind": 0.0},
        distance_radial=R0 + 20.0,
    )
    assert phase.solve_started.wait(timeout=5.0)
    phase.release.set()
    assert _wait_until(lambda: sess.state == SessionState.CONVERGED)

    assert phase.calls[1]["warm_start"] is True
    assert phase.pattern_config["path_parameters"]["r0"] == R0 + 20.0
    assert phase.start_state["distance_radial"] == R0 + 20.0
    np.testing.assert_allclose(
        phase._warm_start_trajectory["distance_radial"], R0 + 20.0
    )
    # Tabulated wind must disable SX expansion
    assert phase.pattern_config["sim_parameters"]["expand_nlp"] is False
    assert sess.status()["wind"]["model_type"] == "tabulated"


def test_failed_step_keeps_previous_record(patched_session):
    sess, config = patched_session
    sess.init(config)
    _run_step_to_success(sess)
    first = sess.trajectory()

    phase = sess.phase
    phase.results.append(None)  # IPOPT failure -> run_simulation_opti None
    sess.step()
    assert phase.solve_started.wait(timeout=5.0)
    phase.release.set()
    assert _wait_until(lambda: sess.state == SessionState.FAILED)
    assert "did not converge" in sess.last_error
    assert sess.successful_steps == 1

    # Last good trajectory still served
    again = sess.trajectory()
    assert again["step_index"] == first["step_index"]

    # Retry after failure is allowed and exceptions are captured too
    phase.results.append(RuntimeError("boom"))
    sess.step()
    assert phase.solve_started.wait(timeout=5.0)
    phase.release.set()
    assert _wait_until(lambda: sess.state == SessionState.FAILED)
    assert "boom" in sess.last_error


def test_reset_clears_session(patched_session):
    sess, config = patched_session
    sess.init(config)
    _run_step_to_success(sess)
    sess.reset()
    assert sess.state == SessionState.UNINITIALIZED
    with pytest.raises(NoTrajectoryError):
        sess.trajectory()
    with pytest.raises(SessionNotInitializedError):
        sess.step()


def test_init_distance_radial_overrides_r0(patched_session):
    sess, config = patched_session
    config["distance_radial"] = 250.0
    config["initial_guess"] = {
        "curve_type": "lissajous",
        "M": 8,
        "n_fit": 100,
        "s_init": 0.0,
        "s_final": 2.0 * math.pi,
        "az_amp0": 0.3,
        "beta0": 0.35,
        "beta_amp0": 0.12,
        "downloops": True,
    }
    sess.init(config)
    assert sess.phase.pattern_config["path_parameters"]["r0"] == 250.0
    assert sess.phase.start_state["distance_radial"] == 250.0


def test_winch_params_map_to_quadratic_force_law(patched_session):
    sess, config = patched_session
    config["winch_params"] = {
        "mode": "reelout", "k_v": 0.02, "v_max": 8.0,
        "f_min": 1000.0, "f_max": 8400.0,
    }
    sess.init(config)
    radial = sess.phase.pattern_config["radial_parameters"]
    assert radial["reeling_strategy"] == "force"
    assert radial["force_model"] == "quadratic"
    # v = k_v*sqrt(F)  <=>  F = (1/k_v^2) * v^2
    assert radial["slope_winch_ro"] == pytest.approx(1.0 / 0.02**2)
    assert radial["offset_winch_ro"] == 0.0
    assert radial["min_tether_force"] == 1000.0
    assert radial["max_tether_force"] == 8400.0
    sim = sess.phase.pattern_config["sim_parameters"]
    assert sim["opti_limits_override"]["speed_radial"][1] == 8.0

    # reelin mode is rejected at init and at step
    config["winch_params"]["mode"] = "reelin"
    with pytest.raises(ValueError, match="reelout"):
        sess.init(config)


def test_trajectory_fit_round_trip(patched_session):
    """A sent degree-path is fitted and re-sampled close to the original."""
    sess, config = patched_session
    n = 200
    s = np.linspace(0.0, 2.0 * math.pi, n, endpoint=True)
    sent = {
        "azimuth": np.degrees(0.3 * np.sin(s)).tolist(),
        "elevation": np.degrees(0.35 + 0.12 * np.sin(2.0 * s)).tolist(),
    }
    config["trajectory"] = sent
    sess.init(config)
    echoed = sess.trajectory_degrees()
    assert len(echoed["azimuth"]) == n
    # closed curve
    assert echoed["azimuth"][0] == pytest.approx(echoed["azimuth"][-1], abs=1e-9)
    # shape preserved (loose: fixture spline has only M=4 control points)
    assert max(echoed["azimuth"]) == pytest.approx(math.degrees(0.3), rel=0.15)
    assert np.mean(echoed["elevation"]) == pytest.approx(
        math.degrees(0.35), rel=0.05
    )


def test_step_blocking_returns_step_params_shape(patched_session):
    sess, config = patched_session
    sess.init(config)
    phase = sess.phase
    phase.results.append(_fake_result())
    phase.release.set()  # let the inline solve pass straight through

    reply = sess.step_blocking()
    assert reply["state"] == "converged"
    assert reply["step_index"] == 1
    assert reply["metrics"]["avg_power_W"] == pytest.approx(1.0e6 / 25.0)
    # no client trajectory -> reply resolution follows n_points
    assert len(reply["trajectory"]["azimuth"]) == N_NODES

    # blocking failure raises and keeps the last good record
    phase.results.append(None)
    phase.release.set()
    from awetrim.server.session import SolveFailedError

    with pytest.raises(SolveFailedError):
        sess.step_blocking()
    assert sess.trajectory()["step_index"] == 1


def test_init_with_named_curve_initial_guess(patched_session):
    """Real B-spline fit: initial_guess replaces path_parameters."""
    sess, config = patched_session
    config["initial_guess"] = {
        "curve_type": "lissajous",
        "M": 8,
        "n_fit": 100,
        "s_init": 0.0,
        "s_final": 2.0 * math.pi,
        "az_amp0": 0.3,
        "beta0": 0.35,
        "beta_amp0": 0.12,
        "downloops": True,
    }
    sess.init(config)
    path = sess.phase.pattern_config["path_parameters"]
    assert path["M"] == 8
    assert len(path["C_phi"]) == 8
    assert path["r0"] == R0  # r0 carried over from the cycle config
    sim = sess.phase.pattern_config["sim_parameters"]
    assert sim["start_angle"] == 0.0
    assert sim["end_angle"] == pytest.approx(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Depower handling
# ---------------------------------------------------------------------------
def test_apply_depower_fixed_drops_it_from_the_nlp():
    sim, params = {}, ["C_phi", "C_beta", "input_depower"]
    mode = ReeloutSession._apply_depower(sim, params, {"mode": "fixed", "value": 1.6})
    assert mode == "fixed"
    assert "input_depower" not in params
    assert sim["input_depower"] == 1.6
    assert sim["optimize_depower_profile"] is False


def test_apply_depower_optimize_adds_it_and_seeds_the_value():
    sim, params = {}, ["C_phi", "C_beta"]
    mode = ReeloutSession._apply_depower(sim, params, {"mode": "optimize", "value": 1.4})
    assert mode == "optimize"
    assert params.count("input_depower") == 1
    assert sim["input_depower"] == 1.4


def test_apply_depower_profile_sets_the_per_node_flag():
    sim, params = {}, ["C_phi"]
    mode = ReeloutSession._apply_depower(sim, params, {"mode": "profile", "value": 1.6})
    assert mode == "profile"
    assert sim["optimize_depower_profile"] is True
    assert "input_depower" in params


def test_apply_depower_clears_a_stale_profile_when_leaving_profile_mode():
    sim = {"input_depower_profile": [1.3, 1.4], "optimize_depower_profile": True}
    params = ["input_depower"]
    ReeloutSession._apply_depower(sim, params, {"mode": "fixed", "value": 1.6})
    assert "input_depower_profile" not in sim


def test_apply_depower_none_derives_the_legacy_mode_without_mutating():
    sim = {"input_depower": 1.6}
    params = ["C_phi", "input_depower"]
    assert ReeloutSession._apply_depower(sim, params, None) == "optimize"
    assert sim == {"input_depower": 1.6} and "input_depower" in params

    params_fixed = ["C_phi"]
    assert ReeloutSession._apply_depower(sim, params_fixed, None) == "fixed"

    sim_profile = {"optimize_depower_profile": True}
    assert ReeloutSession._apply_depower(sim_profile, ["C_phi"], None) == "profile"


def test_apply_depower_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="depower mode"):
        ReeloutSession._apply_depower({}, [], {"mode": "auto"})


def test_init_depower_spec_overrides_optimization_params(patched_session):
    sess, config = patched_session
    config = dict(config, depower={"mode": "fixed", "value": 1.45})
    sess.init(config)
    assert "input_depower" not in sess._optimization_params
    assert sess.phase.pattern_config["sim_parameters"]["input_depower"] == 1.45
    assert sess.init_reply()["depower"] == {
        "mode": "fixed", "value": 1.45, "profile": None
    }


def test_step_reply_and_table_carry_the_depower(patched_session):
    sess, config = patched_session
    sess.init(dict(config, depower={"mode": "optimize", "value": 1.6}))
    phase = sess.phase
    phase.results.append(_fake_result())
    phase.release.set()
    reply = sess.step_blocking()
    assert reply["depower"]["mode"] == "optimize"
    assert reply["depower"]["value"] == pytest.approx(1.6)
    assert reply["depower"]["profile"] is None
    # the stub returns a per-node input_depower, so the table keeps that column
    table = sess.trajectory()["table"]
    assert len(table["input_depower"]) == N_NODES


def test_depower_column_is_constant_when_not_a_node_variable(patched_session):
    sess, config = patched_session
    sess.init(dict(config, depower={"mode": "fixed", "value": 1.45}))
    phase = sess.phase
    result = _fake_result()
    del result.optimized_trajectory["input_depower"]  # scalar depower: no column
    phase.results.append(result)
    phase.release.set()
    sess.step_blocking()
    column = sess.trajectory()["table"]["input_depower"]
    assert len(column) == N_NODES
    assert all(v == pytest.approx(1.45) for v in column)


def test_step_can_switch_the_depower_mode(patched_session):
    sess, config = patched_session
    sess.init(dict(config, depower={"mode": "optimize", "value": 1.6}))
    phase = sess.phase
    phase.results.append(_fake_result())
    phase.release.set()
    reply = sess.step_blocking(depower={"mode": "fixed", "value": 1.3})
    assert reply["depower"] == {"mode": "fixed", "value": 1.3, "profile": None}
    assert "input_depower" not in sess._optimization_params

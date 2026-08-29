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
        "inflow_conditions": {"wind_speed": 8.0, "profile_law": 0},  # CONST
        "initial_guess": None,
        "system_config_path": str(system_yaml),
        "cycle_config_path": str(cycle_yaml),
        "optimization_params": ["C_phi", "C_beta", "input_depower"],
        "target": "power",
        "n_points": N_NODES,
        "input_depower": 1.6,
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


def test_init_from_inflow_conditions(patched_session):
    """InflowConditions is resolved into the Wind model and echoed back."""
    sess, config = patched_session
    config = dict(config)
    config["inflow_conditions"] = {
        "wind_speed": 8.0, "wind_direction": 270.0, "profile_law": 3,
        "alpha": 0.1, "z0": 0.01, "turbulence": 0.2,
    }
    sess.init(config)
    assert sess.state == SessionState.READY
    # EXPLOG is analytic -> SX expansion stays enabled
    assert sess.phase.pattern_config["sim_parameters"]["expand_nlp"] is True
    status = sess.status()
    assert status["wind"]["model_type"] == "explog"
    assert status["wind"]["z_ref"] == 6.0
    assert status["inflow_conditions"]["profile_law"] == 3
    assert sess.init_reply()["inflow_conditions"]["wind_speed"] == 8.0
    # turbulence is accepted (and echoed) but does not reach the Wind model
    assert "turbulence" not in status["wind"]


def test_init_without_any_wind_raises(patched_session):
    sess, config = patched_session
    config = dict(config)
    del config["inflow_conditions"]
    with pytest.raises(ValueError, match="no wind given"):
        sess.init(config)


def test_step_updates_inflow_conditions(patched_session):
    sess, config = patched_session
    sess.init(config)
    _run_step_to_success(
        sess,
        inflow_conditions={"wind_speed": 9.0, "profile_law": 1, "alpha": 0.12},
    )
    status = sess.status()
    assert status["wind"] == {
        "model_type": "power_law", "U_ref": 9.0, "alpha": 0.12,
        "z_ref": 6.0, "direction_wind": pytest.approx(-math.pi / 2),
    }
    assert status["inflow_conditions"]["wind_speed"] == 9.0


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
        inflow_conditions={"wind_speed": 9.0, "profile_law": 2, "z0": 0.03},
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
    # every InflowConditions law is analytic -> SX expansion stays enabled
    assert phase.pattern_config["sim_parameters"]["expand_nlp"] is True
    assert sess.status()["wind"]["model_type"] == "logarithmic"


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


def test_init_length_overrides_r0(patched_session):
    sess, config = patched_session
    config["length"] = 250.0
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
        "mode": "reelout", "k_v": 0.02,
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

    # reelin mode is rejected at init and at step
    config["winch_params"]["mode"] = "reelin"
    with pytest.raises(ValueError, match="reelout"):
        sess.init(config)


def test_winch_use_awe_trim_defaults_to_zero_and_is_threaded_through(patched_session):
    sess, config = patched_session
    config["winch_params"] = {
        "mode": "reelout", "k_v": 0.02,
        "f_min": 1000.0, "f_max": 8400.0,
    }
    sess.init(config)
    radial = sess.phase.pattern_config["radial_parameters"]
    assert radial["use_awe_trim"] == 0.0
    assert "v_reel_in" not in radial
    assert "reel_in_beta" not in radial

    config["winch_params"] = {
        "mode": "reelout", "k_v": 0.02,
        "f_min": 1000.0, "f_max": 8400.0,
        "use_awe_trim": 1.0, "v_reel_in": -3.0, "reel_in_beta": 15.0,
    }
    sess.init(config)
    radial2 = sess.phase.pattern_config["radial_parameters"]
    assert radial2["use_awe_trim"] == pytest.approx(1.0)
    assert radial2["v_reel_in"] == pytest.approx(-3.0)
    assert radial2["reel_in_beta"] == pytest.approx(15.0)

    # use_awe_trim doesn't relax the reelout-only restriction on mode itself.
    config["winch_params"]["mode"] = "reelin"
    with pytest.raises(ValueError, match="reelout"):
        sess.init(config)


def test_optimize_k_v_off_by_default(patched_session):
    """The winch gain stays a constant unless the client asks for it."""
    sess, config = patched_session
    config["winch_params"] = {
        "mode": "reelout", "k_v": 0.04, "f_min": 1000.0, "f_max": 8400.0,
    }
    sess.init(config)
    assert "slope_winch_ro" not in sess._optimization_params
    override = sess.phase.pattern_config["sim_parameters"].get(
        "opti_limits_override", {}
    )
    assert "slope_winch_ro" not in override


def test_optimize_k_v_adds_bounded_design_variable(patched_session):
    """optimize_k_v makes the slope a variable, bracketed around the seed."""
    sess, config = patched_session
    k_v = 0.04
    config["winch_params"] = {
        "mode": "reelout", "k_v": k_v, "f_min": 1000.0, "f_max": 8400.0,
        "optimize_k_v": True,
    }
    sess.init(config)

    assert "slope_winch_ro" in sess._optimization_params
    lo, hi = sess.phase.pattern_config["sim_parameters"][
        "opti_limits_override"
    ]["slope_winch_ro"]
    # slope = 1/k_v^2 falls as k_v rises, so the bracket inverts.
    assert lo == pytest.approx(1.0 / (2.0 * k_v) ** 2)
    assert hi == pytest.approx(1.0 / (0.5 * k_v) ** 2)
    # The seed must lie strictly inside its own box -- the bug the old
    # DEFAULT_OPTI_LIMITS range caused was a seed outside it, clipped silently.
    seed = sess.phase.pattern_config["radial_parameters"]["slope_winch_ro"]
    assert lo < seed < hi


def test_optimize_k_v_honours_explicit_bounds(patched_session):
    sess, config = patched_session
    config["winch_params"] = {
        "mode": "reelout", "k_v": 0.04, "f_min": 1000.0, "f_max": 8400.0,
        "optimize_k_v": True, "k_v_bounds": [0.03, 0.10],
    }
    sess.init(config)
    lo, hi = sess.phase.pattern_config["sim_parameters"][
        "opti_limits_override"
    ]["slope_winch_ro"]
    assert lo == pytest.approx(1.0 / 0.10**2)
    assert hi == pytest.approx(1.0 / 0.03**2)


def test_step_can_toggle_k_v_optimization_off(patched_session):
    """A later /step turning it off drops the variable and its bracket."""
    sess, config = patched_session
    config["winch_params"] = {
        "mode": "reelout", "k_v": 0.04, "f_min": 1000.0, "f_max": 8400.0,
        "optimize_k_v": True,
    }
    sess.init(config)
    assert "slope_winch_ro" in sess._optimization_params

    _run_step_to_success(sess, winch_params={
        "mode": "reelout", "k_v": 0.04, "f_min": 1000.0, "f_max": 8400.0,
        "optimize_k_v": False,
    })
    assert "slope_winch_ro" not in sess._optimization_params
    override = sess.phase.pattern_config["sim_parameters"].get(
        "opti_limits_override", {}
    )
    assert "slope_winch_ro" not in override
    # ...and it is not appended twice when re-enabled.
    _run_step_to_success(sess, winch_params={
        "mode": "reelout", "k_v": 0.04, "f_min": 1000.0, "f_max": 8400.0,
        "optimize_k_v": True,
    })
    assert sess._optimization_params.count("slope_winch_ro") == 1


def test_winch_state_echoes_the_optimized_gain(patched_session):
    """The reply must carry the k_v the path assumes, not the one sent in."""
    sess, config = patched_session
    config["winch_params"] = {
        "mode": "reelout", "k_v": 0.04, "f_min": 1000.0, "f_max": 8400.0,
        "optimize_k_v": True,
    }
    sess.init(config)
    # Stand in for what run_opti writes back after a solve.
    sess.phase.pattern_config["radial_parameters"]["slope_winch_ro"] = 1.0 / 0.05**2

    assert sess.winch_state()["k_v"] == pytest.approx(0.05)
    assert sess.init_reply()["winch_params"]["k_v"] == pytest.approx(0.05)
    # The non-optimized fields are passed through untouched.
    assert sess.winch_state()["f_max"] == 8400.0


@pytest.mark.parametrize(
    "k_v_solved, at_bound", [(0.05, False), (0.08, True), (0.02, True)]
)
def test_optimized_parameters_flag_a_binding_bracket(
    patched_session, k_v_solved, at_bound
):
    """A gain sitting on its bracket is an edge, not an optimum: say so."""
    sess, config = patched_session
    config["winch_params"] = {
        "mode": "reelout", "k_v": 0.04, "f_min": 1000.0, "f_max": 8400.0,
        "optimize_k_v": True,  # default bracket -> k_v in [0.02, 0.08]
    }
    sess.init(config)
    sess.phase.pattern_config["radial_parameters"]["slope_winch_ro"] = (
        1.0 / k_v_solved**2
    )
    _run_step_to_success(sess)

    opt = sess.last_record.optimized_parameters
    assert opt["k_v"] == pytest.approx(k_v_solved)
    assert opt["k_v_at_bound"] is at_bound


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


def test_apply_min_turn_radius_sets_clears_and_keeps():
    sim = {}
    ReeloutSession._apply_min_turn_radius(sim, None)
    assert "min_turn_radius" not in sim  # omitted -> untouched
    ReeloutSession._apply_min_turn_radius(sim, 11.35)
    assert sim["min_turn_radius"] == pytest.approx(11.35)
    ReeloutSession._apply_min_turn_radius(sim, None)
    assert sim["min_turn_radius"] == pytest.approx(11.35)  # still kept
    ReeloutSession._apply_min_turn_radius(sim, 0)
    assert "min_turn_radius" not in sim  # 0 removes it
    with pytest.raises(ValueError, match="min_turn_radius"):
        ReeloutSession._apply_min_turn_radius(sim, -2.0)


def test_min_turn_radius_round_trips_init_step_and_replies(patched_session):
    sess, config = patched_session
    sess.init(dict(config, min_turn_radius=11.35))
    sim = sess.phase.pattern_config["sim_parameters"]
    assert sim["min_turn_radius"] == pytest.approx(11.35)
    assert sess.init_reply()["min_turn_radius"] == pytest.approx(11.35)

    # the stub result carries the NLP's own diagnostics. The first constrained
    # solve is staged (unconstrained cold, then constrained warm): two solves.
    phase = sess.phase
    result = _fake_result()
    result.optimized_trajectory["turn_radius"] = np.linspace(12.0, 40.0, N_NODES)
    result.turn_radius_min = 11.7
    phase.results.extend([_fake_result(), result])
    phase.release.set()
    phase.release.clear = lambda: None  # let both inline solves pass
    reply = sess.step_blocking()
    assert [c["warm_start"] for c in phase.calls] == [False, True]
    assert reply["min_turn_radius"] == pytest.approx(11.35)
    assert reply["metrics"]["turn_radius_min_m"] == pytest.approx(11.7)
    table = sess.trajectory()["table"]
    assert len(table["turn_radius"]) == N_NODES
    assert table["turn_radius"][0] == pytest.approx(12.0)

    # /step can change or remove the limit mid-session
    phase.results.append(_fake_result())
    phase.release.set()
    reply = sess.step_blocking(min_turn_radius=0)
    assert reply["min_turn_radius"] is None
    assert "min_turn_radius" not in sess.phase.pattern_config["sim_parameters"]
    # a result without diagnostics leaves the metric null
    assert reply["metrics"]["turn_radius_min_m"] is None


def test_first_constrained_solve_is_staged(patched_session):
    """With min_turn_radius set and no previous success, the session solves
    cold WITHOUT the constraint, then warm WITH it; later steps solve once.
    If a stage fails it falls back to one constrained cold solve from the
    original configuration."""
    sess, config = patched_session
    sess.init(dict(config, min_turn_radius=11.35))
    phase = sess.phase
    seen = []

    def _run(**kwargs):
        seen.append(
            (kwargs["warm_start"], phase.pattern_config["sim_parameters"].get("min_turn_radius"))
        )
        return StubPhase.run_simulation_opti(phase, **kwargs)

    phase.run_simulation_opti = _run
    phase.results.extend([_fake_result(), _fake_result()])
    phase.release.set()
    phase.release.clear = lambda: None
    sess.step_blocking()
    assert seen == [(False, None), (True, 11.35)]

    # second step: a single warm, constrained solve
    phase.results.append(_fake_result())
    sess.step_blocking()
    assert seen[-1] == (True, 11.35)
    assert len(seen) == 3

    # a failing unconstrained stage -> fallback: constrained cold solve on a
    # rebuilt Phase carrying the same config (still with the constraint)
    sess2, config2 = patched_session[0].__class__(), config
    sess2.init(dict(config2, min_turn_radius=11.35))
    phase2 = sess2.phase
    phase2.results.extend([None])  # stage 1 fails
    phase2.release.set()
    phase2.release.clear = lambda: None
    fallback = sess2.phase  # replaced by the fallback below
    from awetrim.server.session import SolveFailedError
    # the rebuilt Phase is a fresh StubPhase with no scripted result -> it
    # raises inside the worker, which the session reports as a failed step
    with pytest.raises(SolveFailedError):
        sess2.step_blocking()
    assert sess2.phase is not fallback  # rebuilt for the fallback stage
    assert sess2.phase.pattern_config["sim_parameters"]["min_turn_radius"] == pytest.approx(11.35)


def test_apply_pattern_limits_maps_degrees_onto_coefficient_bounds():
    sim = {"opti_limits_override": {"speed_radial": [-10.0, 5.0]}}
    ReeloutSession._apply_pattern_limits(sim, None)
    assert sim == {"opti_limits_override": {"speed_radial": [-10.0, 5.0]}}  # omitted -> untouched
    ReeloutSession._apply_pattern_limits(
        sim, {"azimuth_max": 35.0, "elevation_max": 45.0, "azimuth_amplitude_min": 5.0}
    )
    override = sim["opti_limits_override"]
    assert override["speed_radial"] == [-10.0, 5.0]  # winch entry preserved
    assert override["C_phi"] == pytest.approx([-np.radians(35.0), np.radians(35.0)])
    assert override["C_beta"][0] == pytest.approx(0.01)  # default lower side
    assert override["C_beta"][1] == pytest.approx(np.radians(45.0))
    assert sim["min_azimuth_amplitude"] == pytest.approx(np.radians(5.0))
    # {} clears the pattern limits but not the winch entry
    ReeloutSession._apply_pattern_limits(sim, {})
    assert sim == {"opti_limits_override": {"speed_radial": [-10.0, 5.0]}}
    with pytest.raises(ValueError, match="elevation_max"):
        ReeloutSession._apply_pattern_limits(sim, {"elevation_min": 40.0, "elevation_max": 30.0})


def test_pattern_limits_round_trip_init_step_and_replies(patched_session):
    sess, config = patched_session
    sess.init(dict(config, pattern_limits={"azimuth_max": 35.0, "elevation_max": 45.0}))
    sim = sess.phase.pattern_config["sim_parameters"]
    assert sim["opti_limits_override"]["C_phi"] == pytest.approx([-np.radians(35.0), np.radians(35.0)])
    echoed = sess.init_reply()["pattern_limits"]
    assert echoed["azimuth_max"] == pytest.approx(35.0)
    assert echoed["elevation_max"] == pytest.approx(45.0)
    assert echoed["elevation_min"] == pytest.approx(np.degrees(0.01))
    assert "azimuth_amplitude_min" not in echoed

    phase = sess.phase
    phase.results.append(_fake_result())
    phase.release.set()
    reply = sess.step_blocking()  # omitted -> kept
    assert reply["pattern_limits"]["azimuth_max"] == pytest.approx(35.0)

    phase.results.append(_fake_result())
    phase.release.set()
    reply = sess.step_blocking(pattern_limits={"azimuth_amplitude_min": 5.0})  # replaced as a whole
    assert reply["pattern_limits"] == {"azimuth_amplitude_min": pytest.approx(5.0)}
    assert "C_phi" not in sess.phase.pattern_config["sim_parameters"].get("opti_limits_override", {})

    phase.results.append(_fake_result())
    phase.release.set()
    reply = sess.step_blocking(pattern_limits={})  # cleared
    assert reply["pattern_limits"] is None
    assert "min_azimuth_amplitude" not in sess.phase.pattern_config["sim_parameters"]


def test_winch_mode_defaults_to_unset_and_is_threaded_through(patched_session):
    sess, config = patched_session
    base = {"mode": "reelout", "k_v": 0.02, "f_min": 1000.0, "f_max": 8400.0}

    config["winch_params"] = dict(base)
    sess.init(config)
    # Unset leaves the cycle config's own value, so force_law stays the default.
    assert "winch_mode" not in sess.phase.pattern_config["sim_parameters"]

    config["winch_params"] = dict(base, winch_mode="free_speed")
    sess.init(config)
    assert sess.phase.pattern_config["sim_parameters"]["winch_mode"] == "free_speed"

    config["winch_params"] = dict(base, winch_mode="nonsense")
    with pytest.raises(ValueError, match="winch_mode"):
        sess.init(config)

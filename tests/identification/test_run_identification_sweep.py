"""The identification sweep script: a failed trim ends a chain, not the sweep."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from awetrim.aerostructural.aerodynamic_vsm import CasadiTrimFailed

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "3_rom_identification" / "run_identification_sweep.py"


@pytest.fixture
def sweep(monkeypatch):
    spec = importlib.util.spec_from_file_location("identification_sweep_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_at", lambda config, **_: config)
    monkeypatch.setattr(module, "save_sim_output", lambda _tracking, _meta, folder: (folder / "sim_output.h5").touch())
    return module


class Prepared:
    power_tape_length = 1.0
    steering_half_difference = 0.0

    def __init__(self):
        self.restored = []

    def restore(self, path):
        self.restored.append(path)


class ScriptedDriver:
    """Fails the solves whose number (from 0) is in ``failing``."""

    def __init__(self, failing=()):
        self.failing, self.solves = set(failing), 0

    def solve_step(self, _case, _prepared, _config, final_state):
        self.solves += 1
        if self.solves - 1 in self.failing:
            raise CasadiTrimFailed("scripted")
        return {}, {"converged": True, "n_iter": 3, "opt_x": np.zeros(5)}


class Case:
    config = {}


SPEC = {
    "settle_on_down_branch": False,
    "depower": {"step": 0.05}, "steering": {"step": 0.025},
    "polar_d_alpha_deg": [-1.0, 1.0, 1.0], "max_stalled_fraction": 1.0,
    "wind_speeds": [4.0, 6.0], "start_from_converged": {"max_passes": 3},
}


def _chain(sweep, n_steps=3):
    return sweep.Chain("w5.0_depower_2.00", 5.0, [sweep.Step(None, 0.0) for _ in range(n_steps)])


def _scripted_rows(monkeypatch, sweep):
    written = []
    trim = {"va_vel_world": [20.0, 0.0, 0.0], "aoa_deg": 8.0, "tether_force": 1000.0, "opt_x": np.zeros(5)}
    monkeypatch.setattr(sweep, "anchor_rows", lambda *_: ({"anchor": "a"}, [], trim))
    monkeypatch.setattr(sweep.ds, "append_rows", lambda _path, rows: written.extend(rows))
    return written


def test_a_trim_that_keeps_failing_ends_the_chain_and_records_the_step(sweep, monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "prepare", lambda *_: Prepared())
    written = _scripted_rows(monkeypatch, sweep)
    driver = ScriptedDriver(failing=[1])
    chain = _chain(sweep)

    n_written = sweep.solve_chain(driver, Case, chain, {**SPEC, "start_from_converged": None}, tmp_path)

    assert n_written == 1 and len(written) == 1
    assert driver.solves == 2  # the step after the failure is not tried
    assert (tmp_path / chain.name / "step_01" / sweep.TRIM_FAILED_FILE).read_text(encoding="utf-8") == "scripted\n"
    assert not (tmp_path / chain.name / "step_01" / "sim_output.h5").exists()
    assert (tmp_path / chain.name / sweep.DONE_MARKER).exists()


def test_every_state_is_moved_to_the_down_branch_by_default(sweep, monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "prepare", lambda *_: Prepared())
    written = _scripted_rows(monkeypatch, sweep)
    settled = []

    def settle(_driver, _case, _spec, prepared, tracking, meta, final_state, config, *_):
        settled.append(True)
        return prepared, tracking, meta, final_state, config

    monkeypatch.setattr(sweep, "settle_on_down_branch", settle)
    spec = {k: v for k, v in SPEC.items() if k != "settle_on_down_branch"}
    sweep.solve_chain(ScriptedDriver(), Case, _chain(sweep), {**spec, "start_from_converged": None}, tmp_path)
    assert len(settled) == 3 == len(written)


def test_the_down_branch_re_solve_starts_from_the_down_march(sweep, monkeypatch):
    prepared = Prepared()
    prepared.vsm_solver = "solver"
    monkeypatch.setattr(sweep.aerodynamic_vsm, "down_branch_circulation", lambda *_, **__: np.full(3, 7.0))

    class Driver(ScriptedDriver):
        @staticmethod
        def _trim_seed_of(opt_x, gamma):
            return {"x": opt_x, "gamma": gamma}

        def solve_step(self, case, prep, config, final_state):
            result = super().solve_step(case, prep, config, final_state)
            final_state["results_aero"] = {"tether_force": 900.0, "aoa_deg": 9.0}
            return result

    first = {"results_aero": {"tether_force": 1000.0, "aoa_deg": 8.0, "opt_x": np.zeros(5)}, "body_aero": "body"}
    out = sweep.settle_on_down_branch(
        Driver(), Case, SPEC, prepared, "t", {"converged": True}, first, {}, [1.0], "saved", "a/00"
    )
    np.testing.assert_array_equal(prepared.trim_seed["gamma"], np.full(3, 7.0))
    assert out[3]["results_aero"]["tether_force"] == 900.0  # the re-solved state is the anchor


def test_a_failed_down_branch_re_solve_goes_back_to_the_first_solve(sweep, monkeypatch):
    fresh = Prepared()
    monkeypatch.setattr(sweep, "prepare", lambda *_: fresh)
    monkeypatch.setattr(sweep.aerodynamic_vsm, "down_branch_circulation", lambda *_, **__: np.zeros(3))

    class Driver(ScriptedDriver):
        _trim_seed_of = staticmethod(lambda opt_x, gamma: {})

    prepared = Prepared()
    prepared.vsm_solver = "solver"
    first = {"results_aero": {"tether_force": 1000.0, "aoa_deg": 8.0, "opt_x": np.zeros(5)}, "body_aero": "body"}
    out = sweep.settle_on_down_branch(
        Driver(failing=[0]), Case, SPEC, prepared, "t", {"converged": True}, first, {}, [1.0], "saved", "a/00"
    )
    assert out[0] is fresh and fresh.restored == ["saved"]
    assert out[3] is first


def test_the_lock_waits_when_windows_refuses_a_file_being_deleted(sweep, monkeypatch, tmp_path):
    real_open, calls = sweep.os.open, []

    def refuse_once(path, flags, *args):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, flags, *args)

    monkeypatch.setattr(sweep.os, "open", refuse_once)
    with sweep.sweep_files_lock(tmp_path):
        assert len(calls) == 2


def test_a_failed_base_state_goes_back_to_the_last_saved_shape(sweep, monkeypatch, tmp_path):
    prepared = []

    def prepare(*_):
        prepared.append(Prepared())
        return prepared[-1]

    monkeypatch.setattr(sweep, "prepare", prepare)
    driver = ScriptedDriver(failing=[1])

    # 4 m/s converges, 6 m/s fails; the walk turns (4) and comes back to 6 m/s, which converges.
    sweep.solve_base_states(driver, Case, SPEC, tmp_path)

    assert driver.solves == 4
    assert len(prepared) == 2
    assert prepared[1].restored == [sweep.base_state_path(tmp_path, 4.0)]
    assert (tmp_path / sweep.BASE_DIR / sweep.DONE_MARKER).exists()


def test_base_states_that_keep_failing_stop_the_sweep_by_name(sweep, monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "prepare", lambda *_: Prepared())
    with pytest.raises(SystemExit, match=r"6\.0 m/s .*unconverged"):
        sweep.solve_base_states(ScriptedDriver(failing=[1, 3]), Case, SPEC, tmp_path)
    assert not (tmp_path / sweep.BASE_DIR / sweep.DONE_MARKER).exists()


def test_steering_chains_can_run_past_the_stall_stop_of_the_depower_chains(sweep):
    spec = {"depower": {"targets": [1.8], "step": 0.1},
            "steering": {"power_tape_lengths": [1.5], "max": 0.1, "step": 0.1,
                         "max_stalled_fraction": 1.0},
            "wind_speeds": [5.0], "max_stalled_fraction": 0.2}
    depower, steering = sweep.build_chains(spec, 1.9)
    assert depower.max_stalled_fraction == 0.2
    assert steering.max_stalled_fraction == 1.0

    del spec["steering"]["max_stalled_fraction"]
    assert sweep.build_chains(spec, 1.9)[1].max_stalled_fraction == 0.2


def test_turn_law_jobs_pair_every_anchor_and_add_gravity_states(sweep):
    jobs = sweep.turn_law_jobs({
        "inertia_anchors": ["a/03", "g/07"],
        "gravity_anchors": ["g/07"],
        "gravity_states": [[30, 0, 0], [60, 0, 90]],
    })
    calm = [(base, scale) for base, _, scale, gravity in jobs if not gravity]
    assert calm == [("a/03", 1.0), ("a/03", 0.0), ("g/07", 1.0), ("g/07", 0.0)]
    windy = [(base, state, scale) for base, state, scale, gravity in jobs if gravity]
    assert windy == [("g/07", (30.0, 0.0, 0.0), 1.0), ("g/07", (60.0, 0.0, 90.0), 1.0)]


def test_window_wind_keeps_the_component_along_the_tether(sweep):
    assert sweep.window_wind(5.0, 60.0, 0.0) * np.cos(np.radians(60.0)) == pytest.approx(5.0)

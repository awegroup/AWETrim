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


def test_a_failed_trim_ends_the_chain_and_records_the_step(sweep, monkeypatch, tmp_path):
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

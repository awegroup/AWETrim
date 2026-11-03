"""PreparedWireframe: a structure put back from a saved solve, and a solve
continuing from the one before it."""

import numpy as np
import pytest

from awetrim.aerostructural.results import save_sim_output
from awetrim.aerostructural.wireframe import driver, structural_wireframe


@pytest.fixture(scope="module")
def case():
    from awetrim.aerostructural.case import Case

    return driver.build_case(
        Case(0.0, 0.0, 90.0, 300.0, 0.0, 0.0, 0.0, {"model": "uniform", "speed_wind_ref": 6.0}, 0.0, 0.0),
        kite_name="LEI-V3-KITE",
    )


def test_restore_puts_back_the_saved_structure(case, tmp_path):
    solved = driver.prepare_structure(case)
    reader = solved.reader
    first, second = reader["steering_tape_indices"][:2]
    psystem = solved.psystem
    psystem.update_rest_length(reader["power_tape_index"], 0.12)
    psystem.update_rest_length(first, -0.05)
    psystem.update_rest_length(second, 0.05)
    pulley = reader["pulley_line_indices"][0]
    psystem.update_rest_length(pulley, 0.01)
    positions = np.asarray(psystem.positions, dtype=float) + 0.01
    for particle, position in zip(psystem.particles, positions):
        particle.update_pos(position)
    stiffnesses = np.asarray(structural_wireframe.get_stiffnesses(psystem)) * 1.5
    structural_wireframe.set_stiffnesses(psystem, stiffnesses)
    rest_lengths = np.asarray(structural_wireframe.get_rest_lengths(psystem))
    # tracking positions[-1] is the shape after the final trim rotation; the
    # structure's own positions are saved apart and are what comes back
    h5 = save_sim_output(
        {"positions": np.stack([positions - 0.01, positions + 0.02])},
        {
            "n_iter": 2, "rest_lengths": rest_lengths, "final_stiffnesses": stiffnesses,
            "structure_positions": positions,
        },
        tmp_path,
    )

    fresh = driver.prepare_structure(case)
    fresh.restore(h5)

    np.testing.assert_allclose(fresh.psystem.positions, positions, atol=0)
    np.testing.assert_allclose(
        structural_wireframe.get_rest_lengths(fresh.psystem), rest_lengths, atol=1e-12
    )
    np.testing.assert_allclose(
        structural_wireframe.get_stiffnesses(fresh.psystem), stiffnesses, rtol=1e-12
    )
    # The built tapes stay the reference the steering is measured from.
    assert fresh.power_tape_length == pytest.approx(solved.power_tape_length)
    assert fresh.steering_half_difference == pytest.approx(0.05)
    assert fresh.steering_half_difference == pytest.approx(solved.steering_half_difference)


def test_restore_of_an_older_file_takes_the_last_tracked_positions(case, tmp_path):
    prepared = driver.prepare_structure(case)
    positions = np.asarray(prepared.psystem.positions, dtype=float) + 0.03
    rest_lengths = structural_wireframe.get_rest_lengths(prepared.psystem)
    h5 = save_sim_output(
        {"positions": positions[None]}, {"n_iter": 1, "rest_lengths": rest_lengths}, tmp_path
    )
    prepared.restore(h5)
    np.testing.assert_allclose(prepared.psystem.positions, positions, atol=0)


def test_restore_refuses_another_structure(case, tmp_path):
    prepared = driver.prepare_structure(case)
    h5 = save_sim_output(
        {"positions": np.zeros((1, 3, 3))}, {"n_iter": 1, "rest_lengths": np.ones(2)}, tmp_path
    )
    with pytest.raises(ValueError):
        prepared.restore(h5)


def _fake_solves(monkeypatch, n_panels: int = 3):
    """solve_step down to its call of the coupled solver, each call converging
    at a kite speed of 10, 11, ... m/s; returns the calls' system models and configs."""
    calls = []

    def main(**kwargs):
        calls.append((kwargs["system_model"], dict(kwargs["config"])))
        opt_x = np.array([9.0 + len(calls), 0.0, 0.0, 0.0, 0.0])
        return {}, {"converged": True, "opt_x": opt_x, "gamma_distribution": np.full(n_panels, 1.0 * len(calls))}

    monkeypatch.setattr(driver.coupled_solver_qsm, "main", main)
    return calls


def test_a_second_solve_continues_from_the_first(case, monkeypatch):
    calls = _fake_solves(monkeypatch)
    prepared = driver.prepare_structure(case)
    driver.solve_step(case, prepared)
    driver.solve_step(case, prepared)
    (model_0, config_0), (model_1, config_1) = calls
    assert "quasi_steady_initial_guess" not in config_0  # nothing to continue from yet
    assert model_1 is model_0  # the CasADi trim's graph is tied to the object
    assert config_1["quasi_steady_initial_guess"][0] == 10.0
    np.testing.assert_array_equal(config_1["initial_gamma_distribution"], np.full(3, 1.0))


def test_a_stated_seed_wins_and_the_case_config_stays_clean(case, monkeypatch):
    calls = _fake_solves(monkeypatch)
    prepared = driver.prepare_structure(case)
    driver.solve_step(case, prepared)
    stated = np.array([25.0, 0.0, 0.0, 0.0, 0.0])
    driver.solve_step(case, prepared, {**case.config, "quasi_steady_initial_guess": stated})
    assert calls[1][1]["quasi_steady_initial_guess"] is stated
    assert "quasi_steady_initial_guess" not in case.config


def test_another_flight_state_starts_cold(case, monkeypatch):
    calls = _fake_solves(monkeypatch)
    prepared = driver.prepare_structure(case)
    driver.solve_step(case, prepared)
    driver.solve_step(case, prepared, {**case.config, "distance_radial": 450.0})
    assert "quasi_steady_initial_guess" not in calls[1][1]


def test_an_unconverged_solve_is_not_continued_from(case, monkeypatch):
    prepared = driver.prepare_structure(case)
    prepared.trim_seed = {"quasi_steady_initial_guess": np.array([12.0, 0.0, 0.0, 0.0, 0.0])}
    monkeypatch.setattr(
        driver.coupled_solver_qsm, "main",
        lambda **_: ({}, {"converged": False, "opt_x": np.full(5, 99.0)}),
    )
    driver.solve_step(case, prepared)
    assert prepared.trim_seed["quasi_steady_initial_guess"][0] == 12.0


@pytest.mark.parametrize("gamma", [np.array([1.0, 2.0, 3.0]), None])
def test_restore_brings_the_seed_back(case, tmp_path, gamma):
    prepared = driver.prepare_structure(case)
    meta = {
        "n_iter": 1, "opt_x": np.array([14.0, 0.0, 0.0, 0.0, 0.0]),
        "rest_lengths": np.asarray(structural_wireframe.get_rest_lengths(prepared.psystem)),
        "structure_positions": np.asarray(prepared.psystem.positions, dtype=float),
    }
    if gamma is not None:
        meta["gamma_distribution"] = gamma
    fresh = driver.prepare_structure(case)
    fresh.restore(save_sim_output({"positions": meta["structure_positions"][None]}, meta, tmp_path))
    assert fresh.trim_seed["quasi_steady_initial_guess"][0] == 14.0
    assert ("initial_gamma_distribution" in fresh.trim_seed) == (gamma is not None)


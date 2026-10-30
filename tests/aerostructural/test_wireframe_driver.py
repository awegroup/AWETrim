"""PreparedWireframe.restore: a structure put back from a saved solve."""

import numpy as np
import pytest

from awetrim.aerostructural.results import save_sim_output
from awetrim.aerostructural.wireframe import driver, structural_wireframe


@pytest.fixture(scope="module")
def case():
    return driver.build_case(kite_name="LEI-V3-KITE")


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

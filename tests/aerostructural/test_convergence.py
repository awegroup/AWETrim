import numpy as np
import pytest

from awetrim.aerostructural.convergence import (
    check_convergence,
    compute_adaptive_dt,
    element_elongations,
    max_element_elongation,
    relative_residual_norm,
    resolve_residual_tolerances,
    resultant_tether_force,
)
from awetrim.aerostructural.forces import distribute_total_force_by_particle_mass


def solver_config(**overrides):
    config = {
        "residual_tol_relative": 1.0e-4,
        "n_max_constant_residual_force": 3,
        "stagnation_tol_relative": 4.0e-5,
        "max_iter": 10,
    }
    config.update(overrides)
    return config


def test_compute_adaptive_dt_increases_near_residual_tolerance():
    dt = compute_adaptive_dt(
        residual_norm_history=[2.0e-4],
        dt_initial=0.005,
        dt_max=0.010,
        residual_tol=1.0e-4,
    )

    assert dt == 0.0075


def test_resultant_tether_force_is_the_resultant_external_load():
    # Internal spring forces cancel; the tether carries the resultant of the
    # applied nodal loads, whatever their distribution over the nodes.
    external_force = np.asarray([[0.0, 0.0, 1000.0], [30.0, 0.0, 2000.0]])

    assert resultant_tether_force(external_force) == pytest.approx(
        np.hypot(30.0, 3000.0)
    )
    # Flattened input is accepted and gives the same answer.
    assert resultant_tether_force(external_force.reshape(-1)) == pytest.approx(
        resultant_tether_force(external_force)
    )


def test_relative_residual_norm_is_undefined_without_a_normalising_force():
    assert np.isnan(relative_residual_norm(np.asarray([1.0, 0.0, 0.0]), 0.0))
    assert np.isnan(relative_residual_norm(np.asarray([1.0, 0.0, 0.0]), np.nan))


def test_check_convergence_uses_the_residual_normalised_by_tether_force():
    residual = np.asarray([0.0, 0.0, 0.2])  # 0.2 N on a 4 kN tether force = 5e-5

    converged, should_break, is_stagnated = check_convergence(
        iteration=0,
        residual=residual,
        residual_norm_history=[5.0e-5],
        force_reference=4000.0,
        aero_forces_vsm_format=np.zeros((1, 3)),
        solver_config=solver_config(),
        is_run_only_1_time_step=False,
    )

    assert converged is True
    assert should_break is False
    assert is_stagnated is False


def test_check_convergence_rejects_the_same_residual_at_a_lower_tether_force():
    # 0.2 N is converged at 4 kN but not at 400 N: the criterion is relative.
    converged, _, _ = check_convergence(
        iteration=0,
        residual=np.asarray([0.0, 0.0, 0.2]),
        residual_norm_history=[5.0e-4],
        force_reference=400.0,
        aero_forces_vsm_format=np.zeros((1, 3)),
        solver_config=solver_config(),
        is_run_only_1_time_step=False,
    )

    assert converged is False


def test_check_convergence_does_not_converge_on_a_broken_normalisation():
    converged, should_break, _ = check_convergence(
        iteration=0,
        residual=np.asarray([0.0, 0.0, 0.2]),
        residual_norm_history=[np.nan],
        force_reference=0.0,
        aero_forces_vsm_format=np.zeros((1, 3)),
        solver_config=solver_config(),
        is_run_only_1_time_step=False,
    )

    assert converged is False
    assert should_break is False


def test_check_convergence_breaks_on_nan_residual():
    _, should_break, _ = check_convergence(
        iteration=0,
        residual=np.asarray([np.nan, 0.0, 0.0]),
        residual_norm_history=[np.nan],
        force_reference=4000.0,
        aero_forces_vsm_format=np.zeros((1, 3)),
        solver_config=solver_config(),
        is_run_only_1_time_step=False,
    )

    assert should_break is True


def test_legacy_absolute_force_tolerances_are_rejected():
    # `tol: 5` [N] read as a relative tolerance would "converge" immediately.
    with pytest.raises(ValueError, match="residual_tol_relative"):
        resolve_residual_tolerances({"tol": 5.0})
    with pytest.raises(ValueError, match="stagnation_tol_relative"):
        resolve_residual_tolerances({"stagnation_tol": 2.0})


def test_resolve_residual_tolerances_falls_back_to_defaults():
    tol, stagnation_tol = resolve_residual_tolerances({})

    assert tol == 1.0e-4
    assert stagnation_tol == 4.0e-5


def test_max_element_elongation_reports_the_most_stretched_element():
    nodes = np.asarray([[0.0, 0.0, 0.0], [1.02, 0.0, 0.0], [1.02, 0.5, 0.0]])
    connectivity = [[0, 1, "wing"], [1, 2, "wing"]]
    rest_lengths = np.asarray([1.0, 0.5])

    assert max_element_elongation(nodes, connectivity, rest_lengths) == pytest.approx(
        0.02
    )


def test_distribute_total_force_by_particle_mass_preserves_total_force():
    nodal_forces = distribute_total_force_by_particle_mass(
        total_force=np.asarray([3.0, 6.0, 9.0]),
        m_arr=np.asarray([1.0, 2.0]),
    )

    np.testing.assert_allclose(np.sum(nodal_forces, axis=0), [3.0, 6.0, 9.0])
    np.testing.assert_allclose(nodal_forces[0], [1.0, 2.0, 3.0])


def test_element_elongations_share_the_rope_strain_across_a_pulley():
    # Arm 0 lengthens by 0.02 m while its partner (element 2) shortens by the
    # same amount: the rope is unstrained, and both arms must report ~0.
    nodes = np.asarray([[0.0, 0.0, 0.0], [1.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.98]])
    connectivity = [[0, 1, "arm_a"], [0, 1, "spacer"], [2, 3, "arm_b"]]
    rest_lengths = np.asarray([1.0, 1.0, 2.0])
    pulley_pairs = {"0": [2, 3, 2.0, 1.0, 0], "2": [0, 1, 1.0, 2.0, 0]}

    elongations = element_elongations(
        nodes, connectivity, rest_lengths, pulley_pairs=pulley_pairs
    )

    assert elongations[0] == pytest.approx(0.0, abs=1e-12)
    assert elongations[2] == pytest.approx(0.0, abs=1e-12)
    # The non-pulley element keeps its own strain.
    assert elongations[1] == pytest.approx(0.02)

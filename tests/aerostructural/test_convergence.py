import numpy as np

from awetrim.aerostructural.convergence import check_convergence, compute_adaptive_dt
from awetrim.aerostructural.forces import distribute_total_force_by_particle_mass


def test_compute_adaptive_dt_increases_near_residual_tolerance():
    dt = compute_adaptive_dt(
        residual_norm_history=[10.0],
        dt_initial=0.005,
        dt_max=0.010,
        residual_tol=5.0,
    )

    assert dt == 0.0075


def test_check_convergence_detects_residual_below_tolerance():
    converged, should_break, is_stagnated = check_convergence(
        iteration=0,
        residual=np.asarray([0.0, 0.0, 1.0]),
        residual_norm_history=[1.0],
        aero_forces_vsm_format=np.zeros((1, 3)),
        solver_config={
            "tol": 2.0,
            "n_max_constant_residual_force": 3,
            "stagnation_tol": 0.1,
            "max_iter": 10,
        },
        is_run_only_1_time_step=False,
    )

    assert converged is True
    assert should_break is False
    assert is_stagnated is False


def test_distribute_total_force_by_particle_mass_preserves_total_force():
    nodal_forces = distribute_total_force_by_particle_mass(
        total_force=np.asarray([3.0, 6.0, 9.0]),
        m_arr=np.asarray([1.0, 2.0]),
    )

    np.testing.assert_allclose(np.sum(nodal_forces, axis=0), [3.0, 6.0, 9.0])
    np.testing.assert_allclose(nodal_forces[0], [1.0, 2.0, 3.0])

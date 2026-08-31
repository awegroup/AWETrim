import numpy as np
import pytest

from awetrim.aerostructural import convergence
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


# --------------------------------------------------- tether-normalised criterion
def _solver_config(**overrides):
    config = {
        "tol": 5.0,
        "stagnation_tol": 2.0,
        "n_max_constant_residual_force": 10,
        "max_iter": 100,
    }
    config.update(overrides)
    return config


def test_resultant_tether_force_is_the_resultant_of_the_applied_load():
    forces = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    assert convergence.resultant_tether_force(forces) == pytest.approx(5.0)
    # Flattened input is the same load, so it must give the same reference.
    assert convergence.resultant_tether_force(forces.ravel()) == pytest.approx(5.0)


def test_relative_residual_norm_is_undefined_without_a_normalising_force():
    residual = np.array([3.0, 4.0, 0.0])
    assert convergence.relative_residual_norm(residual, 2.0) == pytest.approx(2.5)
    assert np.isnan(convergence.relative_residual_norm(residual, 0.0))
    assert np.isnan(convergence.relative_residual_norm(residual, np.nan))


def test_resolve_tolerances_keeps_the_absolute_gate_by_default():
    tol, stagnation, is_relative = convergence.resolve_residual_tolerances(
        _solver_config()
    )
    assert (tol, stagnation, is_relative) == (5.0, 2.0, False)


def test_resolve_tolerances_switches_to_relative_when_configured():
    tol, stagnation, is_relative = convergence.resolve_residual_tolerances(
        _solver_config(residual_tol_relative=1e-4)
    )
    assert is_relative
    assert tol == pytest.approx(1e-4)
    assert stagnation == pytest.approx(convergence.DEFAULT_STAGNATION_TOL_RELATIVE)


def test_a_force_in_the_relative_key_is_rejected_not_reinterpreted():
    # `tol: 5` read as a ratio would "converge" on iteration 1, so the mistake
    # has to be loud rather than silently accepted.
    with pytest.raises(ValueError, match="RATIO"):
        convergence.resolve_residual_tolerances(
            _solver_config(residual_tol_relative=5.0)
        )
    with pytest.raises(ValueError, match="positive ratio"):
        convergence.resolve_residual_tolerances(
            _solver_config(residual_tol_relative=0.0)
        )


def test_relative_criterion_judges_the_same_residual_by_the_tether_load():
    # 1 N of residual is converged at 8 kN of tether load and not at 2 kN --
    # exactly the distinction the absolute gate cannot make.
    residual = np.array([1.0, 0.0, 0.0])
    aero = np.zeros((3, 3))
    config = _solver_config(residual_tol_relative=2.5e-4)

    def run(force_reference):
        return convergence.check_convergence(
            iteration=0,
            residual=residual,
            residual_norm_history=[],
            aero_forces_vsm_format=aero,
            solver_config=config,
            is_run_only_1_time_step=False,
            force_reference=force_reference,
        )

    assert run(8000.0)[0] is True
    assert run(2000.0)[0] is False
    # The absolute gate calls both of them converged at tol = 5 N.
    absolute = convergence.check_convergence(
        iteration=0,
        residual=residual,
        residual_norm_history=[],
        aero_forces_vsm_format=aero,
        solver_config=_solver_config(),
        is_run_only_1_time_step=False,
    )
    assert absolute[0] is True


def test_relative_criterion_requires_the_normalising_force():
    with pytest.raises(ValueError, match="force_reference"):
        convergence.check_convergence(
            iteration=0,
            residual=np.array([1.0, 0.0, 0.0]),
            residual_norm_history=[],
            aero_forces_vsm_format=np.zeros((3, 3)),
            solver_config=_solver_config(residual_tol_relative=1e-4),
            is_run_only_1_time_step=False,
        )


def test_undefined_normalisation_neither_converges_nor_kills_the_run():
    is_converged, should_break, _ = convergence.check_convergence(
        iteration=0,
        residual=np.array([1.0, 0.0, 0.0]),
        residual_norm_history=[],
        aero_forces_vsm_format=np.zeros((3, 3)),
        solver_config=_solver_config(residual_tol_relative=1e-4),
        is_run_only_1_time_step=False,
        force_reference=0.0,
    )
    assert is_converged is False
    assert should_break is False


# ------------------------------------------------- plateau fallback tolerance
def test_fallback_defaults_to_ten_times_the_main_tolerance():
    tol, _, _ = convergence.resolve_residual_tolerances(
        _solver_config(residual_tol_relative=1e-4)
    )
    fallback = convergence.resolve_fallback_tolerance(
        _solver_config(residual_tol_relative=1e-4), tol
    )
    assert fallback == pytest.approx(1e-3)


def test_fallback_can_be_named_explicitly():
    config = _solver_config(
        residual_tol_relative=1e-4, residual_tol_relative_fallback=5e-4
    )
    assert convergence.resolve_fallback_tolerance(config, 1e-4) == pytest.approx(5e-4)


def test_fallback_is_disabled_by_a_factor_of_one_or_less():
    # Returning the main tolerance means "no looser tier"; the driver compares
    # fallback > residual_tol before it will accept a plateaued solve.
    config = _solver_config(
        residual_tol_relative=1e-4, residual_tol_fallback_factor=1.0
    )
    assert convergence.resolve_fallback_tolerance(config, 1e-4) == pytest.approx(1e-4)


def test_fallback_applies_to_the_absolute_criterion_too():
    # The tier is expressed in whatever measure is active, so a legacy
    # newton-gated solve gets a newton fallback.
    assert convergence.resolve_fallback_tolerance(_solver_config(), 5.0) == (
        pytest.approx(50.0)
    )

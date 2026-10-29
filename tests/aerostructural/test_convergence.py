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


def test_set_absolute_residual_gate_makes_newtons_the_live_measure():
    # Setting `tol` alone is overridden once residual_tol_relative is present;
    # the helper must remove the relative keys, fallback included (a ratio
    # fallback read as newtons would accept anything).
    config = _solver_config(
        residual_tol_relative=1e-4,
        stagnation_tol_relative=4e-5,
        residual_tol_relative_fallback=2e-3,
    )
    assert convergence.set_absolute_residual_gate(config, 0.5, 0.05) is config
    assert "residual_tol_relative" not in config
    assert "stagnation_tol_relative" not in config
    assert "residual_tol_relative_fallback" not in config
    assert convergence.resolve_residual_tolerances(config) == (0.5, 0.05, False)


def test_the_lei_v3_as_config_ships_the_relative_gate_only():
    # One criterion in the shipped file: a dead `tol` next to the live
    # relative key is how the Billow driver's silent absolute gate went unseen.
    from awetrim.aerostructural.utils import load_yaml
    from awetrim.utils.config_paths import REPO_ROOT

    solver = load_yaml(REPO_ROOT / "data" / "LEI-V3-KITE" / "as_config.yaml")[
        "aero_structural_solver"
    ]
    assert "tol" not in solver
    assert "stagnation_tol" not in solver
    tol, stagnation, is_relative = convergence.resolve_residual_tolerances(solver)
    assert is_relative is True
    assert tol == pytest.approx(1e-4)
    assert stagnation == pytest.approx(4e-5)


# ------------------------------------------------------- tether-force settle
# |sum f_ext| per iteration on the LEI-V3 Billow baseline (run_simulation_BILLOW
# inputs, 2026-10-12). The residual gate 1e-4 passed at index 5; index 1 read
# 1.5e-4 with the force 3.8% short.
_BILLOW_BASELINE_FORCE = [4296.3, 4300.0, 4324.8, 4459.5, 4469.7, 4470.4, 4469.9, 4470.0]


def test_force_settle_is_off_without_a_tolerance():
    assert convergence.force_settle_state([1.0], 0.0)[0] is True
    assert convergence.force_settle_state([1.0], None or 0.0)[0] is True


def test_force_settle_needs_a_full_window_of_history():
    flat = [1000.0] * (convergence.FORCE_SETTLE_WINDOW - 1)
    assert convergence.force_settle_state(flat, 1e-4) == (False, float("inf"))
    assert convergence.force_settle_state(flat + [1000.0], 1e-4)[0] is True


def test_force_settle_holds_the_billow_baseline_until_the_force_stops():
    settled = [
        convergence.force_settle_state(_BILLOW_BASELINE_FORCE[: k + 1], 1e-4)[0]
        for k in range(len(_BILLOW_BASELINE_FORCE))
    ]
    assert settled == [False] * 7 + [True]


def test_force_settle_accepts_a_negligible_limit_cycle_only():
    # A cycle has no geometric tail (remaining_drift is infinite); its spread
    # decides. 0.1 N on 4 kN passes 1e-4, 10 N does not.
    small = [4000.0, 4000.1, 4000.0, 4000.1, 4000.0, 4000.1]
    large = [4000.0, 4010.0, 4000.0, 4010.0, 4000.0, 4010.0]
    assert convergence.force_settle_state(small, 1e-4)[0] is True
    assert convergence.force_settle_state(large, 1e-4)[0] is False


def test_the_lei_v3_as_config_ships_the_force_settle_rule():
    from awetrim.aerostructural.utils import load_yaml
    from awetrim.utils.config_paths import REPO_ROOT

    solver = load_yaml(REPO_ROOT / "data" / "LEI-V3-KITE" / "as_config.yaml")[
        "aero_structural_solver"
    ]
    assert solver["force_settle_rtol"] == pytest.approx(1e-4)

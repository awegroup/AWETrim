"""Unit tests for :mod:`awetrim.timeseries.phase` result helpers.

Covers the optimizer-output capture added to ``Phase`` /
``SimulationResult``: reading the per-node trajectory straight from the NLP
solution (``_extract_optimized_trajectory``) and dumping it to CSV
(``SimulationResult.save_trajectory_csv``). These use a tiny fake solution
object so they run fast and deterministically -- no IPOPT, no kite data.
"""

import numpy as np
import pytest

from awetrim.timeseries.phase import Phase, SimulationResult


class _FakeSolution:
    """Minimal CasADi-solution stand-in: ``value(x)`` returns ``x`` verbatim."""

    def value(self, x):
        return x


def _phase(n_points):
    return Phase(
        system_model=None,
        pattern_config={"sim_parameters": {"n_points": n_points}},
    )


def _result(trajectory=None):
    kwargs = dict(
        solution=None,
        optimized_config={},
        final_distance=0.0,
        phase_variables={},
        energy_objective=0.0,
        total_time=0.0,
    )
    if trajectory is not None:
        kwargs["optimized_trajectory"] = trajectory
    return SimulationResult(**kwargs)


def test_extract_optimized_trajectory_trims_s_grid_to_n_points():
    n = 5
    phase = _phase(n)
    opti_vars = {
        "s": np.linspace(0.0, 1.0, n + 1),  # s-grid carries n+1 entries
        "s_dot": np.full(n, 2.0),
        "input_steering": np.linspace(-0.1, 0.1, n),
        "speed_radial": np.full(n, 1.0),
        "distance_radial": np.linspace(200.0, 250.0, n),
        "tension_tether_ground": np.full(n, 5.0e4),
        "input_depower": np.linspace(1.5, 1.7, n),
    }
    traj = phase._extract_optimized_trajectory(_FakeSolution(), opti_vars)

    assert set(traj) == set(opti_vars)
    for values in traj.values():
        assert len(values) == n  # s trimmed from n+1 -> n
        assert np.all(np.isfinite(values))
    assert np.allclose(traj["s"], np.linspace(0.0, 1.0, n + 1)[:n])


def test_extract_optimized_trajectory_skips_missing_and_unevaluable():
    n = 4
    phase = _phase(n)

    class _PartialSolution:
        def value(self, x):
            if isinstance(x, str):  # simulate a var that cannot be evaluated
                raise RuntimeError("cannot evaluate")
            return x

    opti_vars = {
        "s_dot": np.full(n, 3.0),
        "input_steering": "not-evaluable",
        # input_depower deliberately absent -> skipped
    }
    traj = phase._extract_optimized_trajectory(_PartialSolution(), opti_vars)

    assert "s_dot" in traj and len(traj["s_dot"]) == n
    assert "input_steering" not in traj
    assert "input_depower" not in traj


def test_extract_optimized_trajectory_without_n_points_keeps_full_length():
    # No n_points -> no trimming; arrays come back at their native length.
    phase = Phase(system_model=None, pattern_config={})
    opti_vars = {"s_dot": np.arange(7.0)}
    traj = phase._extract_optimized_trajectory(_FakeSolution(), opti_vars)
    assert len(traj["s_dot"]) == 7


def test_save_trajectory_csv_roundtrip(tmp_path):
    traj = {
        "s": np.array([0.0, 0.5, 1.0]),
        "input_steering": np.array([-0.1, 0.0, 0.1]),
        "input_depower": np.array([1.5, 1.6, 1.7]),
    }
    result = _result(traj)

    out = tmp_path / "traj.csv"
    result.save_trajectory_csv(out)

    assert out.exists()
    assert out.read_text().splitlines()[0] == "s,input_steering,input_depower"
    data = np.loadtxt(out, delimiter=",", skiprows=1)
    assert data.shape == (3, 3)
    assert np.allclose(data[:, 0], traj["s"])
    assert np.allclose(data[:, 2], traj["input_depower"])


def test_save_trajectory_csv_empty_is_noop(tmp_path):
    result = _result()  # optimized_trajectory defaults to {}
    out = tmp_path / "traj.csv"
    result.save_trajectory_csv(out)
    assert not out.exists()


def test_trajectory_from_phase_trims_to_grid_and_rejects_truncated():
    """``Phase._trajectory_from_phase`` turns a forward simulation into the
    length-n_points seed dict opti_phase consumes, and returns {} (no seed)
    when the march stopped short of the grid."""
    import numpy as np

    from awetrim.timeseries.phase import Phase

    class _FakeTS:
        def __init__(self, n):
            self.n = n

        def return_variable(self, name):
            if name == "angle_course":
                raise KeyError(name)  # not every model exposes every column
            return np.arange(self.n, dtype=float) + hash(name) % 7

    ph = Phase.__new__(Phase)  # no system model needed for this helper
    ph.pattern_config = {"sim_parameters": {"n_points": 10}}

    traj = ph._trajectory_from_phase(_FakeTS(11))  # march records N + 1
    assert set(traj) == set(Phase._WARM_START_SEED_KEYS) - {"angle_course"}
    assert all(v.shape == (10,) for v in traj.values())
    assert traj["s"][0] == pytest.approx(hash("s") % 7)

    assert ph._trajectory_from_phase(_FakeTS(7)) == {}  # truncated march



def _mocked_profile_solve(monkeypatch, sim_parameters):
    """Run ``run_simulation_opti`` with the NLP build and solve stubbed out,
    returning the depower profile it persists for the given sim keys."""
    n = 4
    dep_opt = np.array([1.7, 1.9, 2.1, 1.8])
    opti_vars = {
        "s": np.linspace(0.0, 1.0, n + 1),
        "s_dot": np.full(n, 2.0),
        "input_steering": np.zeros(n),
        "speed_radial": np.ones(n),
        "distance_radial": np.full(n, 200.0),
        "tension_tether_ground": np.full(n, 5.0e3),
        "input_depower": dep_opt,
    }
    objective = {"energy": 1.0, "total_time": 1.0, "power_scale": 1.0}
    phase = Phase(
        system_model=None,
        pattern_config={"sim_parameters": {"n_points": n, **sim_parameters}},
    )
    monkeypatch.setattr(
        Phase,
        "get_opti_components",
        lambda self, **kw: (None, opti_vars, objective, {}),
    )
    monkeypatch.setattr(Phase, "run_opti", lambda self, *a, **kw: _FakeSolution())
    result = phase.run_simulation_opti(optimization_params=[])
    assert result is not None
    return dep_opt, np.asarray(phase.pattern_config["sim_parameters"]["input_depower_profile"])


def test_persisted_depower_profile_pads_with_last_node_by_default(monkeypatch):
    dep_opt, profile = _mocked_profile_solve(monkeypatch, {})
    assert profile.size == dep_opt.size + 1
    assert np.allclose(profile[:-1], dep_opt)
    assert profile[-1] == dep_opt[-1]


def test_persisted_depower_profile_wraps_to_node_zero_under_periodic_wrap(monkeypatch):
    dep_opt, profile = _mocked_profile_solve(monkeypatch, {"periodic_wrap": True})
    assert profile.size == dep_opt.size + 1
    assert np.allclose(profile[:-1], dep_opt)
    assert profile[-1] == dep_opt[0]

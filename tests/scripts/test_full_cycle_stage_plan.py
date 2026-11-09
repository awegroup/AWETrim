"""Unit tests for the stage plan of run_full_cycle_opti.py (roll-gain continuation, box retries).

The script is loaded by file path (scripts are not an installed package). Only
the solver-free stage-plan helpers are exercised: the halving schedule back to
the kite's own roll gain and the midpoint retry of a failed step.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CYCLE = REPO / "examples" / "4_trajectory_optimization" / "cycle"


@pytest.fixture(scope="module")
def rfco():
    sys.path.insert(0, str(CYCLE))
    try:
        spec = importlib.util.spec_from_file_location(
            "scripts_run_full_cycle_opti", CYCLE / "run_full_cycle_opti.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"run_full_cycle_opti not importable: {exc}")
    finally:
        sys.path.remove(str(CYCLE))
    return module


STAGES = [{"label": "close"}, {"label": "power"}, {"label": "polish"}]


def test_no_assist_keeps_the_plan_at_scale_one(rfco):
    plan = rfco._roll_assist_stages(STAGES, None)
    assert [s["label"] for s in plan] == ["close", "power", "polish"]
    assert all(s["roll_scale"] == 1.0 for s in plan)


def test_assist_halves_back_to_one_before_the_power_stages(rfco):
    plan = rfco._roll_assist_stages(STAGES, 1.75)
    scales = [s["roll_scale"] for s in plan]
    assert scales == [1.75, 1.375, 1.1875, 1.09375, 1.0, 1.0, 1.0]
    assert [s["label"] for s in plan[-2:]] == ["power", "polish"]
    assert all(s.get("continuation") for s in plan[1:5])


def test_last_step_jumps_to_one_below_the_minimum_step(rfco):
    # 1.2 -> 1.1 -> 1.05; the next half step (0.025) is below the minimum
    plan = rfco._roll_assist_stages(STAGES, 1.2)
    assert [s["roll_scale"] for s in plan[:5]] == pytest.approx([1.2, 1.1, 1.05, 1.0, 1.0])
    assert plan[4]["label"] == "power"


def test_failed_step_retries_from_the_midpoint_then_gives_up(rfco):
    plan = rfco._roll_assist_stages(STAGES, 1.5)
    i = 1  # the 1.5 -> 1.25 step
    assert rfco._retry_continuation(plan, i)
    assert plan[i]["roll_scale"] == pytest.approx(1.375)
    assert plan[i + 1]["roll_scale"] == pytest.approx(1.25)
    for _ in range(rfco.roll_assist.ROLL_ASSIST_MAX_HALVINGS - 1):
        assert rfco._retry_continuation(plan, i)
    assert not rfco._retry_continuation(plan, i)


def test_a_power_stage_is_never_retried(rfco):
    plan = rfco._roll_assist_stages(STAGES, 1.5)
    assert not rfco._retry_continuation(plan, len(plan) - 1)


def test_failed_power_stage_retries_with_the_box_halved_then_gives_up(rfco):
    plan = [dict(s) for s in rfco.STAGES]
    i = 1  # shape+r0
    wide = dict(plan[i]["step_bounds"])
    assert rfco._retry_halved_box(plan, i)
    assert plan[i]["step_bounds"] == pytest.approx({k: 0.5 * v for k, v in wide.items()})
    for _ in range(rfco.MAX_BOX_HALVINGS - 1):
        assert rfco._retry_halved_box(plan, i)
    assert not rfco._retry_halved_box(plan, i)
    assert rfco.STAGES[i]["step_bounds"] == wide  # the module's plan is untouched


def test_the_closing_stage_keeps_its_box(rfco):
    plan = [dict(s) for s in rfco.STAGES]
    assert not rfco._retry_halved_box(plan, 0)


class _Cycle:
    def __init__(self, adopted):
        self.adopted_failed_iterate = adopted


def test_a_stalled_stage_resumes_from_its_kept_iterate_a_bounded_number_of_times(rfco):
    stage = dict(rfco.STAGES[1])
    assert not rfco._resume_stalled(stage, _Cycle(False))
    for _ in range(rfco.MAX_RESUMES):
        assert rfco._resume_stalled(stage, _Cycle(True))
    assert not rfco._resume_stalled(stage, _Cycle(True))


def test_box_recentring_stops_when_a_pass_gains_too_little(rfco):
    stage = dict(rfco.STAGES[1])
    assert rfco._recentre_box(stage, None, 9000.0)  # first pass: re-centre
    assert rfco._recentre_box(stage, 9000.0, 9200.0)  # +2.2 %: again
    assert not rfco._recentre_box(stage, 9200.0, 9201.0)  # +0.01 %: done


def test_box_recentring_is_bounded(rfco):
    stage = dict(rfco.STAGES[1])
    for _ in range(rfco.MAX_RECENTRES):
        assert rfco._recentre_box(stage, None, 1.0)
    assert not rfco._recentre_box(stage, None, 1.0)


def test_mean_power_of_a_constant_trajectory(rfco):
    traj = {"s": [0.0, 0.5, 1.0], "s_dot": [0.01, 0.01, 0.01],
            "tension_tether_ground": [1000.0] * 3, "speed_radial": [2.0] * 3}
    assert rfco._trajectory_mean_power(traj) == pytest.approx(2000.0)
    assert rfco._trajectory_mean_power({}) is None


def test_warm_start_is_resampled_onto_the_run_grid(rfco):
    traj = {"s": [0.0, 0.25, 0.5, 0.75], "x": [0.0, 1.0, 2.0, 3.0]}
    out = rfco._resample_trajectory(traj, {"n_points": 8, "start_angle": 0.0, "end_angle": 1.0})
    assert out["s"] == pytest.approx([i / 8 for i in range(8)])
    assert out["x"][:7] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


def test_warm_start_on_the_grid_is_unchanged(rfco):
    traj = {"s": [0.0, 0.5], "x": [1.0, 2.0]}
    assert rfco._resample_trajectory(traj, {"n_points": 2, "start_angle": 0.0, "end_angle": 1.0}) is traj


def test_halved_box_is_remembered_for_the_recentred_passes(rfco):
    plan = [dict(s) for s in rfco.STAGES]
    full = dict(plan[1]["step_bounds"])
    assert rfco._retry_halved_box(plan, 1)
    assert plan[1]["step_bounds_full"] == full


def test_closed_cycle_resamples_across_the_seam(rfco):
    traj = {"s": [0.0, 0.5], "x": [0.0, 2.0]}
    out = rfco._resample_trajectory(
        traj, {"n_points": 4, "start_angle": 0.0, "end_angle": 1.0, "periodic_wrap": True})
    assert out["x"] == pytest.approx([0.0, 1.0, 2.0, 1.0])  # 0.75 lies between 2 and node 0

"""Unit tests for the roll-gain continuation plan of run_full_cycle_opti.py.

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
    for _ in range(rfco.ROLL_ASSIST_MAX_HALVINGS - 1):
        assert rfco._retry_continuation(plan, i)
    assert not rfco._retry_continuation(plan, i)


def test_a_power_stage_is_never_retried(rfco):
    plan = rfco._roll_assist_stages(STAGES, 1.5)
    assert not rfco._retry_continuation(plan, len(plan) - 1)

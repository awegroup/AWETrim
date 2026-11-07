"""Roll-gain assist: march scale selection and the continuation schedule (no solver)."""

from types import SimpleNamespace

import numpy as np
import pytest

from awetrim.timeseries import roll_assist
from awetrim.timeseries.roll_assist import (
    ROLL_ASSIST_MAX_HALVINGS,
    continuation_scales,
    continue_roll_gain,
    march_with_roll_assist,
)

N = 10


def test_continuation_scales_halve_the_excess_to_one():
    assert continuation_scales(2.0) == [1.5, 1.25, 1.125, 1.0625, 1.0]
    assert continuation_scales(1.25) == [1.125, 1.0625, 1.0]
    assert continuation_scales(1.0) == []
    assert continuation_scales(None) == []
    assert all(
        b - 1.0 >= roll_assist.ROLL_ASSIST_MIN_STEP or b == 1.0
        for b in continuation_scales(1.75)
    )


def _recording_solve(kite, fail_at=()):
    """solve() that records the scale and fails (None) at listed scales once."""
    scales, failed = [], set()

    def solve():
        scale = kite.gain_roll_scale
        scales.append(scale)
        if scale in fail_at and scale not in failed:
            failed.add(scale)
            return None
        return f"x{scale:g}"

    return scales, solve


def test_continue_roll_gain_reaches_one_and_restores_the_kite():
    kite = SimpleNamespace(gain_roll_scale=1.5)
    scales, solve = _recording_solve(kite)
    assert continue_roll_gain(kite, 1.5, solve) == "x1"
    assert scales == [1.25, 1.125, 1.0625, 1.0]
    assert kite.gain_roll_scale == 1.0


def test_continue_roll_gain_retries_a_failed_step_from_its_midpoint():
    kite = SimpleNamespace(gain_roll_scale=2.0)
    scales, solve = _recording_solve(kite, fail_at={1.5})
    assert continue_roll_gain(kite, 2.0, solve) == "x1"
    # 1.5 failed -> midpoint 1.75 from the solved 2.0, then 1.5 again
    assert scales[:3] == [1.5, 1.75, 1.5]
    assert scales[-1] == 1.0


def test_continue_roll_gain_gives_up_after_the_halvings_and_restores_the_kite():
    kite = SimpleNamespace(gain_roll_scale=2.0)
    calls = []

    def solve():
        calls.append(kite.gain_roll_scale)
        return None

    assert continue_roll_gain(kite, 2.0, solve) is None
    assert len(calls) == ROLL_ASSIST_MAX_HALVINGS + 1
    assert kite.gain_roll_scale == 1.0


class FakePhase:
    """Marches ``reach(scale)`` nodes (None = raises) at the kite's scale."""

    def __init__(self, kite, reach):
        self.kite, self.reach = kite, reach

    def run_simulation(self, **kwargs):
        self.kwargs = kwargs
        nodes = self.reach(self.kite.gain_roll_scale)
        if nodes is None:
            raise RuntimeError("Phase residual solve failed at the first node")
        return SimpleNamespace(return_variable=lambda name: np.zeros(nodes)), None


def _march(reach):
    kite = SimpleNamespace(gain_roll_scale=1.0)
    return kite, march_with_roll_assist(
        kite, lambda: FakePhase(kite, reach), N, seed_warm_start=True
    )


def test_march_returns_the_smallest_scale_that_trims():
    kite, (cycle, phase, scale, failure) = _march(
        lambda s: N if s >= 1.5 else (None if s == 1.0 else 4)
    )
    assert scale == 1.5 and kite.gain_roll_scale == 1.5
    assert isinstance(failure, RuntimeError)
    assert cycle.kwargs == {"phase_sim": True, "seed_warm_start": True}


def test_march_at_scale_one_reports_no_failure():
    kite, (_, _, scale, failure) = _march(lambda s: N)
    assert scale == 1.0 and failure is None


def test_march_without_a_trimming_scale_keeps_the_furthest_attempt():
    kite, (cycle, phase, scale, failure) = _march(lambda s: int(4 * s))
    assert scale is None and failure is not None
    assert phase.reached_scale == pytest.approx(2.0)
    assert kite.gain_roll_scale == pytest.approx(2.0)

"""The power-curve sweep's walk: cold until a wind stalls, then continuation."""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples" / "4_trajectory_optimization" / "cycle" / "run_cycle_power_curve.py"
)


@pytest.fixture(scope="module")
def curve():
    import sys

    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_cycle_power_curve", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _walk(curve, winds, cold_fails=(), continued_fails=(), results=None):
    results = {} if results is None else results
    calls = []

    def cold(wind):
        calls.append(("cold", wind))
        return {"wind": wind, "ok": wind not in cold_fails, "method": "cold"}

    def continued(wind, previous):
        calls.append(("continued", wind, previous["wind"]))
        return {"wind": wind, "ok": wind not in continued_fails, "method": "continued"}

    curve.walk_direction(winds, {"wind": 7.0, "ok": True, "method": "cold"}, results, cold, continued)
    return calls, results


def test_parse_winds_range_and_list(curve):
    assert curve.parse_winds("4:5:0.5") == [4.0, 4.5, 5.0]
    assert curve.parse_winds("7,4,5") == [4.0, 5.0, 7.0]


def test_outward_winds_start_at_the_nearest_listed_wind(curve):
    assert curve.outward_winds([4.0, 5.0, 6.0, 7.0], 5.2) == (5.0, [6.0, 7.0], [4.0])


def test_cold_until_a_wind_stalls_then_only_continuation(curve):
    calls, results = _walk(curve, [8.0, 9.0, 10.0, 11.0], cold_fails={9.0})
    assert calls == [
        ("cold", 8.0), ("cold", 9.0), ("continued", 9.0, 8.0),
        ("continued", 10.0, 9.0), ("continued", 11.0, 10.0),
    ]
    assert all(results[w]["ok"] for w in (8.0, 9.0, 10.0, 11.0))


def test_a_failed_continuation_ends_the_direction(curve):
    calls, results = _walk(curve, [8.0, 9.0, 10.0], cold_fails={8.0}, continued_fails={9.0})
    assert calls == [("cold", 8.0), ("continued", 8.0, 7.0), ("continued", 9.0, 8.0)]
    assert not results[9.0]["ok"] and 10.0 not in results


def test_resume_keeps_rows_and_walks_on_as_they_would_have(curve):
    stored = {
        8.0: {"wind": 8.0, "ok": True, "method": "cold"},
        9.0: {"wind": 9.0, "ok": False, "method": "cold"},
    }
    calls, _ = _walk(curve, [8.0, 9.0, 10.0], results=stored)
    assert calls == [("continued", 9.0, 8.0), ("continued", 10.0, 9.0)]


def test_resume_stops_at_a_stored_failed_continuation(curve):
    stored = {8.0: {"wind": 8.0, "ok": False, "method": "continued"}}
    calls, _ = _walk(curve, [8.0, 9.0], results=stored)
    assert calls == []

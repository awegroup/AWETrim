"""Unit tests for the pure analysis layer of awetrim.timeseries.timeseries.TimeSeries.

The ``TimeSeries`` analysis methods operate on a list of plain state dicts
(``State.to_dict()``), so they can be exercised with synthetic states and no
solver / kite model at all. Only the model-function fallback of
``return_variable`` needs a model, and that is covered with a tiny CasADi
``Function`` stub.
"""

import casadi as ca
import numpy as np
import pytest

from awetrim.system.state import State
from awetrim.timeseries.timeseries import TimeSeries


def _states(**arrays):
    """Build a list of state dicts from equal-length keyword arrays."""
    n = len(next(iter(arrays.values())))
    return [State(**{k: arrays[k][i] for k in arrays}).to_dict() for i in range(n)]


@pytest.fixture
def constant_series():
    ts = TimeSeries(kite_model=None)
    ts.states = _states(
        t=[0.0, 1.0, 2.0],
        tension_tether_ground=[1000.0, 1000.0, 1000.0],
        speed_radial=[2.0, 2.0, 2.0],
        speed_tangential=[20.0, 20.0, 20.0],
        s=[0.0, 0.1, 0.2],
        distance_radial=[200.0, 200.0, 200.0],
        angle_elevation=[0.0, 0.0, 0.0],
        angle_azimuth=[0.0, 0.0, 0.0],
    )
    return ts


def test_return_variable_reads_direct_field(constant_series):
    t = constant_series.return_variable("t")
    assert isinstance(t, np.ndarray)
    assert t == pytest.approx([0.0, 1.0, 2.0])


def test_return_variable_derives_cartesian_on_axis():
    ts = TimeSeries(kite_model=None)
    ts.states = _states(
        distance_radial=[100.0],
        angle_elevation=[0.0],
        angle_azimuth=[0.0],
    )
    # On the wind-x axis: x = r, y = z = 0.
    assert ts.return_variable("x")[0] == pytest.approx(100.0)
    assert ts.return_variable("y")[0] == pytest.approx(0.0)
    assert ts.return_variable("z")[0] == pytest.approx(0.0)


def test_return_variable_cartesian_elevation():
    ts = TimeSeries(kite_model=None)
    ts.states = _states(
        distance_radial=[100.0],
        angle_elevation=[np.pi / 2],  # straight up
        angle_azimuth=[0.0],
    )
    assert ts.return_variable("z")[0] == pytest.approx(100.0)
    assert ts.return_variable("x")[0] == pytest.approx(0.0, abs=1e-9)


def test_return_variable_model_function_fallback():
    """A variable that is neither a state field nor x/y/z is computed via the
    kite model's ``extract_function``."""

    class StubModel:
        def extract_function(self, name):
            r = ca.MX.sym("distance_radial")
            return ca.Function("f", [r], [2.0 * r], ["distance_radial"], [name])

    ts = TimeSeries(kite_model=StubModel())
    ts.states = _states(distance_radial=[10.0, 20.0])
    assert ts.return_variable("my_derived")[0] == pytest.approx(20.0)
    assert ts.return_variable("my_derived")[1] == pytest.approx(40.0)


def test_energy_left_riemann_sum(constant_series):
    # dt = diff(t, prepend=t[0]) = [0, 1, 1]; energy = sum(T * vr * dt) = 4000.
    assert constant_series.energy == pytest.approx(4000.0)


def test_total_time(constant_series):
    assert constant_series.total_time == pytest.approx(2.0)


def test_energy_metrics_returns_expected_keys(constant_series):
    metrics = constant_series.energy_metrics()
    expected = {
        "energy",
        "avg_power",
        "total_time",
        "tension_mean",
        "tension_max",
        "tension_min",
        "vtau_mean",
        "vtau_max",
        "vtau_min",
        "phase_start_deg",
        "phase_end_deg",
    }
    assert expected <= set(metrics)
    assert metrics["tension_mean"] == pytest.approx(1000.0)
    assert metrics["vtau_mean"] == pytest.approx(20.0)
    assert metrics["energy"] >= 0.0

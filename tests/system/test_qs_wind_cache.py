"""solve_quasi_steady must not reuse a graph built with a different wind."""

import contextlib
import io

import numpy as np
import pytest

from awetrim.system.factory import create_system_model_from_yaml
from awetrim.system.state import State
from awetrim.utils.config_paths import LEI_V3_SYSTEM_FLOWN_2019_CONFIG


def _model():
    with contextlib.redirect_stdout(io.StringIO()):
        model = create_system_model_from_yaml(LEI_V3_SYSTEM_FLOWN_2019_CONFIG)
    model.wind.wind_model = "uniform"
    model.wind.direction_wind = 0.0
    return model


def _solve(model, wind_speed):
    model.wind.speed_wind_ref = wind_speed
    state = State(
        distance_radial=200.0, angle_elevation=np.deg2rad(30.0), angle_azimuth=0.0,
        angle_course=np.pi / 2, speed_radial=0.0, timeder_speed_radial=0.0,
        timeder_speed_tangential=0.0, input_depower=1.7, input_steering=0.0,
        speed_tangential=4.0 * wind_speed, timeder_angle_course=0.0,
        tension_tether_ground=3000.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return model.solve_quasi_steady(state)


def test_changing_the_wind_rebuilds_the_solver_and_derived_functions():
    reused = _model()
    _solve(reused, 6.0)
    second = _solve(reused, 10.0)
    fresh = _solve(_model(), 10.0)
    assert second is not None and fresh is not None
    for name in ("speed_tangential", "tension_tether_ground", "angle_of_attack"):
        assert getattr(second, name) == pytest.approx(getattr(fresh, name), rel=1e-6)


def test_unchanged_wind_keeps_the_built_solver():
    model = _model()
    _solve(model, 8.0)
    solver = model._qs_solver
    _solve(model, 8.0)
    assert model._qs_solver is solver

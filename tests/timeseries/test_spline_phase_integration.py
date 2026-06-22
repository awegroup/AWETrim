"""Integration tests: build a Lissajous and a helix B-spline reel-out path for
the LEI-V3 kite and march the parametrised quasi-steady phase along each.

Mirrors the flow in
``scripts/reduced-order-model/optimization/reelout/{generate_spline_config,helix_pattern}.py``:
generate path parameters from a named initial curve, drop them into the V3
helix template config, build the V3 system model (rigid lumped tether), then
run ``PhaseParameterized.run_simulation_phase``.

The QS marching solve calls IPOPT at every s-node, so the end-to-end test is
marked ``slow``. The spline-generation check above it is pure and always runs.
"""

import copy

import numpy as np
import pytest

from awetrim.kinematics.parametrized_patterns import (
    make_bspline_path_parameters_from_named_curve,
)
from awetrim.utils.config_paths import (
    LEI_V3_HELIX_SPLINE_CONFIG,
    LEI_V3_SYSTEM_CONFIG,
)
from awetrim.utils.utils import load_cycle_config_from_yaml

# Initial-curve shape (matches the defaults in generate_spline_config.py).
_M = 10
_R0 = 230.0
_S_INIT = 0.0
_S_FINAL = 2.0 * np.pi
_CURVE_KWARGS = dict(az_amp0=0.32, beta0=0.3, beta_amp0=0.15, downloops=True)

pytestmark = pytest.mark.skipif(
    not LEI_V3_SYSTEM_CONFIG.exists() or not LEI_V3_HELIX_SPLINE_CONFIG.exists(),
    reason="LEI-V3 kite data not available",
)


def _spline_path_parameters(curve_type):
    return make_bspline_path_parameters_from_named_curve(
        spline_type="periodic",
        M=_M,
        r0=_R0,
        s_init=_S_INIT,
        s_final=_S_FINAL,
        n_fit=200,
        curve_type=curve_type,
        **_CURVE_KWARGS,
    )


def _reelout_config(curve_type, n_points):
    """V3 helix template config with its path swapped for a named-curve spline."""
    reelout_config, _ = load_cycle_config_from_yaml(LEI_V3_HELIX_SPLINE_CONFIG)
    reelout_config = copy.deepcopy(reelout_config)
    reelout_config["pattern_type"] = "spline_periodic"
    reelout_config["path_parameters"] = _spline_path_parameters(curve_type)
    sim = reelout_config["sim_parameters"]
    sim["start_angle"] = _S_INIT
    sim["end_angle"] = _S_FINAL
    sim["n_points"] = n_points
    return reelout_config


def _v3_system_model(wind_speed=10.0):
    from awetrim.environment.Wind import Wind
    from awetrim.system.factory import create_system_model_from_yaml

    system_model = create_system_model_from_yaml(yaml_path=LEI_V3_SYSTEM_CONFIG)
    wind = Wind(wind_model="uniform", z0=0.03, direction_wind=0)
    wind.speed_wind_ref = wind_speed
    system_model.wind = wind
    return system_model


# --- pure: spline generation -----------------------------------------------


@pytest.mark.parametrize("curve_type", ["lissajous", "lemniscate", "helix"])
def test_named_curve_spline_parameters_shape(curve_type):
    params = _spline_path_parameters(curve_type)
    assert params["M"] == _M
    assert params["r0"] == pytest.approx(_R0)
    assert params["s_init"] == pytest.approx(_S_INIT)
    assert params["s_final"] == pytest.approx(_S_FINAL)
    assert len(params["C_phi"]) == _M
    assert len(params["C_beta"]) == _M
    assert np.all(np.isfinite(params["C_phi"]))
    assert np.all(np.isfinite(params["C_beta"]))


def test_lissajous_and_helix_differ():
    """The two curves share the azimuth law but differ in elevation
    (sin(2s) vs cos(s)), so their fitted elevation coefficients must differ."""
    liss = _spline_path_parameters("lissajous")
    heli = _spline_path_parameters("helix")
    assert not np.allclose(liss["C_beta"], heli["C_beta"])


# --- slow: run the parametrised QS phase along each spline ------------------


@pytest.mark.slow
@pytest.mark.parametrize("curve_type", ["lissajous", "lemniscate", "helix"])
def test_spline_phase_runs_quasi_steady(curve_type):
    from awetrim.timeseries.phase_parametrized import PhaseParameterized

    n_points = 30
    config = _reelout_config(curve_type, n_points=n_points)
    system_model = _v3_system_model()

    start_state = {
        "t": 0.0,
        "s": _S_INIT,
        "s_dot": 3.0,
        "input_steering": 0.0,
        "tension_tether_ground": 8.4e4,  # initial guess (N)
        "speed_radial": 0.0,
        "distance_radial": config["path_parameters"]["r0"],
    }

    phase = PhaseParameterized(
        system_model,
        quasi_steady=True,
        pattern_config=config,
    )
    phase.run_simulation_phase(start_state=start_state, allow_failure=False)

    # One recorded state per s-node.
    assert len(phase.states) == n_points

    # Every directly recorded decision/state variable is finite.
    for var in (
        "tension_tether_ground",
        "speed_radial",
        "input_steering",
        "s_dot",
        "distance_radial",
    ):
        values = phase.return_variable(var)
        assert len(values) == n_points
        assert np.all(np.isfinite(values)), f"{var} has non-finite entries"

    # The phase coordinate marches across the full s-grid [0, 2π].
    s = phase.return_variable("s")
    assert s[0] == pytest.approx(_S_INIT, abs=1e-6)
    assert s[-1] > s[0]

    # Tether stays in tension and the kite stays above the horizon.
    assert np.all(phase.return_variable("tension_tether_ground") > 0.0)
    assert np.all(phase.return_variable("angle_elevation") > 0.0)


# --- slow: per-node depower-profile optimization ---------------------------


@pytest.mark.slow
def test_depower_profile_builds_per_node_decision():
    """With ``optimize_depower_profile``, ``opti_phase`` exposes the depower
    input as one decision per s-node (like ``input_steering``) instead of a
    scalar, and the per-node NLP assembles without error."""
    import casadi as ca

    from awetrim.timeseries.phase_parametrized import PhaseParameterized

    n_points = 20
    config = _reelout_config("lissajous", n_points=n_points)
    config["sim_parameters"]["input_depower"] = 1.6
    config["sim_parameters"]["optimize_depower_profile"] = True
    config["sim_parameters"]["depower_rate"] = (-0.2, 0.2)
    system_model = _v3_system_model()

    start_state = {
        "t": 0.0,
        "s": _S_INIT,
        "s_dot": 3.0,
        "input_steering": 0.0,
        "tension_tether_ground": 8.4e4,
        "speed_radial": 0.0,
        "distance_radial": config["path_parameters"]["r0"],
    }

    phase = PhaseParameterized(system_model, quasi_steady=True, pattern_config=config)
    opti, opti_vars, _ = phase.opti_phase(start_state=start_state, opti_params={})

    # Depower is now a per-node trajectory decision, not a scalar.
    assert "input_depower" in opti_vars
    depower = opti_vars["input_depower"]
    assert isinstance(depower, ca.MX)
    assert depower.shape == (n_points, 1)

    # Without the flag, no per-node depower variable is created (scalar path).
    config_scalar = _reelout_config("lissajous", n_points=n_points)
    config_scalar["sim_parameters"]["input_depower"] = 1.6
    phase_scalar = PhaseParameterized(
        system_model, quasi_steady=True, pattern_config=config_scalar
    )
    _, opti_vars_scalar, _ = phase_scalar.opti_phase(
        start_state=start_state, opti_params={}
    )
    assert "input_depower" not in opti_vars_scalar

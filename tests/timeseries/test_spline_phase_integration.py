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


# --- slow: r0 as a named optimization parameter -----------------------------


@pytest.mark.slow
def test_r0_opti_param_pins_initial_radius_symbolically():
    """With ``r0`` in ``opti_params``, the node-0 radius pin follows the r0
    symbol instead of the numeric start state, so the operating radius is a
    design variable. Asserts constraint structure, not solver values."""
    import casadi as ca

    from awetrim.timeseries.phase_parametrized import PhaseParameterized

    n_points = 20
    config = _reelout_config("lissajous", n_points=n_points)
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

    opti = ca.Opti()
    r0_var = opti.variable()
    phase = PhaseParameterized(system_model, quasi_steady=True, pattern_config=config)
    opti_out, _, _ = phase.opti_phase(
        start_state=start_state, opti=opti, opti_params={"r0": r0_var}
    )

    # r0 enters the constraint set: its own bounds (2 rows) plus the node-0
    # radius pin (a 3rd row). Bounds alone would only give 2 dependent rows,
    # so >= 3 proves the pin is tied to the symbol.
    assert ca.depends_on(opti_out.g, r0_var)
    jac_rows = ca.jacobian(opti_out.g, r0_var)
    assert jac_rows.nnz() >= 3


# --- slow: per-element trust-region step bounds ------------------------------


@pytest.mark.slow
def test_vector_param_step_bound_builds_and_validates_length():
    """``param_step_bound`` accepts a per-element delta vector (wider box for
    the reel-in control points); a wrong-length vector fails loudly."""
    import casadi as ca

    from awetrim.timeseries.phase_parametrized import PhaseParameterized

    n_points = 20
    config = _reelout_config("lissajous", n_points=n_points)
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

    # Per-element box (last two points wider) assembles and ties the symbol
    # into the constraint set.
    config["sim_parameters"]["param_step_bound"] = {
        "C_beta": [0.1] * (_M - 2) + [0.5] * 2
    }
    opti = ca.Opti()
    c_beta_var = opti.variable(_M)
    phase = PhaseParameterized(system_model, quasi_steady=True, pattern_config=config)
    opti_out, _, _ = phase.opti_phase(
        start_state=start_state, opti=opti, opti_params={"C_beta": c_beta_var}
    )
    assert ca.depends_on(opti_out.g, c_beta_var)

    # Wrong-length vector: refused up front, not silently broadcast.
    config_bad = _reelout_config("lissajous", n_points=n_points)
    config_bad["sim_parameters"]["param_step_bound"] = {"C_beta": [0.1, 0.2]}
    opti_bad = ca.Opti()
    c_beta_bad = opti_bad.variable(_M)
    phase_bad = PhaseParameterized(
        system_model, quasi_steady=True, pattern_config=config_bad
    )
    with pytest.raises(ValueError, match="param_step_bound"):
        phase_bad.opti_phase(
            start_state=start_state,
            opti=opti_bad,
            opti_params={"C_beta": c_beta_bad},
        )


# --- slow: previous-optimum warm start seeds the NLP decisions ---------------


@pytest.mark.slow
def test_warm_start_trajectory_seeds_decisions_over_forward_sim():
    """A stored NLP optimum (``warm_start_trajectory``, injected between
    staged solves) seeds the per-node decisions instead of the force-law
    forward simulation; variables it lacks still fall back to the sim."""
    import casadi as ca

    from awetrim.timeseries.phase_parametrized import PhaseParameterized

    n_points = 20
    config = _reelout_config("lissajous", n_points=n_points)
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
    marker_sdot, marker_vr = 7.7, 0.123
    phase.warm_start_trajectory = {
        "s_dot": np.full(n_points, marker_sdot),
        "speed_radial": np.full(n_points, marker_vr),
        "stale_wrong_size": np.zeros(3),  # grid mismatch: must be ignored
    }
    opti, opti_vars, _ = phase.opti_phase(start_state=start_state, opti_params={})

    seeded_sdot = np.asarray(opti.value(opti_vars["s_dot"], opti.initial())).ravel()
    seeded_vr = np.asarray(
        opti.value(opti_vars["speed_radial"], opti.initial())
    ).ravel()
    assert np.allclose(seeded_sdot, marker_sdot)
    assert np.allclose(seeded_vr, marker_vr)

    # Variables not in the stored optimum keep the forward-sim seed (finite,
    # non-constant physical trajectory -- not the marker).
    seeded_steering = np.asarray(
        opti.value(opti_vars["input_steering"], opti.initial())
    ).ravel()
    assert np.all(np.isfinite(seeded_steering))
    assert not np.allclose(seeded_steering, marker_sdot)


# --- slow: constraint report exposes the NLP's constrained expressions ------


@pytest.mark.slow
def test_constraint_report_exposes_nlp_expressions():
    """``opti_phase`` exports every constrained quantity (height, AoA, rates,
    per-node bounds, equality residuals) through the objective dict, so
    ``Phase._print_constraint_report`` can show margins after each solve."""
    import casadi as ca

    from awetrim.timeseries.phase_parametrized import PhaseParameterized

    n_points = 20
    config = _reelout_config("lissajous", n_points=n_points)
    config["sim_parameters"]["input_depower"] = 1.6
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
    _, _, objective_dict = phase.opti_phase(start_state=start_state, opti_params={})

    report = objective_dict["constraint_report"]

    # Inequality rows carry the NLP expression plus the applied bounds.
    for name, n_expected in {
        "height": n_points,
        "angle_of_attack": n_points - 1,
        "distance_radial": n_points,
        "input_steering_rate": n_points - 1,
        "tension_tether_ground": n_points,
    }.items():
        spec = report[name]
        assert isinstance(spec["expr"], ca.MX), name
        assert spec["expr"].numel() == n_expected, name
        assert spec["lb"] < spec["ub"], name

    # Equality groups expose the NLP's own scaled residuals.
    assert report["trim_residual (scaled)"]["equality"]
    assert report["trim_residual (scaled)"]["expr"].numel() == 3 * n_points
    assert report["radial_continuity (scaled)"]["expr"].numel() == n_points - 1
    assert report["winch_tension_law (scaled)"]["expr"].numel() == n_points


# --- slow: minimum turn-radius constraint --------------------------------------


@pytest.mark.slow
def test_min_turn_radius_adds_dense_rows_and_diagnostics():
    """``sim_parameters["min_turn_radius"]`` adds three rows per sub-sample per
    node (default 4 per interval), exports the ``turn_radius`` report row and
    always exposes the per-node ``turn_radius`` / scalar ``turn_radius_min``
    diagnostics; without it the constraint set is unchanged."""
    import casadi as ca

    from awetrim.timeseries.phase_parametrized import PhaseParameterized

    n_points = 20
    system_model = _v3_system_model()

    def _build(**sim_overrides):
        config = _reelout_config("lissajous", n_points=n_points)
        config["sim_parameters"]["input_depower"] = 1.6
        config["sim_parameters"].update(sim_overrides)
        start_state = {
            "t": 0.0,
            "s": _S_INIT,
            "s_dot": 3.0,
            "input_steering": 0.0,
            "tension_tether_ground": 8.4e4,
            "speed_radial": 0.0,
            "distance_radial": config["path_parameters"]["r0"],
        }
        phase = PhaseParameterized(
            system_model, quasi_steady=True, pattern_config=config
        )
        opti, _, objective_dict = phase.opti_phase(
            start_state=start_state, opti_params={}
        )
        return opti, objective_dict

    opti_off, od_off = _build()
    opti_on, od_on = _build(min_turn_radius=12.0)
    opti_k1, _ = _build(min_turn_radius=12.0, turn_radius_subsamples=1)

    # three rows per sample: -r s'^3 <= R_min kappa s'^3 <= r s'^3 and the
    # parametric-speed floor s'/s'_ref >= turn_radius_sigma_floor
    assert opti_on.g.numel() - opti_off.g.numel() == 3 * 4 * n_points
    assert opti_k1.g.numel() - opti_off.g.numel() == 3 * n_points
    assert "turn_radius" not in od_off["constraint_report"]
    assert "path_param_speed (sigma'/ref)" in od_on["constraint_report"]
    row = od_on["constraint_report"]["turn_radius"]
    assert row["expr"].numel() == 4 * n_points
    assert row["lb"] == 12.0 and row["ub"] == float("inf")
    for od in (od_off, od_on):
        assert isinstance(od["turn_radius"], ca.MX)
        assert od["turn_radius"].numel() == n_points
        assert od["turn_radius_min"].numel() == 1

    with pytest.raises(ValueError, match="min_turn_radius"):
        _build(min_turn_radius=-1.0)

    # sim_parameters["min_azimuth_amplitude"] (the PatternLimits azimuth
    # amplitude floor) is ONE smooth row when C_phi is optimized; with a fixed
    # pattern it is only a check on the given shape; negative values rejected.
    def _build_cphi(**sim_overrides):
        config = _reelout_config("lissajous", n_points=n_points)
        config["sim_parameters"]["input_depower"] = 1.6
        config["sim_parameters"].update(sim_overrides)
        phase = PhaseParameterized(
            system_model, quasi_steady=True, pattern_config=config
        )
        opti = ca.Opti()
        phase.pattern_config_opti = copy.deepcopy(config)
        c_phi = opti.variable(len(config["path_parameters"]["C_phi"]))
        phase.pattern_config_opti["path_parameters"]["C_phi"] = c_phi
        start_state = {
            "t": 0.0, "s": _S_INIT, "s_dot": 3.0, "input_steering": 0.0,
            "tension_tether_ground": 8.4e4, "speed_radial": 0.0,
            "distance_radial": config["path_parameters"]["r0"],
        }
        opti, _, objective_dict = phase.opti_phase(
            start_state=start_state, opti=opti, opti_params={"C_phi": c_phi}
        )
        return opti, objective_dict

    opti_c_off, od_c_off = _build_cphi()
    opti_amp, od_amp = _build_cphi(min_azimuth_amplitude=np.radians(5.0))
    assert opti_amp.g.numel() - opti_c_off.g.numel() == 1
    row = od_amp["constraint_report"]["azimuth_amplitude (rms*sqrt2)"]
    assert row["lb"] == pytest.approx(np.radians(5.0)) and row["ub"] == float("inf")
    assert "azimuth_amplitude (rms*sqrt2)" not in od_c_off["constraint_report"]
    # fixed pattern: satisfied -> no row; violated -> error
    opti_fixed, _ = _build(min_azimuth_amplitude=np.radians(5.0))
    assert opti_fixed.g.numel() == opti_off.g.numel()
    with pytest.raises(ValueError, match="min_azimuth_amplitude"):
        _build(min_azimuth_amplitude=np.radians(89.0))
    with pytest.raises(ValueError, match="min_azimuth_amplitude"):
        _build(min_azimuth_amplitude=-0.1)

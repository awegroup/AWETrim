"""Schema validation tests for the reelout optimization server."""

import inspect

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from pydantic import ValidationError

from awetrim.server.schemas import (
    InflowConditions,
    InitialGuess,
    InitRequest,
    StepRequest,
    wind_kwargs_from_inflow_schema,
)


def test_initial_guess_matches_named_curve_signature():
    """InitialGuess fields must be valid kwargs of the fitting helper."""
    from awetrim.kinematics.parametrized_patterns import (
        make_bspline_path_parameters_from_named_curve,
    )

    accepted = set(
        inspect.signature(make_bspline_path_parameters_from_named_curve).parameters
    )
    fields = set(InitialGuess.model_fields)
    unknown = fields - accepted
    assert not unknown, f"InitialGuess fields not accepted by helper: {unknown}"


def test_inflow_conditions_defaults():
    """Optional fields default to the values documented in the struct."""
    inflow = InflowConditions(wind_speed=8.0, profile_law=2)
    assert inflow.wind_direction == 0.0
    assert inflow.alpha == pytest.approx(0.08163)
    assert inflow.z0 == pytest.approx(0.0002)
    assert inflow.turbulence == 0.0
    assert inflow.heights is None and inflow.speeds is None

    kwargs = wind_kwargs_from_inflow_schema(inflow)
    assert kwargs["model_type"] == "logarithmic"
    assert kwargs["U_ref"] == 8.0
    assert kwargs["z_ref"] == 6.0  # wind_speed is given at 6 m


def test_inflow_conditions_rejects_invalid_input():
    with pytest.raises(ValidationError):
        InflowConditions(wind_speed=0.0, profile_law=2)  # gt=0
    with pytest.raises(ValidationError):
        InflowConditions(wind_speed=8.0, profile_law=7)  # unknown law
    with pytest.raises(ValidationError):
        InflowConditions(wind_speed=8.0, profile_law=2, turbulence=1.5)
    with pytest.raises(ValidationError):
        InflowConditions(wind_speed=8.0, profile_law=2, heights=[10.0])
    with pytest.raises(ValidationError):  # CUSTOM_JET needs >= 5 samples
        InflowConditions(wind_speed=8.0, profile_law=6,
                         heights=[10.0, 100.0], speeds=[5.0, 8.0])


def test_init_request_defaults():
    req = InitRequest(inflow_conditions={"wind_speed": 8.0, "profile_law": 0})
    assert req.optimization_params == ["C_phi", "C_beta", "input_depower"]
    assert req.target == "power"
    assert req.n_points == 100
    assert req.initial_guess is None
    # the solver knobs are unset by default: the cycle config decides
    assert req.input_depower is None
    assert req.reg_weight is None
    assert req.detect_simple_bounds is None


def test_init_request_needs_inflow_conditions():
    with pytest.raises(ValidationError):
        InitRequest()
    # the removed low-level wind field is rejected, not silently ignored
    with pytest.raises(ValidationError):
        InitRequest(inflow_conditions={"wind_speed": 8.0, "profile_law": 0},
                    wind={"model_type": "uniform", "U_ref": 8.0})
    # so is the removed generic solver-knob dict
    with pytest.raises(ValidationError):
        InitRequest(inflow_conditions={"wind_speed": 8.0, "profile_law": 0},
                    sim_parameters={"input_depower": 1.6})


def test_step_request_optional_fields():
    req = StepRequest()
    assert req.inflow_conditions is None
    assert req.length is None and req.max_iter is None
    with pytest.raises(ValidationError):
        StepRequest(length=-10.0)


# --- develop features grafted onto the struct contract -------------------------


def _init_kwargs(**extra):
    return dict(
        inflow_conditions={"wind_speed": 6.0, "wind_direction": 270.0, "profile_law": 2},
        **extra,
    )


def test_depower_struct_is_optional_and_validated():
    from awetrim.server.schemas import DepowerReply, DepowerSpec

    assert InitRequest(**_init_kwargs()).depower is None
    assert StepRequest().depower is None
    req = InitRequest(**_init_kwargs(depower={"mode": "fixed", "value": 1.6}))
    assert req.depower.mode == "fixed" and req.depower.value == 1.6
    assert DepowerSpec().mode == "optimize"
    with pytest.raises(ValidationError):
        DepowerSpec(mode="disabled")
    reply = DepowerReply(mode="profile", value=1.4, profile=[1.3, 1.5])
    assert reply.profile == [1.3, 1.5]


def test_winch_params_take_v_max_or_p_max_not_both():
    from awetrim.server.schemas import WinchParams

    base = dict(mode="reelout", k_v=0.0408, f_min=350.0, f_max=7600.0)
    assert WinchParams(**base).reel_speed_limit() is None
    assert WinchParams(**base, v_max=8.0).reel_speed_limit() == pytest.approx(8.0)
    assert WinchParams(**base, p_max=38000.0).reel_speed_limit() == pytest.approx(5.0)
    with pytest.raises(ValidationError):
        WinchParams(**base, v_max=8.0, p_max=38000.0)


def test_winch_k_v_optimization_is_opt_in_and_bracket_is_validated():
    from awetrim.server.schemas import WinchParams

    base = dict(mode="reelout", k_v=0.04, f_min=1000.0, f_max=8400.0)
    assert WinchParams(**base).optimize_k_v is False
    assert WinchParams(**base).k_v_bounds is None
    assert WinchParams(**base, optimize_k_v=True, k_v_bounds=[0.02, 0.08])

    # The bracket must be a positive, ordered pair containing k_v.
    for bad in ([0.05, 0.08], [0.01, 0.03], [0.08, 0.02], [-0.01, 0.08], [0.04]):
        with pytest.raises(ValidationError):
            WinchParams(**base, optimize_k_v=True, k_v_bounds=bad)


def test_min_turn_radius_is_optional_and_non_negative():
    assert InitRequest(**_init_kwargs()).min_turn_radius is None
    assert StepRequest().min_turn_radius is None
    assert InitRequest(**_init_kwargs(min_turn_radius=11.35)).min_turn_radius == pytest.approx(11.35)
    assert StepRequest(min_turn_radius=0).min_turn_radius == 0  # "remove"
    with pytest.raises(ValidationError):
        StepRequest(min_turn_radius=-1.0)


def test_replies_and_metrics_carry_depower_and_turn_radius():
    from awetrim.server.schemas import InitReply, SolveMetrics, StepReply

    traj = {"azimuth": [0.0] * 8, "elevation": [20.0] * 8}
    init = InitReply(name="x", trajectory=traj, state="ready")
    assert init.depower is None and init.min_turn_radius is None
    metrics = SolveMetrics(energy_J=1.0, total_time_s=1.0, avg_power_W=1.0)
    assert metrics.turn_radius_min_m is None
    step = StepReply(
        trajectory=traj, state="converged", step_index=1,
        metrics=dict(metrics.model_dump(), turn_radius_min_m=12.3),
        depower={"mode": "optimize", "value": 1.52, "profile": None},
        min_turn_radius=11.35,
    )
    assert step.depower.value == pytest.approx(1.52)
    assert step.min_turn_radius == pytest.approx(11.35)
    assert step.metrics.turn_radius_min_m == pytest.approx(12.3)


def test_pattern_limits_struct_validates_and_is_optional():
    from awetrim.server.schemas import InitReply, PatternLimits, StepReply

    assert InitRequest(**_init_kwargs()).pattern_limits is None
    assert StepRequest().pattern_limits is None
    lim = PatternLimits(azimuth_max=35.0, elevation_max=45.0, azimuth_amplitude_min=5.0)
    assert lim.elevation_min is None
    req = InitRequest(**_init_kwargs(pattern_limits=lim.model_dump()))
    assert req.pattern_limits.azimuth_max == pytest.approx(35.0)
    # {} on /step is a valid "clear" request and survives as an (empty) struct
    assert StepRequest(pattern_limits={}).pattern_limits is not None
    with pytest.raises(ValidationError):
        PatternLimits(elevation_min=40.0, elevation_max=30.0)
    with pytest.raises(ValidationError):
        PatternLimits(azimuth_max=0.0)
    with pytest.raises(ValidationError):
        PatternLimits(unknown=1.0)
    traj = {"azimuth": [0.0] * 8, "elevation": [20.0] * 8}
    assert InitReply(name="x", trajectory=traj, state="ready").pattern_limits is None
    step = StepReply(
        trajectory=traj, state="converged", step_index=1,
        metrics={"energy_J": 1.0, "total_time_s": 1.0, "avg_power_W": 1.0},
        pattern_limits={"azimuth_max": 35.0},
    )
    assert step.pattern_limits.azimuth_max == pytest.approx(35.0)

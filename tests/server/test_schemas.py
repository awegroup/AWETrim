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
    assert req.sim_parameters == {}


def test_init_request_needs_inflow_conditions():
    with pytest.raises(ValidationError):
        InitRequest()
    # the removed low-level wind field is rejected, not silently ignored
    with pytest.raises(ValidationError):
        InitRequest(inflow_conditions={"wind_speed": 8.0, "profile_law": 0},
                    wind={"model_type": "uniform", "U_ref": 8.0})


def test_step_request_optional_fields():
    req = StepRequest()
    assert req.inflow_conditions is None
    assert req.length is None and req.max_iter is None
    with pytest.raises(ValidationError):
        StepRequest(length=-10.0)

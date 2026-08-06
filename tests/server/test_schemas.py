"""Schema validation tests for the reelout optimization server."""

import inspect

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from pydantic import TypeAdapter, ValidationError

from awetrim.server.schemas import (
    InflowConditions,
    InitialGuess,
    InitRequest,
    StepRequest,
    WindLogarithmic,
    WindProfile,
    WindTabulated,
    WindUniform,
    wind_kwargs_from_inflow_schema,
    wind_kwargs_from_schema,
)

WIND_ADAPTER = TypeAdapter(WindProfile)


def test_wind_union_discriminates_on_model_type():
    log = WIND_ADAPTER.validate_python(
        {"model_type": "logarithmic", "U_ref": 8.0, "z_ref": 100.0, "z0": 0.03}
    )
    assert isinstance(log, WindLogarithmic)
    uni = WIND_ADAPTER.validate_python({"model_type": "uniform", "U_ref": 7.0})
    assert isinstance(uni, WindUniform)
    tab = WIND_ADAPTER.validate_python(
        {"model_type": "tabulated", "heights": [10, 100], "speeds": [5, 8]}
    )
    assert isinstance(tab, WindTabulated)


def test_wind_union_rejects_unknown_model_type():
    with pytest.raises(ValidationError):
        WIND_ADAPTER.validate_python({"model_type": "gaussian", "U_ref": 8.0})


def test_tabulated_wind_rejects_bad_tables():
    with pytest.raises(ValidationError):
        WindTabulated(model_type="tabulated", heights=[10, 100], speeds=[5])
    with pytest.raises(ValidationError):
        WindTabulated(model_type="tabulated", heights=[100, 10], speeds=[5, 8])
    with pytest.raises(ValidationError):
        WindTabulated(model_type="tabulated", heights=[-5, 10], speeds=[5, 8])
    with pytest.raises(ValidationError):
        WindTabulated(model_type="tabulated", heights=[10], speeds=[5])


def test_wind_kwargs_round_trip():
    log = WindLogarithmic(model_type="logarithmic", U_ref=8.0)
    kw = wind_kwargs_from_schema(log)
    assert kw["model_type"] == "logarithmic"
    assert kw["U_ref"] == 8.0
    assert kw["z_ref"] == 100.0
    tab = WindTabulated(
        model_type="tabulated", heights=[10, 100], speeds=[5, 8]
    )
    kw = wind_kwargs_from_schema(tab)
    assert kw["heights"] == [10, 100]
    assert "U_ref" not in kw


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
    assert req.wind is None


def test_init_request_needs_wind():
    with pytest.raises(ValidationError, match="inflow_conditions is required"):
        InitRequest()
    # the low-level wind field remains an accepted alternative
    assert InitRequest(wind={"model_type": "uniform", "U_ref": 8.0}) is not None


def test_step_request_optional_fields():
    req = StepRequest()
    assert req.inflow_conditions is None and req.wind is None
    assert req.length is None and req.max_iter is None
    with pytest.raises(ValidationError):
        StepRequest(length=-10.0)

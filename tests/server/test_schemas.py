"""Schema validation tests for the reelout optimization server."""

import inspect

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from pydantic import TypeAdapter, ValidationError

from awetrim.server.schemas import (
    InitialGuess,
    InitRequest,
    StepRequest,
    WindLogarithmic,
    WindProfile,
    WindTabulated,
    WindUniform,
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


def test_init_request_defaults():
    req = InitRequest(wind={"model_type": "uniform", "U_ref": 8.0})
    assert req.optimization_params == ["C_phi", "C_beta", "input_depower"]
    assert req.target == "power"
    assert req.n_points == 100
    assert req.initial_guess is None
    assert req.sim_parameters == {}


def test_step_request_optional_fields():
    req = StepRequest()
    assert req.wind is None and req.distance_radial is None and req.max_iter is None
    with pytest.raises(ValidationError):
        StepRequest(distance_radial=-10.0)


def test_depower_spec_defaults_and_validation():
    from awetrim.server.schemas import DepowerSpec

    spec = DepowerSpec()
    assert spec.mode == "optimize" and spec.value is None
    assert DepowerSpec(mode="fixed", value=1.6).value == 1.6
    with pytest.raises(ValidationError):
        DepowerSpec(mode="disabled")


def test_depower_is_optional_on_both_requests():
    assert InitRequest(wind={"model_type": "uniform", "U_ref": 8.0}).depower is None
    assert StepRequest().depower is None
    req = StepRequest(depower={"mode": "profile", "value": 1.4})
    assert req.depower.mode == "profile" and req.depower.value == 1.4


def test_depower_reply_carries_profile():
    from awetrim.server.schemas import DepowerReply

    reply = DepowerReply(mode="optimize", value=1.52)
    assert reply.profile is None
    profiled = DepowerReply(mode="profile", value=1.4, profile=[1.3, 1.5])
    assert profiled.profile == [1.3, 1.5]

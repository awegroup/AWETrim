"""AWETrim states its VSM settings instead of inheriting VSM's defaults."""

import pytest

pytest.importorskip("VSM")

from awetrim.aerodynamics import vsm_settings  # noqa: E402
from awetrim.aerodynamics.vsm_quasi_steady import default_vsm_solver  # noqa: E402
from awetrim.aerodynamics.vsm_settings import (  # noqa: E402
    VSM_SOLVER_DEFAULTS,
    make_vsm_solver,
    vsm_body_kwargs,
)


def test_every_default_reaches_the_solver():
    solver = make_vsm_solver()
    for key, value in VSM_SOLVER_DEFAULTS.items():
        if hasattr(solver, key):
            assert getattr(solver, key) == value, key


def test_quarter_chord_force_directions_with_attached_trailed_force():
    solver = default_vsm_solver(reference_point=[0.0, 0.0, 0.0])
    assert solver.aerodynamic_model_type == "VSM"
    assert solver.is_aoa_corrected is True
    assert solver.relaxation_factor == VSM_SOLVER_DEFAULTS["relaxation_factor"]
    assert solver.is_with_attached_trailed_vortex_force is True


def test_overrides_take_precedence():
    solver = make_vsm_solver(allowed_error=1e-9, gamma_loop_type="anderson")
    assert solver.allowed_error == 1e-9
    assert solver.gamma_loop_type == "anderson"


def test_absent_option_is_dropped_only_at_its_no_op_value():
    def old_vsm(self, aerodynamic_model_type="VSM"):
        pass

    kept = vsm_settings._supported(
        old_vsm, {"aerodynamic_model_type": "VSM", "is_with_attached_trailed_vortex_force": False}
    )
    assert kept == {"aerodynamic_model_type": "VSM"}
    with pytest.raises(TypeError, match="is_with_attached_trailed_vortex_force"):
        vsm_settings._supported(old_vsm, {"is_with_attached_trailed_vortex_force": True})
    with pytest.raises(TypeError, match="unknown_option"):
        vsm_settings._supported(old_vsm, {"unknown_option": 1})


def test_body_kwargs_place_the_stations():
    kwargs = vsm_body_kwargs()
    assert kwargs["aerodynamic_center_location"] == 0.25
    assert kwargs["control_point_location"] == 0.75


# --------------------------------------------- the ML airfoil models' folder
def test_ml_models_dir_from_config_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(vsm_settings.ML_MODELS_DIR_ENV, "/from/env")
    assert vsm_settings.ml_models_dir({"ml_models_dir": "/from/config"}) == "/from/config"
    assert vsm_settings.ml_models_dir({}) == "/from/env"
    assert vsm_settings.ml_models_dir(None) == "/from/env"


def test_no_ml_models_dir_leaves_the_body_kwargs_unchanged(monkeypatch):
    monkeypatch.delenv(vsm_settings.ML_MODELS_DIR_ENV, raising=False)
    assert vsm_settings.ml_models_dir({}) is None
    assert vsm_settings.vsm_body_kwargs_for({}) == vsm_settings.vsm_body_kwargs()


def test_ml_models_dir_reaches_body_instantiation(monkeypatch):
    """masure_regression needs the folder in BodyAerodynamics.instantiate."""
    monkeypatch.delenv(vsm_settings.ML_MODELS_DIR_ENV, raising=False)
    kwargs = vsm_settings.vsm_body_kwargs_for({"ml_models_dir": "/models"})
    assert kwargs["ml_models_dir"] == "/models"
    assert {k: v for k, v in kwargs.items() if k != "ml_models_dir"} == vsm_settings.vsm_body_kwargs()

"""Tests for awetrim.environment.profile_laws and its use inside Wind.

The laws are written once (NumPy/CasADi agnostic); these tests pin that the
CasADi path in ``Wind`` and the NumPy path agree, that the two amplitude
parametrisations (reference speed / friction velocity) stay consistent, and
that the legacy free-symbol contract of an unset ``Wind`` is preserved.
"""

import math

import casadi as ca
import numpy as np
import pytest

from awetrim.environment.Wind import Wind
from awetrim.environment.profile_laws import (
    ALL_MODELS,
    ANALYTIC_MODELS,
    KAPPA,
    LOG_BASED_MODELS,
    explog_law,
    friction_velocity,
    jet_law,
    log_law,
    power_law,
    speed_from_friction_velocity,
    speed_profile,
)
from awetrim.environment.wind_factory import create_wind_model

Z = np.array([6.0, 20.0, 100.0, 250.0])
U_REF, Z_REF, Z0, ALPHA = 8.0, 6.0, 0.0002, 0.1
JET = dict(jet_amplitude=3.0, jet_height=150.0, jet_width=40.0)


def _model_kwargs(model):
    if model == "uniform":
        return dict(U_ref=U_REF)
    kw = dict(U_ref=U_REF, z_ref=Z_REF, z0=Z0, alpha=ALPHA)
    if model == "jet":
        kw.update(JET)
    return kw


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
class TestKernels:
    def test_log_law_hits_reference(self):
        assert log_law(Z_REF, U_REF, Z_REF, Z0) == pytest.approx(U_REF)
        assert np.all(np.diff(log_law(Z, U_REF, Z_REF, Z0)) > 0)

    def test_power_law_hits_reference(self):
        assert power_law(Z_REF, U_REF, Z_REF, ALPHA) == pytest.approx(U_REF)

    def test_explog_is_two_log_minus_exp(self):
        got = explog_law(Z, U_REF, Z_REF, Z0, ALPHA)
        want = 2.0 * log_law(Z, U_REF, Z_REF, Z0) - power_law(Z, U_REF, Z_REF, ALPHA)
        assert got == pytest.approx(want)

    def test_jet_peak_adds_amplitude_on_top_of_log_background(self):
        zc = JET["jet_height"]
        bg = log_law(zc, U_REF, Z_REF, Z0)
        assert jet_law(zc, U_REF, Z_REF, Z0, **JET) == pytest.approx(
            bg + JET["jet_amplitude"]
        )

    def test_friction_velocity_round_trip(self):
        u_star = friction_velocity(U_REF, Z_REF, Z0)
        assert u_star == pytest.approx(KAPPA * U_REF / math.log(Z_REF / Z0))
        assert speed_from_friction_velocity(u_star, Z_REF, Z0) == pytest.approx(U_REF)
        # The friction form IS the log law.
        assert speed_from_friction_velocity(u_star, Z, Z0) == pytest.approx(
            log_law(Z, U_REF, Z_REF, Z0)
        )

    def test_kernels_accept_casadi_namespace(self):
        z = ca.MX.sym("z")
        for model in ANALYTIC_MODELS:
            expr = speed_profile(
                model, z, u_ref=U_REF, z_ref=Z_REF, z0=Z0, alpha=ALPHA, **JET, xp=ca
            )
            f = ca.Function("f", [z], [expr])
            num = speed_profile(
                model, Z, u_ref=U_REF, z_ref=Z_REF, z0=Z0, alpha=ALPHA, **JET, xp=np
            )
            num = np.broadcast_to(num, Z.shape)
            assert np.array([float(f(zi)) for zi in Z]) == pytest.approx(num)

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError, match="Unknown analytic wind model"):
            speed_profile("gaussian", 10.0, u_ref=1.0, z_ref=1.0)
        with pytest.raises(ValueError, match="Unknown wind model"):
            Wind(wind_model="gaussian")
        assert set(ALL_MODELS) == set(ANALYTIC_MODELS) | {"tabulated"}


# ---------------------------------------------------------------------------
# Wind (CasADi) == kernels (NumPy)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ANALYTIC_MODELS)
def test_wind_matches_numpy_kernel(model):
    wind = create_wind_model(model, direction_wind=0.0, **_model_kwargs(model))
    got = np.array([float(wind.speed_wind(z)) for z in Z])
    want = np.broadcast_to(
        speed_profile(model, Z, u_ref=U_REF, z_ref=Z_REF, z0=Z0, alpha=ALPHA, **JET),
        Z.shape,
    )
    assert got == pytest.approx(want)
    # profile_numeric is the NumPy evaluation of the same Wind instance.
    assert wind.profile_numeric(Z) == pytest.approx(want)


def test_profile_numeric_tabulated():
    wind = create_wind_model("tabulated", heights=[10.0, 100.0], speeds=[5.0, 15.0])
    assert wind.profile_numeric([10.0, 55.0, 100.0]) == pytest.approx([5.0, 10.0, 15.0])
    assert float(wind.speed_wind(55.0)) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Amplitude bookkeeping
# ---------------------------------------------------------------------------
class TestAmplitude:
    def test_setting_friction_then_reading_reference(self):
        wind = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0.0)
        u_star = friction_velocity(10.0, wind.height_ref, 0.1)
        wind.speed_friction = u_star
        assert float(wind.speed_wind_ref) == pytest.approx(10.0)
        assert float(wind.speed_wind_ref_value) == pytest.approx(10.0)
        assert float(wind.speed_wind(wind.height_ref)) == pytest.approx(10.0)

    def test_no_stale_state_when_height_ref_changes_after_speed(self):
        """The old implementation baked ln(height_ref/z0) into speed_friction
        at set time; the lazy derivation must follow later changes."""
        wind = Wind(wind_model="logarithmic", z0=0.03, direction_wind=0.0)
        wind.speed_wind_ref = 10.0  # at the default 6 m ...
        wind.height_ref = 100.0  # ... then moved to 100 m
        assert float(wind.speed_wind(100.0)) == pytest.approx(10.0)
        assert float(wind.speed_friction) == pytest.approx(
            friction_velocity(10.0, 100.0, 0.03)
        )

    def test_no_stale_state_when_z0_changes_after_friction(self):
        wind = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0.0)
        wind.speed_friction = 0.5
        wind.z0 = 0.01
        assert float(wind.speed_wind(50.0)) == pytest.approx(
            speed_from_friction_velocity(0.5, 50.0, 0.01)
        )

    def test_power_law_ignores_z0_for_reference_speed(self):
        wind = Wind(wind_model="power_law", z0=0.1, direction_wind=0.0, alpha=0.2)
        wind.height_ref = 100.0
        wind.speed_wind_ref = 9.0
        assert float(wind.speed_wind(100.0)) == pytest.approx(9.0)
        assert float(wind.speed_wind(50.0)) == pytest.approx(9.0 * 0.5**0.2)

    @pytest.mark.parametrize("model", LOG_BASED_MODELS)
    def test_unset_log_based_wind_exposes_free_speed_friction(self, model):
        kw = dict(jet_height=100.0, jet_width=30.0) if model == "jet" else {}
        wind = Wind(wind_model=model, z0=0.1, direction_wind=0.0, **kw)
        sym = wind.speed_friction
        assert isinstance(sym, ca.MX) and sym.is_symbolic()
        assert sym.name() == "speed_friction"
        assert wind.speed_wind_ref_value is None
        # The profile depends on that one symbol only.
        free = {s.name() for s in ca.symvar(wind.speed_wind(80.0))}
        assert free == {"speed_friction"}

    @pytest.mark.parametrize("model", ["uniform", "power_law"])
    def test_unset_other_wind_exposes_free_speed_wind_ref(self, model):
        wind = Wind(wind_model=model, direction_wind=0.0)
        sym = wind.speed_wind_ref
        assert isinstance(sym, ca.MX) and sym.is_symbolic()
        assert sym.name() == "speed_wind_ref"
        free = {s.name() for s in ca.symvar(wind.speed_wind(80.0))}
        assert free == {"speed_wind_ref"}

    def test_symbolic_reference_speed_parametrises_every_law(self):
        """residual_solver swaps speed_wind_ref for a symbol; all analytic
        laws must then depend on that symbol and evaluate to the kernel."""
        u = ca.MX.sym("speed_wind_ref")
        for model in ANALYTIC_MODELS:
            wind = create_wind_model(model, direction_wind=0.0, **_model_kwargs(model))
            wind.speed_wind_ref = u
            expr = wind.speed_wind(100.0)
            assert {s.name() for s in ca.symvar(expr)} == {"speed_wind_ref"}
            f = ca.Function("f", [u], [expr])
            want = speed_profile(
                model, 100.0, u_ref=U_REF, z_ref=Z_REF, z0=Z0, alpha=ALPHA, **JET
            )
            assert float(f(U_REF)) == pytest.approx(float(want))

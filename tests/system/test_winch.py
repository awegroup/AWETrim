"""Unit tests for awetrim.system.winch.Winch

Tests verify the nominal tether-force model and the radial algebraic equation:
- tension_curve shapes: linear / quadratic / custom_spline
- offset and slope discovery via the ``offset_winch_*`` / ``slope_winch_*`` keys
- optional softplus / softminus force-limiting smoothing
- radial_equation for the "force" and "constant" reeling strategies
- error paths for missing/invalid configuration

Per AGENTS.md @tester role:
- Test CasADi expression structure and symbolic shapes; only assert numeric
  values where the model is deterministic (no smoothing).
- One test file per source module.
"""

import casadi as ca
import numpy as np
import pytest

from awetrim.system.winch import Winch
from awetrim.utils.defaults import DEFAULT_WINCH_CONFIG


# ============================================================================
# FIXTURES
# ============================================================================


def _linear_config(**overrides):
    cfg = {
        "reeling_strategy": "force",
        "force_model": "linear",
        "max_tether_force": 5000.0,
        "min_tether_force": 500.0,
        "slope_winch_force": 1000.0,
    }
    cfg.update(overrides)
    return cfg


def _quadratic_config(**overrides):
    cfg = {
        "reeling_strategy": "force",
        "force_model": "quadratic",
        "max_tether_force": 8400.0,
        "min_tether_force": 0.0,
        "slope_winch_force": 200.0,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def linear_winch():
    return Winch(pattern_config=_linear_config())


@pytest.fixture
def quadratic_winch():
    return Winch(pattern_config=_quadratic_config())


# ============================================================================
# INITIALIZATION
# ============================================================================


class TestWinchInitialization:
    """Winch stores winch limits from config and the supplied pattern_config."""

    def test_default_config_limits_loaded(self):
        winch = Winch(pattern_config=_linear_config())
        assert winch.max_tether_length == DEFAULT_WINCH_CONFIG["max_tether_length"]
        assert winch.min_tether_length == DEFAULT_WINCH_CONFIG["min_tether_length"]
        assert winch.max_speed == DEFAULT_WINCH_CONFIG["max_speed"]
        assert winch.min_speed == DEFAULT_WINCH_CONFIG["min_speed"]
        assert winch.max_acceleration == DEFAULT_WINCH_CONFIG["max_acceleration"]
        assert winch.min_acceleration == DEFAULT_WINCH_CONFIG["min_acceleration"]

    def test_custom_config_overrides_limits(self):
        custom = dict(DEFAULT_WINCH_CONFIG, max_speed=12.0, min_speed=-9.0)
        winch = Winch(pattern_config=_linear_config(), config=custom)
        assert winch.max_speed == 12.0
        assert winch.min_speed == -9.0

    def test_pattern_config_stored(self):
        cfg = _linear_config()
        winch = Winch(pattern_config=cfg)
        assert winch.pattern_config is cfg


# ============================================================================
# TENSION CURVE — LINEAR / QUADRATIC SHAPES
# ============================================================================


class TestTensionCurveLinear:
    """Linear model: T = slope * (v_r - offset)."""

    def test_linear_value_without_offset(self, linear_winch):
        # offset defaults to 0 when no offset_winch_* key is present.
        assert float(linear_winch.tension_curve(5.0)) == pytest.approx(5000.0)

    def test_linear_value_with_offset(self):
        winch = Winch(pattern_config=_linear_config(offset_winch_force=2.0))
        # 1000 * (5 - 2) = 3000
        assert float(winch.tension_curve(5.0)) == pytest.approx(3000.0)

    def test_linear_passes_through_offset(self):
        winch = Winch(pattern_config=_linear_config(offset_winch_force=1.5))
        assert float(winch.tension_curve(1.5)) == pytest.approx(0.0)

    def test_slope_discovered_by_prefix_not_exact_name(self):
        # Any "slope_winch_*" suffix must be picked up.
        cfg = _linear_config()
        del cfg["slope_winch_force"]
        cfg["slope_winch_custom_name"] = 750.0
        winch = Winch(pattern_config=cfg)
        assert float(winch.tension_curve(4.0)) == pytest.approx(3000.0)


class TestTensionCurveQuadratic:
    """Quadratic model: T = slope * (v_r - offset)^2."""

    def test_quadratic_value_without_offset(self, quadratic_winch):
        # 200 * 3^2 = 1800
        assert float(quadratic_winch.tension_curve(3.0)) == pytest.approx(1800.0)

    def test_quadratic_is_symmetric_about_offset(self):
        winch = Winch(pattern_config=_quadratic_config(offset_winch_force=1.0))
        below = float(winch.tension_curve(1.0 - 2.0))
        above = float(winch.tension_curve(1.0 + 2.0))
        assert below == pytest.approx(above)

    def test_quadratic_non_negative_for_positive_slope(self, quadratic_winch):
        for v in (-3.0, 0.0, 2.5, 6.0):
            assert float(quadratic_winch.tension_curve(v)) >= 0.0


# ============================================================================
# TENSION CURVE — SYMBOLIC STRUCTURE
# ============================================================================


class TestTensionCurveDepowerOffset:
    """Depower-dependent offset: offset(l_dp) = offset0 + gain*(l_dp - ref)."""

    def test_no_gain_key_ignores_depower(self):
        # Backward compatible: without the gain key the depower arg has no effect.
        winch = Winch(pattern_config=_linear_config(offset_winch_force=2.0))
        base = float(winch.tension_curve(5.0))
        assert float(winch.tension_curve(5.0, input_depower=2.1)) == pytest.approx(
            base
        )

    def test_offset_unchanged_at_reference_depower(self):
        winch = Winch(
            pattern_config=_linear_config(
                offset_winch_force=2.0,
                winch_offset_depower_gain=-3.0,
                winch_depower_ref=1.7,
            )
        )
        # At l_dp == ref the shift is zero: 1000 * (5 - 2) = 3000.
        assert float(winch.tension_curve(5.0, input_depower=1.7)) == pytest.approx(
            3000.0
        )

    def test_negative_gain_shifts_offset_down_when_depowered(self):
        winch = Winch(
            pattern_config=_linear_config(
                offset_winch_force=2.0,
                winch_offset_depower_gain=-3.0,
                winch_depower_ref=1.7,
            )
        )
        # offset(2.1) = 2.0 + (-3.0)*(2.1 - 1.7) = 0.8 -> 1000 * (5 - 0.8) = 4200.
        assert float(winch.tension_curve(5.0, input_depower=2.1)) == pytest.approx(
            4200.0
        )

    def test_gain_key_not_mistaken_for_base_offset(self):
        # The gain key must not be picked up by the offset_winch_* discovery loop.
        winch = Winch(
            pattern_config=_linear_config(
                winch_offset_depower_gain=-3.0, winch_depower_ref=1.7
            )
        )
        # No offset_winch_* key -> base offset is 0; at ref depower T = 1000*5.
        assert float(winch.tension_curve(5.0, input_depower=1.7)) == pytest.approx(
            5000.0
        )

    def test_symbolic_depower_returns_mx(self):
        winch = Winch(
            pattern_config=_linear_config(
                winch_offset_depower_gain=-3.0, winch_depower_ref=1.7
            )
        )
        l_dp = ca.MX.sym("l_dp")
        expr = winch.tension_curve(5.0, input_depower=l_dp)
        assert isinstance(expr, ca.MX)
        assert expr.shape == (1, 1)


class TestTensionCurveSymbolic:
    """Symbolic v_r produces a CasADi MX scalar expression."""

    def test_symbolic_input_returns_mx(self, linear_winch):
        vr = ca.MX.sym("vr")
        expr = linear_winch.tension_curve(vr)
        assert isinstance(expr, ca.MX)
        assert expr.shape == (1, 1)

    def test_symbolic_matches_numeric_via_function(self, quadratic_winch):
        vr = ca.MX.sym("vr")
        f = ca.Function("T", [vr], [quadratic_winch.tension_curve(vr)])
        assert float(f(3.0)) == pytest.approx(
            float(quadratic_winch.tension_curve(3.0))
        )


# ============================================================================
# TENSION CURVE — SMOOTHING (softplus / softminus)
# ============================================================================


class TestTensionCurveSmoothing:
    """Optional softplus/softminus limiters bend the curve toward the caps.

    Smoothing magnitude is an approximation, so only the *direction* of the
    effect is asserted (softplus lowers above-cap forces; softminus raises
    below-floor forces).
    """

    def test_softplus_lowers_above_cap_force(self):
        unsmoothed = Winch(pattern_config=_linear_config())
        smoothed = Winch(
            pattern_config=_linear_config(softplus=True, softplus_beta=1e-4)
        )
        # v=10 -> unsmoothed 10000 N, well above the 5000 N cap.
        raw = float(unsmoothed.tension_curve(10.0))
        cap_limited = float(smoothed.tension_curve(10.0))
        assert np.isfinite(cap_limited)
        assert cap_limited < raw

    def test_softminus_raises_below_floor_force(self):
        unsmoothed = Winch(pattern_config=_linear_config())
        smoothed = Winch(
            pattern_config=_linear_config(softminus=True, softminus_beta=1e-4)
        )
        # v=-4 -> unsmoothed -4000 N, below the 500 N floor.
        raw = float(unsmoothed.tension_curve(-4.0))
        floor_limited = float(smoothed.tension_curve(-4.0))
        assert np.isfinite(floor_limited)
        assert floor_limited > raw

    def test_smoothing_preserves_symbolic_type(self):
        winch = Winch(
            pattern_config=_linear_config(
                softplus=True, softplus_beta=1e-4, softminus=True, softminus_beta=1e-4
            )
        )
        vr = ca.MX.sym("vr")
        assert isinstance(winch.tension_curve(vr), ca.MX)


# ============================================================================
# TENSION CURVE — CUSTOM SPLINE
# ============================================================================


class TestTensionCurveCustomSpline:
    """custom_spline builds a CasADi bspline interpolant from knots/coeffs."""

    def _spline_config(self):
        return {
            "reeling_strategy": "force",
            "force_model": "custom_spline",
            "v_knots": [0.0, 1.0, 2.0, 3.0, 4.0],
            "C_fitted": [0.0, 1000.0, 2000.0, 3000.0, 4000.0],
        }

    def test_custom_spline_evaluates_finite(self):
        winch = Winch(pattern_config=self._spline_config())
        assert np.isfinite(float(winch.tension_curve(2.0)))

    def test_custom_spline_allows_missing_max_force(self):
        # custom_spline is exempt from the max_tether_force requirement.
        cfg = self._spline_config()
        assert "max_tether_force" not in cfg
        winch = Winch(pattern_config=cfg)
        assert np.isfinite(float(winch.tension_curve(1.0)))

    def test_custom_spline_requires_knots_and_coeffs(self):
        cfg = self._spline_config()
        del cfg["v_knots"]
        winch = Winch(pattern_config=cfg)
        with pytest.raises(ValueError, match="v_knots"):
            winch.tension_curve(2.0)


# ============================================================================
# TENSION CURVE — ERROR PATHS
# ============================================================================


class TestTensionCurveErrors:
    def test_missing_max_tether_force_raises(self):
        cfg = _linear_config()
        del cfg["max_tether_force"]
        winch = Winch(pattern_config=cfg)
        with pytest.raises(ValueError, match="max_tether_force"):
            winch.tension_curve(3.0)

    def test_missing_slope_raises(self):
        cfg = _linear_config()
        del cfg["slope_winch_force"]
        winch = Winch(pattern_config=cfg)
        with pytest.raises(ValueError, match="slope_winch"):
            winch.tension_curve(3.0)

    def test_unknown_force_model_raises(self):
        winch = Winch(pattern_config=_linear_config(force_model="cubic"))
        with pytest.raises(ValueError, match="Unknown force_model"):
            winch.tension_curve(3.0)


# ============================================================================
# RADIAL EQUATION
# ============================================================================


class TestRadialEquationForce:
    """Force reeling: residual = F_t - tension_curve(v_r)."""

    def test_force_residual_value(self, linear_winch):
        # tension_curve(3) = 3000; residual = 4000 - 3000 = 1000
        residual = linear_winch.radial_equation(
            speed_radial=3.0, tension_tether_ground=4000.0
        )
        assert float(residual) == pytest.approx(1000.0)

    def test_force_residual_zero_on_curve(self, linear_winch):
        tension = float(linear_winch.tension_curve(2.0))
        residual = linear_winch.radial_equation(
            speed_radial=2.0, tension_tether_ground=tension
        )
        assert float(residual) == pytest.approx(0.0)

    def test_force_symbolic_residual_is_mx(self, linear_winch):
        vr = ca.MX.sym("vr")
        ft = ca.MX.sym("ft")
        residual = linear_winch.radial_equation(
            speed_radial=vr, tension_tether_ground=ft
        )
        assert isinstance(residual, ca.MX)
        assert residual.shape == (1, 1)

    def test_force_requires_speed_radial(self, linear_winch):
        with pytest.raises(ValueError, match="speed_radial"):
            linear_winch.radial_equation(tension_tether_ground=4000.0)

    def test_force_requires_tension(self, linear_winch):
        with pytest.raises(ValueError, match="tension_tether_ground"):
            linear_winch.radial_equation(speed_radial=3.0)


class TestRadialEquationConstant:
    """Constant reeling: residual = v_r - reeling_speed."""

    def test_constant_residual_value(self):
        winch = Winch(
            pattern_config={"reeling_strategy": "constant", "reeling_speed": 2.0}
        )
        residual = winch.radial_equation(speed_radial=5.0)
        assert float(residual) == pytest.approx(3.0)

    def test_constant_residual_zero_at_setpoint(self):
        winch = Winch(
            pattern_config={"reeling_strategy": "constant", "reeling_speed": 2.0}
        )
        assert float(winch.radial_equation(speed_radial=2.0)) == pytest.approx(0.0)


class TestRadialEquationErrors:
    def test_unknown_reeling_strategy_raises(self):
        winch = Winch(pattern_config={"reeling_strategy": "magic"})
        with pytest.raises(ValueError, match="Unknown reeling_strategy"):
            winch.radial_equation(speed_radial=3.0, tension_tether_ground=4000.0)

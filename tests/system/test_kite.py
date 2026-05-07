"""
Comprehensive tests for Kite and RigidLumpedTether equations.

Tests verify equations from:
- Cayon, van Deursen, Schmehl (2026) WES 11, 1097 (ROM paper)
- Cayon, Gaunaa, Schmehl (2023) Energies 16, 3061 (Aerostructural paper)

These tests use the V3 kite aerodynamic model loaded from YAML configuration,
demonstrating the complete framework with real aerodynamic coefficients.
"""

import casadi as ca
import numpy as np
import pytest
import yaml

from awetrim.system.factory import load_aero_input_from_system_config
from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG

from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.environment.Wind import Wind


def load_v3_kite_config():
    """Load V3 kite configuration from YAML."""
    with open(LEI_V3_SYSTEM_CONFIG, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def v3_kite_config():
    """Fixture providing V3 kite configuration."""
    return load_v3_kite_config()


def _extract_from_config(cfg: dict) -> tuple:
    """Extract kite parameters from awesIO or legacy system config format."""
    if "components" in cfg:
        kite = cfg["components"].get("kite", cfg["components"])
        wing_struct = kite["wing"]["structure"]
        cs_struct = kite.get("control_system", {}).get("structure", {})
        return (
            load_aero_input_from_system_config(cfg, config_path=LEI_V3_SYSTEM_CONFIG),
            wing_struct.get("mass", 15),
            wing_struct.get("projected_surface_area", 19.75),
            cs_struct.get("mass", 0.0),
        )
    wing = cfg.get("wing", {})
    return (
        wing.get("aerodynamics", {}),
        wing.get("mass", 15),
        wing.get("area", 19.75),
        cfg.get("kcu", {}).get("mass", 0.0),
    )


@pytest.fixture
def v3_kite(v3_kite_config):
    """Fixture providing initialized V3 Kite instance."""
    aero_input, mass_wing, area_wing, _ = _extract_from_config(v3_kite_config)
    return Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=8.4,
        steering_control="asymmetric",
    )


@pytest.fixture
def v3_tether():
    """Fixture providing V3 RigidLumpedTether."""
    return RigidLumpedTether(diameter=0.01)


@pytest.fixture
def v3_wind():
    """Fixture providing wind model for V3."""
    return Wind(wind_model="logarithmic", z0=0.1, direction_wind=0)


def assert_close_vector(actual, expected, tol=1e-6, name=""):
    """Helper to compare vectors or scalars."""
    if isinstance(actual, (int, float)):
        actual_arr = np.array([actual])
    else:
        actual_arr = np.array(actual).flatten()

    if isinstance(expected, (int, float)):
        expected_arr = np.array([expected])
    else:
        expected_arr = np.array(expected).flatten()

    assert np.allclose(
        actual_arr, expected_arr, rtol=tol, atol=tol
    ), f"{name}: Expected {expected_arr}, got {actual_arr}"


# ============================================================================
# AERODYNAMIC MODEL TESTS (V3 KITE)
# ============================================================================


class TestV3AerodynamicModel:
    """Test aerodynamic model with V3 kite coefficients."""

    def test_v3_aerodynamic_config_loads(self, v3_kite_config):
        """Verify V3 kite configuration loads with expected aerodynamic coefficients."""
        aero, _, _, _ = _extract_from_config(v3_kite_config)

        # Verify model type
        assert aero["model"] == "coeffs", "V3 should use coefficient-based model"

        # Verify key parameters exist
        assert "CD0" in aero["params"], "CD0 should be defined"
        assert "CL0" in aero["params"], "CL0 should be defined"
        assert aero["params"]["CD0"] == 0.1130532, "CD0 coefficient mismatch"
        assert aero["params"]["CL0"] == 0.04671295, "CL0 coefficient mismatch"

        # Verify aerodynamic coefficient terms
        assert "CL" in aero["coefficients"], "CL coefficients should be defined"
        assert "CD" in aero["coefficients"], "CD coefficients should be defined"

        # Verify polynomial order (CL and CD are both functions of alpha)
        cl_terms = aero["coefficients"]["CL"]
        assert any(
            term["var"] == "alpha" and term["power"] == 1 for term in cl_terms
        ), "CL should have linear alpha term"
        assert any(
            term["var"] == "alpha" and term["power"] == 2 for term in cl_terms
        ), "CL should have quadratic alpha term"

    def test_v3_kite_initialization(self, v3_kite):
        """Verify V3 kite initializes with correct parameters."""
        assert v3_kite.mass_wing == pytest.approx(11.4746), "V3 wing mass from PSS model"
        assert v3_kite.area_wing == 19.75, "V3 wing area should be 19.75 m²"
        assert v3_kite.mass_kcu == 8.4, "V3 KCU mass should be 8.4 kg"
        assert v3_kite.rho == 1.225, "Air density should be standard"


# ============================================================================
# GRAVITY FORCE TESTS (V3 KITE)
# ============================================================================


class TestGravityForces:
    """Test gravity force equations."""

    def test_gravity_wing_formula_at_horizontal(self, v3_kite):
        """Verify gravity_wing formula: F_g = -m*g*[cos(elev)*cos(chi), ...]."""
        # Create symbolic versions with explicit inputs
        elev = ca.MX.sym("elev")
        chi = ca.MX.sym("chi")
        mass = v3_kite.mass_wing
        g = v3_kite.g

        # Manual formula from paper
        expected_gravity = (
            -mass
            * g
            * ca.vertcat(
                ca.cos(elev) * ca.cos(chi), ca.cos(elev) * ca.sin(chi), ca.sin(elev)
            )
        )

        # Set kite properties and create function
        v3_kite.angle_elevation = elev
        v3_kite.angle_course = chi
        impl_gravity = v3_kite.force_gravity_wing

        # Compare via residual
        residual_fn = ca.Function(
            "gravity_residual", [elev, chi], [impl_gravity - expected_gravity]
        )

        # Test at horizontal (elev=0, chi=0)
        result_h = np.array(residual_fn(0.0, 0.0).full().flatten())
        assert_close_vector(result_h, 0.0, tol=1e-10, name="Gravity at horizontal")

        # Test at angle (typical kite elevation ~0.4 rad, course ~0.8 rad)
        result_a = np.array(residual_fn(0.4, 0.8).full().flatten())
        assert_close_vector(
            result_a, 0.0, tol=1e-10, name="Gravity at typical flight angle"
        )

    def test_gravity_wing_magnitude_is_mg(self, v3_kite):
        """Verify gravity magnitude: |F_g| = m*g at all elevations."""
        # At any position, gravity magnitude should be m*g
        elev = ca.MX.sym("elev")
        chi = ca.MX.sym("chi")
        v3_kite.angle_elevation = elev
        v3_kite.angle_course = chi

        gravity_force = v3_kite.force_gravity_wing
        gravity_mag = ca.norm_2(gravity_force)

        expected_mag = v3_kite.mass_wing * v3_kite.g

        mag_fn = ca.Function("mag", [elev, chi], [gravity_mag - expected_mag])

        # Test at multiple elevations (0, 0.2, 0.4, 0.6 rad)
        for elev_test in [0.0, 0.2, 0.4, 0.6]:
            result = float(mag_fn(elev_test, 0.5))
            assert_close_vector(
                result, 0.0, tol=1e-10, name=f"Gravity magnitude at elev={elev_test}"
            )


# ============================================================================
# TETHER FORCE TESTS (V3 KITE)
# ============================================================================


class TestRigidLumpedTetherForces:
    """Test RigidLumpedTether force assembly equations."""

    def test_tether_initialized_with_v3_params(self, v3_tether):
        """Verify tether initializes with expected V3 parameters."""
        assert v3_tether.diameter_tether == 0.01, "V3 tether diameter should be 0.01 m"
        assert (
            v3_tether.drag_coefficient_tether == 1.1
        ), "Tether drag coefficient should be 1.1"
        assert hasattr(
            v3_tether, "tension_tether_ground"
        ), "Should have tension_tether_ground"

    def test_drag_formula_matches_van_der_vlugt(self, v3_tether):
        """Verify drag_tether_at_kite matches Van Der Vlugt et al. (2019) eq. 14.

        D = 0.125 * Cd * r * d_tether * rho * v_apparent * ||v_apparent||
        """
        # Create symbolic parameters
        v_a = ca.vertcat(ca.MX.sym("v_ax"), ca.MX.sym("v_ay"), ca.MX.sym("v_az"))
        r = ca.MX.sym("r")
        d = ca.MX.sym("d")
        rho_sym = ca.MX.sym("rho")

        # Set symbolic parameters (these are writable attributes, not properties)
        v3_tether.velocity_apparent_wind = v_a
        v3_tether.distance_radial = r
        v3_tether.diameter_tether = d
        v3_tether.rho = rho_sym

        # Expected formula from paper: 0.125 * Cd * r * d * rho * v_a * ||v_a||
        Cd = v3_tether.drag_coefficient_tether
        v_a_norm = ca.norm_2(v_a)
        expected_drag = 0.125 * Cd * r * d * rho_sym * v_a * v_a_norm

        # Get implementation
        drag_impl = v3_tether.drag_tether_at_kite

        # Create residual function
        residual_fn = ca.Function(
            "drag_res",
            [v_a[0], v_a[1], v_a[2], r, d, rho_sym],
            [ca.norm_2(drag_impl - expected_drag)],
        )

        # Test with V3 typical values: v_a~8 m/s, r~200m, d~0.01m, rho~1.225 kg/m³
        result = float(residual_fn(8.0, 0.0, 0.0, 200.0, 0.01, 1.225))
        assert_close_vector(
            result, 0.0, tol=1e-10, name="Drag formula (Van Der Vlugt eq. 14)"
        )


# ============================================================================
# KINETIC ACCELERATION TESTS (V3 KITE)
# ============================================================================


class TestKineticAccelerations:
    """Test acceleration equation structures from kinematics."""

    def test_acceleration_inertial_formula(self, v3_kite):
        """Verify acceleration_inertial formula: [-vt*vr/r, vt*sin(chi)*tan(beta)/r, vt²/r]."""
        # Create symbolic variables with typical V3 values
        vt = ca.MX.sym("vt")
        vr = ca.MX.sym("vr")
        r = ca.MX.sym("r")
        chi = ca.MX.sym("chi")
        beta = ca.MX.sym("beta")

        v3_kite.speed_tangential = vt
        v3_kite.speed_radial = vr
        v3_kite.distance_radial = r
        v3_kite.angle_course = chi
        v3_kite.angle_elevation = beta

        # Expected formula from paper
        expected_accel = ca.vertcat(
            -vt * vr / r, vt * ca.sin(chi) * ca.tan(beta) / r, vt**2 / r
        )

        accel_impl = v3_kite.acceleration_inertial

        residual_fn = ca.Function(
            "accel_residual", [vt, vr, r, chi, beta], [accel_impl - expected_accel]
        )

        # Test with V3 typical reel-out conditions
        # vt~30 m/s, vr~2 m/s, r~200m, chi~0.5 rad, beta~0.4 rad
        result = np.array(residual_fn(30.0, 2.0, 200.0, 0.5, 0.4).full().flatten())
        assert_close_vector(
            result, 0.0, tol=1e-10, name="Acceleration inertial formula"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

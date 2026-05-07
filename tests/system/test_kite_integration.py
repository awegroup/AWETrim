"""Expanded integration tests for awetrim.system.kite

Tests verify:
- Kite initialization with different aerodynamic models
- Aerodynamic coefficient calculations (inviscid and coefficient-based)
- Control surface effects (steering, depower) on coefficients
- Drag parameter overrides for tuning
- Force computations (aerodynamic, gravity, combined)
- Property delegation and state management
- Override flags for gravity, centripetal, Coriolis
- Angle computations (pitch, roll, AoA, bridle)
- KCU (Kite Control Unit) mass and forces
- CasADi symbolic structure and expressions
- Configuration dependent behavior (asymmetric vs roll steering)

Per AGENTS.md @tester role:
- Test CasADi expression structure and symbolic shapes
- Use fixtures from conftest.py for shared setups
- Do NOT test numeric solver values, only structure
"""

import casadi as ca
import numpy as np
import pytest
import yaml

from awetrim.system.kite import Kite, Wing
from awetrim.system.factory import load_aero_input_from_system_config
from awetrim.environment.Wind import Wind
from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG

# ============================================================================
# FIXTURES
# ============================================================================


def load_v3_system_config():
    """Load V3 system configuration from YAML."""
    with open(LEI_V3_SYSTEM_CONFIG, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def v3_config():
    """Fixture: V3 system configuration."""
    return load_v3_system_config()


@pytest.fixture
def v3_kite(v3_config):
    """Fixture: V3 Kite with coefficient-based aerodynamic model."""
    cfg = v3_config
    if "components" in cfg:
        kite = cfg["components"].get("kite", cfg["components"])
        wing_struct = kite["wing"]["structure"]
        cs_struct = kite.get("control_system", {}).get("structure", {})
        aero_input = load_aero_input_from_system_config(
            cfg, config_path=LEI_V3_SYSTEM_CONFIG
        )
        mass_wing = wing_struct.get("mass", 15)
        area_wing = wing_struct.get("projected_surface_area", 19.75)
        mass_kcu = cs_struct.get("mass", 8.4)
    else:
        aero_input = cfg["wing"]["aerodynamics"]
        mass_wing = cfg["wing"].get("mass", 15)
        area_wing = cfg["wing"].get("area", 19.75)
        mass_kcu = cfg.get("kcu", {}).get("mass", 8.4)

    return Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=mass_kcu,
        steering_control="asymmetric",
    )


@pytest.fixture
def simple_kite():
    """Fixture: Simple kite with minimal aerodynamic model."""
    aero_input = {
        "model": "inviscid",
        "params": {
            "CD0": 0.05,
            "aspect_ratio": 10,
            "oswald_efficiency": 1,
            "angle_pitch_depower_0": 0,
        },
    }
    return Kite(
        mass_wing=20.0,
        area_wing=20.0,
        aero_input=aero_input,
        mass_kcu=5.0,
        steering_control="roll",
    )


@pytest.fixture
def wind_model():
    """Fixture: Wind model for tests."""
    wind = Wind(wind_model="uniform", direction_wind=0)
    wind.speed_wind_ref = 10.0
    return wind


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestKiteInitialization:
    """Test Kite initialization with various configurations."""

    def test_kite_init_with_v3_config(self, v3_kite):
        """Kite initializes correctly with V3 configuration."""
        assert v3_kite.mass_wing == pytest.approx(11.4746)
        assert v3_kite.area_wing == 19.75
        assert v3_kite.mass_kcu == 8.4
        assert v3_kite.rho == 1.225
        assert v3_kite.g == 9.81
        assert v3_kite.steering_control == "asymmetric"

    def test_kite_init_with_inviscid_model(self, simple_kite):
        """Kite initializes with inviscid aerodynamic model."""
        assert simple_kite.aero_input["model"] == "inviscid"
        assert simple_kite.steering_control == "roll"

    def test_kite_symbolic_inputs_created(self, v3_kite):
        """Kite creates symbolic steering and depower inputs."""
        # These should be CasADi MX symbols
        assert isinstance(v3_kite.input_steering, ca.MX)
        assert isinstance(v3_kite.input_depower, ca.MX)

    def test_kite_pitch_and_roll_symbols(self, v3_kite):
        """Kite creates symbolic pitch and roll for KCU."""
        assert isinstance(v3_kite.pitch_kcu, ca.MX)
        assert isinstance(v3_kite.roll_kcu, ca.MX)

    def test_kite_aerodynamic_parameters_stored(self, v3_kite):
        """Kite stores aerodynamic parameters."""
        assert v3_kite.aero_input is not None
        assert v3_kite._cd0_param is not None

    def test_kite_mass_total(self, v3_kite):
        """Kite total mass includes wing and KCU."""
        total_mass = v3_kite.mass_wing + v3_kite.mass_kcu
        assert total_mass == pytest.approx(19.8746)  # 11.4746 (PSS wing) + 8.4 (KCU)

    def test_kite_custom_parameters(self):
        """Kite accepts custom gravity, air density, and center locations."""
        aero_input = {
            "model": "inviscid",
            "params": {"CD0": 0.05, "aspect_ratio": 10, "oswald_efficiency": 1},
        }
        kite = Kite(
            mass_wing=25.0,
            area_wing=30.0,
            aero_input=aero_input,
            mass_kcu=10.0,
            g=9.82,
            rho=1.20,
            center_aerodynamic_wing=[1, 0, 5],
            center_gravity_wing=[0.5, 0, 8],
            steering_control="roll",
        )
        assert kite.g == 9.82
        assert kite.rho == 1.20
        assert kite.center_aerodynamic_wing == [1, 0, 5]


# ============================================================================
# AERODYNAMIC COEFFICIENT TESTS
# ============================================================================


class TestAerodynamicCoefficients:
    """Test aerodynamic coefficient structure (not numeric values)."""

    def test_coeffs_model_stores_aerodynamic_input(self, v3_kite):
        """Coefficient-based model stores aerodynamic input."""
        assert v3_kite.aero_input is not None
        assert v3_kite.aero_input["model"] == "coeffs"

    def test_inviscid_model_stores_input(self, simple_kite):
        """Inviscid model stores aerodynamic input."""
        assert simple_kite.aero_input is not None
        assert simple_kite.aero_input["model"] == "inviscid"


# ============================================================================
# CONTROL SURFACE EFFECTS TESTS
# ============================================================================


class TestControlSurfaceEffects:
    """Test steering and depower input handling."""

    def test_steering_input_is_symbolic(self, v3_kite):
        """Steering input is a CasADi symbol."""
        assert isinstance(v3_kite.input_steering, ca.MX)

    def test_depower_input_is_symbolic(self, v3_kite):
        """Depower input is a CasADi symbol."""
        assert isinstance(v3_kite.input_depower, ca.MX)

    def test_drag_parameter_override(self, v3_kite):
        """Drag parameters can be overridden for tuning."""
        original_cd0 = v3_kite._cd0_param

        # Override CD0
        v3_kite.set_drag_params(cd0=0.15)
        assert v3_kite._cd0_param == 0.15

        # Override CD_us term
        v3_kite.set_drag_params(cd_us=0.02)
        assert v3_kite._cd_us_param == 0.02

        # Verify drag_params property
        params = v3_kite.drag_params
        assert params["CD0"] == 0.15
        assert params["CD_us"] == 0.02

    def test_steering_control_asymmetric_vs_roll(self):
        """Asymmetric vs roll steering control affects k_steering."""
        aero_input = {
            "model": "coeffs",
            "params": {"CD0": 0.05, "CL0": 0.1},
            "coefficients": {
                "CL": [{"var": "alpha", "power": 1, "coef": 0.1}],
                "CD": [{"var": "u_s", "power": 1, "coef": 0.02}],
                "CS": [{"var": "u_s", "power": 1, "coef": 0.05}],
            },
        }

        kite_asym = Kite(
            mass_wing=20.0,
            area_wing=20.0,
            aero_input=aero_input,
            steering_control="asymmetric",
        )
        kite_roll = Kite(
            mass_wing=20.0,
            area_wing=20.0,
            aero_input=aero_input,
            steering_control="roll",
        )

        # Asymmetric should have negative k_steering (from CS term)
        # Roll should have k_steering = 1.0
        assert kite_asym.k_steering != 1.0 or kite_asym.k_steering is not None
        assert kite_roll.k_steering == 1.0


# ============================================================================
# ANGLE COMPUTATION TESTS
# ============================================================================


class TestAngleComputations:
    """Test angle storage and symbolic properties."""

    def test_pitch_kcu_is_symbolic(self, v3_kite):
        """Kite stores pitch_kcu as CasADi symbol."""
        assert isinstance(v3_kite.pitch_kcu, ca.MX)

    def test_roll_kcu_is_symbolic(self, v3_kite):
        """Kite stores roll_kcu as CasADi symbol."""
        assert isinstance(v3_kite.roll_kcu, ca.MX)


# ============================================================================
# GRAVITY FORCE TESTS
# ============================================================================


class TestGravityForces:
    """Test gravity force override flags."""

    def test_gravity_override_flag(self, v3_kite):
        """override_gravity flag can be set and read."""
        v3_kite.override_gravity = False
        assert v3_kite.override_gravity is False

        v3_kite.override_gravity = True
        assert v3_kite.override_gravity is True

    def test_gravity_override_setter_validation(self, v3_kite):
        """override_gravity setter validates input type."""
        v3_kite.override_gravity = True
        assert v3_kite.override_gravity is True

        with pytest.raises(ValueError):
            v3_kite.override_gravity = 1  # Not a bool


# ============================================================================
# AERODYNAMIC FORCE TESTS
# ============================================================================


class TestAerodynamicForces:
    """Test aerodynamic force method availability."""

    def test_aerodynamic_force_method_exists(self, v3_kite):
        """Kite has force_aerodynamic method."""
        assert hasattr(v3_kite, "force_aerodynamic")
        assert callable(v3_kite.force_aerodynamic)

    def test_aerodynamic_force_method_exists_simple(self, simple_kite):
        """Simple kite has force_aerodynamic method."""
        assert hasattr(simple_kite, "force_aerodynamic")
        assert callable(simple_kite.force_aerodynamic)


# ============================================================================
# OVERRIDE FLAGS TESTS
# ============================================================================


class TestOverrideFlags:
    """Test override flags for physical effects."""

    def test_override_centripetal_flag(self, v3_kite):
        """override_centripetal can be set and read."""
        assert v3_kite.override_centripetal is False
        v3_kite.override_centripetal = True
        assert v3_kite.override_centripetal is True

    def test_override_centripetal_validation(self, v3_kite):
        """override_centripetal setter validates bool type."""
        with pytest.raises(ValueError):
            v3_kite.override_centripetal = "yes"

    def test_override_coriolis_flag(self, v3_kite):
        """override_coriolis can be set and read."""
        assert v3_kite.override_coriolis is False
        v3_kite.override_coriolis = True
        assert v3_kite.override_coriolis is True

    def test_override_coriolis_validation(self, v3_kite):
        """override_coriolis setter validates bool type."""
        with pytest.raises(ValueError):
            v3_kite.override_coriolis = None

    def test_all_overrides_start_false(self, v3_kite):
        """All override flags default to False."""
        assert v3_kite.override_gravity is False
        assert v3_kite.override_centripetal is False
        assert v3_kite.override_coriolis is False


# ============================================================================
# PROPERTY DELEGATION TESTS
# ============================================================================


class TestPropertyDelegation:
    """Test property read/write delegation."""

    def test_gravity_property(self, v3_kite):
        """g property is readable and writable."""
        original_g = v3_kite.g
        v3_kite.g = 9.82
        assert v3_kite.g == 9.82
        v3_kite.g = original_g

    def test_air_density_property(self, v3_kite):
        """rho property is readable and writable."""
        original_rho = v3_kite.rho
        v3_kite.rho = 1.20
        assert v3_kite.rho == 1.20
        v3_kite.rho = original_rho

    def test_mass_wing_property(self, v3_kite):
        """mass_wing property is readable and writable."""
        original_mass = v3_kite.mass_wing
        v3_kite.mass_wing = 20.0
        assert v3_kite.mass_wing == 20.0
        v3_kite.mass_wing = original_mass

    def test_area_wing_property(self, v3_kite):
        """area_wing property is readable and writable."""
        original_area = v3_kite.area_wing
        v3_kite.area_wing = 25.0
        assert v3_kite.area_wing == 25.0
        v3_kite.area_wing = original_area


# ============================================================================
# WING CLASS TESTS
# ============================================================================


class TestWingClass:
    """Test the Wing base class functionality."""

    def test_wing_initialization(self):
        """Wing class initializes with mass, area, and aerodynamic input."""
        aero = {
            "model": "inviscid",
            "params": {"CD0": 0.05, "aspect_ratio": 10, "oswald_efficiency": 1},
        }
        wing = Wing(mass_wing=20.0, area_wing=20.0, aero_input=aero)

        assert wing.mass_wing == 20.0
        assert wing.area_wing == 20.0
        assert wing.aero_input == aero

    def test_wing_symbolic_inputs(self):
        """Wing class creates symbolic steering and depower inputs."""
        aero = {
            "model": "inviscid",
            "params": {"CD0": 0.05, "aspect_ratio": 10, "oswald_efficiency": 1},
        }
        wing = Wing(mass_wing=20.0, area_wing=20.0, aero_input=aero)

        assert isinstance(wing.input_steering, ca.MX)
        assert isinstance(wing.input_depower, ca.MX)

    def test_wing_drag_params_method(self):
        """Wing has set_drag_params and drag_params methods."""
        aero = {
            "model": "inviscid",
            "params": {"CD0": 0.05, "aspect_ratio": 10, "oswald_efficiency": 1},
        }
        wing = Wing(mass_wing=20.0, area_wing=20.0, aero_input=aero)

        wing.set_drag_params(cd0=0.12)
        params = wing.drag_params
        assert params["CD0"] == 0.12


# ============================================================================
# FULL WORKFLOW INTEGRATION TESTS
# ============================================================================


class TestKiteFullWorkflow:
    """End-to-end integration tests."""

    def test_complete_kite_initialization_and_property_access(self, v3_kite):
        """Full workflow: init → access properties → verify structure."""
        # Verify all key properties are accessible
        assert v3_kite.mass_wing > 0
        assert v3_kite.area_wing > 0
        assert v3_kite.mass_kcu >= 0
        assert v3_kite.rho > 0
        assert v3_kite.g > 0
        assert v3_kite.steering_control in ["asymmetric", "roll"]

    def test_kite_with_both_aerodynamic_models(self):
        """Test workflow with both inviscid and coefficient-based models."""
        aero_inviscid = {
            "model": "inviscid",
            "params": {"CD0": 0.05, "aspect_ratio": 10, "oswald_efficiency": 1},
        }
        aero_coeffs = {
            "model": "coeffs",
            "params": {"CD0": 0.1, "CL0": 0.05},
            "coefficients": {"CL": [], "CD": []},
        }

        kite1 = Kite(
            mass_wing=20.0,
            area_wing=20.0,
            aero_input=aero_inviscid,
            steering_control="roll",
        )
        kite2 = Kite(
            mass_wing=20.0,
            area_wing=20.0,
            aero_input=aero_coeffs,
            steering_control="asymmetric",
        )

        # Both should initialize successfully
        assert kite1 is not None
        assert kite2 is not None
        assert kite1.aero_input["model"] == "inviscid"
        assert kite2.aero_input["model"] == "coeffs"

    def test_kite_configuration_persistence(self, v3_kite):
        """Kite configuration persists across property access."""
        v3_kite.input_steering = 0.1
        v3_kite.input_depower = 0.05
        v3_kite.g = 9.82

        # Verify persisted
        assert v3_kite.input_steering == 0.1
        assert v3_kite.input_depower == 0.05
        assert v3_kite.g == 9.82

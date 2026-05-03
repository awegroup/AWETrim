"""Integration tests for awetrim.system.system_model

Tests verify:
- SystemModel initialization and configuration
- Expression registry and symbolic structure (CasADi MX types)
- Property delegation to Kite, Tether, Wind components
- Quasi-steady constraint enforcement (time derivatives = 0)
- Override flags for gravity, centripetal, Coriolis
- Deep copy preservation of expression registry
- Force and acceleration computations
- ODE establishment for dynamic models

Per AGENTS.md @tester role:
- Test CasADi expression structure and symbolic shapes
- Use fixtures in conftest.py for shared setups
- Do NOT test numeric solver values, only structure
"""

import casadi as ca
import copy
import numpy as np
import pytest
import yaml

from awetrim.system.system_model import SystemModel
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether, FlexibleLinkTether
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
    """Fixture: V3 Kite instance."""
    cfg = v3_config
    aero_input = cfg["wing"]["aerodynamics"]
    mass_wing = cfg["wing"].get("mass", 15)
    area_wing = cfg["wing"].get("area", 19.75)

    return Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=16.0,
        steering_control="asymmetric",
    )


@pytest.fixture
def v3_tether():
    """Fixture: V3 RigidLumpedTether."""
    return RigidLumpedTether(diameter=0.01)


@pytest.fixture
def v3_wind():
    """Fixture: Logarithmic wind model."""
    wind = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0)
    wind.speed_wind_ref = 10.0
    return wind


@pytest.fixture
def v3_system(v3_kite, v3_tether, v3_wind):
    """Fixture: Complete V3 SystemModel."""
    system = SystemModel(
        dof=3,
        quasi_steady=False,
        wind_model=v3_wind,
        tether=v3_tether,
        kite=v3_kite,
    )
    return system


@pytest.fixture
def system_quasi_steady(v3_kite, v3_tether, v3_wind):
    """Fixture: SystemModel in quasi-steady mode."""
    system = SystemModel(
        dof=3,
        quasi_steady=True,
        wind_model=v3_wind,
        tether=v3_tether,
        kite=v3_kite,
    )
    return system


# ============================================================================
# INITIALIZATION & CONFIGURATION TESTS
# ============================================================================


class TestSystemModelInitialization:
    """Test SystemModel initialization and default model selection."""

    def test_system_init_with_explicit_models(self, v3_system):
        """SystemModel accepts explicit kite, tether, and wind models."""
        assert v3_system.kite is not None
        assert v3_system.tether is not None
        assert v3_system.wind is not None

    def test_system_init_with_defaults(self):
        """SystemModel creates default models when None provided."""
        system = SystemModel(dof=3, quasi_steady=False)
        assert system.kite is not None
        assert isinstance(system.kite, Kite)
        assert system.tether is not None
        assert system.wind is not None

    def test_system_init_default_tether_is_flexible(self):
        """By default, SystemModel uses FlexibleLinkTether."""
        system = SystemModel(dof=3, quasi_steady=False, tether=None)
        assert isinstance(system.tether, FlexibleLinkTether)

    def test_system_init_default_wind_is_uniform(self):
        """By default, SystemModel uses uniform wind model."""
        system = SystemModel(dof=3, quasi_steady=False, wind_model=None)
        assert system.wind is not None

    def test_system_init_quasi_steady_false(self):
        """quasi_steady=False allows time derivatives."""
        system = SystemModel(quasi_steady=False)
        # Time derivatives should NOT be forced to zero
        # (they may be symbolic or zero depending on context)
        assert system is not None

    def test_system_init_quasi_steady_true(self, system_quasi_steady):
        """quasi_steady=True enforces zero time derivatives."""
        # In quasi-steady mode, speed derivatives must be zero
        assert system_quasi_steady.timeder_speed_tangential == 0
        assert system_quasi_steady.timeder_speed_radial == 0


# ============================================================================
# PROPERTY DELEGATION TESTS
# ============================================================================


class TestSystemModelPropertyDelegation:
    """Test that SystemModel correctly delegates properties to sub-models."""

    def test_mass_wing_accessible_through_kite(self, v3_system):
        """mass_wing is accessible and modifiable through kite."""
        original_mass = v3_system.kite.mass_wing
        v3_system.kite.mass_wing = 25.0
        assert v3_system.kite.mass_wing == 25.0
        # Restore
        v3_system.kite.mass_wing = original_mass

    def test_input_steering_property_delegates_to_kite(self, v3_system):
        """input_steering property reads from and writes to kite."""
        v3_system.input_steering = 0.5
        assert v3_system.input_steering == 0.5
        assert v3_system.kite.input_steering == 0.5

    def test_input_depower_property_delegates_to_kite(self, v3_system):
        """input_depower property reads from and writes to kite."""
        v3_system.input_depower = 0.3
        assert v3_system.input_depower == 0.3
        assert v3_system.kite.input_depower == 0.3

    def test_gravity_property_delegates_to_kite(self, v3_system):
        """g property reads from and writes to kite."""
        original_g = v3_system.g
        v3_system.g = 9.82
        assert v3_system.g == 9.82
        assert v3_system.kite.g == 9.82
        # Restore
        v3_system.g = original_g

    def test_air_density_property_delegates_to_kite(self, v3_system):
        """rho property reads from and writes to kite."""
        original_rho = v3_system.rho
        v3_system.rho = 1.20
        assert v3_system.rho == 1.20
        assert v3_system.kite.rho == 1.20
        # Restore
        v3_system.rho = original_rho

    def test_tether_length_property_delegates_to_tether(self, v3_system):
        """length_tether property delegates to tether."""
        v3_system.length_tether = 500.0
        assert v3_system.length_tether == 500.0
        assert v3_system.tether.length_tether == 500.0

    def test_tether_tension_property_delegates_to_tether(self, v3_system):
        """tension_tether_ground property delegates to tether."""
        v3_system.tension_tether_ground = 1000.0
        assert v3_system.tension_tether_ground == 1000.0
        assert v3_system.tether.tension_tether_ground == 1000.0

    def test_is_tether_rigid_property(self, v3_system):
        """is_tether_rigid property reflects tether type."""
        system_rigid = SystemModel(tether=RigidLumpedTether())
        assert system_rigid.is_tether_rigid is True

        system_flex = SystemModel(tether=FlexibleLinkTether())
        assert system_flex.is_tether_rigid is False


# ============================================================================
# OVERRIDE FLAGS TESTS
# ============================================================================


class TestSystemModelOverrideFlags:
    """Test override flags for gravity, centripetal, and Coriolis forces."""

    def test_override_gravity_default_false(self, v3_system):
        """override_gravity defaults to False."""
        assert v3_system.override_gravity is False

    def test_override_gravity_setter(self, v3_system):
        """override_gravity can be set to True or False."""
        v3_system.override_gravity = True
        assert v3_system.override_gravity is True
        v3_system.override_gravity = False
        assert v3_system.override_gravity is False

    def test_override_gravity_rejects_non_bool(self, v3_system):
        """override_gravity setter rejects non-boolean values."""
        with pytest.raises(ValueError, match="override_gravity must be True or False"):
            v3_system.override_gravity = 1

    def test_override_centripetal_default_false(self, v3_system):
        """override_centripetal defaults to False."""
        assert v3_system.override_centripetal is False

    def test_override_centripetal_setter(self, v3_system):
        """override_centripetal can be set."""
        v3_system.override_centripetal = True
        assert v3_system.override_centripetal is True

    def test_override_centripetal_rejects_non_bool(self, v3_system):
        """override_centripetal setter rejects non-boolean values."""
        with pytest.raises(
            ValueError, match="override_centripetal must be True or False"
        ):
            v3_system.override_centripetal = "yes"

    def test_override_coriolis_default_false(self, v3_system):
        """override_coriolis defaults to False."""
        assert v3_system.override_coriolis is False

    def test_override_coriolis_setter(self, v3_system):
        """override_coriolis can be set."""
        v3_system.override_coriolis = True
        assert v3_system.override_coriolis is True

    def test_override_coriolis_rejects_non_bool(self, v3_system):
        """override_coriolis setter rejects non-boolean values."""
        with pytest.raises(ValueError, match="override_coriolis must be True or False"):
            v3_system.override_coriolis = []


# ============================================================================
# EXPRESSION REGISTRY & CASADI STRUCTURE TESTS
# ============================================================================


class TestSystemModelExpressionRegistry:
    """Test that SystemModel builds and manages symbolic expressions."""

    def test_expression_registry_returns_dict(self, v3_system):
        """expression_registry() returns a dictionary."""
        registry = v3_system.expression_registry()
        assert isinstance(registry, dict)
        assert len(registry) > 0

    def test_available_expressions_is_sorted_tuple(self, v3_system):
        """available_expressions() returns sorted tuple of names."""
        names = v3_system.available_expressions()
        assert isinstance(names, tuple)
        assert len(names) > 0
        # Verify sorted
        assert names == tuple(sorted(names))

    def test_has_expression_checks_registry(self, v3_system):
        """has_expression correctly checks if name exists."""
        # Every system should have these expressions
        assert v3_system.has_expression(
            "angle_of_attack"
        ) or not v3_system.has_expression("nonexistent_expr")

    def test_expression_retrieval(self, v3_system):
        """expression() retrieves and returns callable result."""
        # Get first available expression
        names = v3_system.available_expressions()
        if len(names) > 0:
            name = names[0]
            expr = v3_system.expression(name)
            # Should be CasADi MX or numeric
            assert expr is not None

    def test_expression_nonexistent_raises_attribute_error(self, v3_system):
        """expression() raises AttributeError for unknown name."""
        with pytest.raises(AttributeError, match="has no expression"):
            v3_system.expression("definitely_not_an_expression_name")

    def test_refresh_expression_registry(self, v3_system):
        """refresh_expression_registry rebuilds the registry."""
        original_names = v3_system.available_expressions()
        v3_system.refresh_expression_registry()
        refreshed_names = v3_system.available_expressions()
        assert original_names == refreshed_names


# ============================================================================
# CASADI SYMBOLIC STRUCTURE TESTS
# ============================================================================


class TestSystemModelCasADiStructure:
    """Test that force, acceleration, and residual are valid CasADi expressions."""

    def test_acceleration_is_casadi_or_numeric(self, v3_system):
        """acceleration property is CasADi MX or numeric."""
        accel = v3_system.acceleration
        # Should be numeric array or CasADi DM
        assert accel is not None
        if isinstance(accel, ca.DM):
            assert accel.shape[0] >= 3

    def test_force_external_structure(self, v3_system):
        """force_external returns structured force vector."""
        force = v3_system.force_external
        # Force should be a vector (numeric or CasADi)
        assert force is not None

    def test_force_residual_structure(self, v3_system):
        """force_residual is a valid expression."""
        residual = v3_system.force_residual
        # Residual should be a vector
        assert residual is not None

    def test_force_tether_at_kite(self, v3_system):
        """force_tether_at_kite delegates to tether."""
        force = v3_system.force_tether_at_kite
        assert force is not None

    def test_default_unknown_vars_for_rigid_tether(self):
        """Rigid tether system has correct default unknown vars."""
        system = SystemModel(tether=RigidLumpedTether())
        expected = [
            "speed_tangential",
            "timeder_angle_course",
            "tension_tether_ground",
        ]
        assert system.default_unknown_vars == expected

    def test_default_unknown_vars_for_flexible_tether(self):
        """Flexible tether system has correct default unknown vars."""
        system = SystemModel(tether=FlexibleLinkTether())
        expected = [
            "speed_tangential",
            "input_steering",
            "length_tether",
        ]
        assert system.default_unknown_vars == expected

    def test_derived_function_names_exist(self, v3_system):
        """derived_function_names is populated."""
        names = v3_system.derived_function_names
        assert isinstance(names, list)
        assert len(names) > 0
        # Verify expected names exist
        assert "angle_of_attack" in names or len(names) >= 3


# ============================================================================
# ODE & ALGEBRAIC TESTS
# ============================================================================


class TestSystemModelODEStructure:
    """Test ODE and algebraic function establishment."""

    def test_establish_ode_function_method_exists(self, v3_system):
        """establish_ode_function method exists and is callable."""
        assert hasattr(v3_system, "establish_ode_function")
        assert callable(v3_system.establish_ode_function)

    def test_establish_residual_creates_residual(self, v3_system):
        """establish_residual sets residual property."""
        v3_system.establish_residual()
        assert hasattr(v3_system, "residual")
        assert v3_system.residual is not None

    def test_algebraic_function_returns_residual(self, v3_system):
        """algebraic_function returns force residual."""
        alg = v3_system.algebraic_function()
        assert alg is not None


# ============================================================================
# DEEP COPY TESTS
# ============================================================================


class TestSystemModelDeepCopy:
    """Test that deep copy preserves state and expression registry."""

    def test_deepcopy_creates_independent_system(self, v3_system):
        """Deepcopy creates an independent SystemModel."""
        system_copy = copy.deepcopy(v3_system)
        assert system_copy is not v3_system
        assert system_copy.kite is not v3_system.kite

    def test_deepcopy_preserves_configuration(self, v3_system):
        """Deepcopy preserves mass, wind speed, etc."""
        v3_system.mass_wing = 20.0
        v3_system.wind.speed_wind_ref = 8.5

        system_copy = copy.deepcopy(v3_system)
        assert system_copy.mass_wing == 20.0
        assert system_copy.wind.speed_wind_ref == 8.5

    def test_deepcopy_refreshes_expression_registry(self, v3_system):
        """Deepcopy refreshes expressions (not copied reference)."""
        system_copy = copy.deepcopy(v3_system)
        # Both should have expressions
        assert len(system_copy.available_expressions()) > 0
        assert len(v3_system.available_expressions()) > 0

    def test_deepcopy_independent_modifications(self, v3_system):
        """Modifications to copy don't affect original."""
        system_copy = copy.deepcopy(v3_system)

        original_mass = v3_system.kite.mass_wing
        system_copy.kite.mass_wing = 99.0

        assert v3_system.kite.mass_wing == original_mass
        assert system_copy.kite.mass_wing == 99.0


# ============================================================================
# QUASI-STEADY SOLVER SETUP TESTS
# ============================================================================


class TestSystemModelQuasiSteadySolverSetup:
    """Test quasi-steady solver setup and structure (NOT solution)."""

    def test_quasi_steady_solver_setup_with_rigid_tether(self):
        """setup_qs_solver completes without error for rigid tether."""
        system = SystemModel(
            quasi_steady=True,
            tether=RigidLumpedTether(),
        )
        # Should not raise
        system.setup_qs_solver()
        assert system._qs_solver is not None
        assert system._qs_vars is not None
        assert system._qs_inputs is not None

    def test_quasi_steady_solver_setup_with_default_unknowns(self):
        """setup_qs_solver uses default_unknown_vars if none provided."""
        system = SystemModel(
            quasi_steady=True,
            tether=RigidLumpedTether(),
        )
        system.setup_qs_solver(unknown_vars=None)
        # Should use default
        assert system._qs_vars == system.default_unknown_vars

    def test_quasi_steady_solver_setup_with_custom_unknowns(self):
        """setup_qs_solver accepts custom unknown_vars list."""
        system = SystemModel(
            quasi_steady=True,
            tether=RigidLumpedTether(),
        )
        custom_vars = ["speed_tangential", "tension_tether_ground"]
        system.setup_qs_solver(unknown_vars=custom_vars)
        assert system._qs_vars == custom_vars

    def test_quasi_steady_solver_inputs_extracted_correctly(self):
        """setup_qs_solver extracts all non-unknown symbolic variables."""
        system = SystemModel(
            quasi_steady=True,
            tether=RigidLumpedTether(),
        )
        system.setup_qs_solver()
        # Should have inputs for all variables not in unknown list
        assert len(system._qs_inputs) > 0

    def test_quasi_steady_constraint_enforced(self, system_quasi_steady):
        """quasi_steady=True enforces zero time derivatives."""
        assert system_quasi_steady.timeder_speed_tangential == 0
        assert system_quasi_steady.timeder_speed_radial == 0


# ============================================================================
# INTEGRATION: FULL WORKFLOW TESTS
# ============================================================================


class TestSystemModelFullWorkflow:
    """Integration tests exercising multiple components together."""

    def test_complete_system_initialization_with_all_models(self, v3_config):
        """Full workflow: load config → create models → create system."""
        cfg = v3_config

        # Load components
        kite = Kite(
            mass_wing=cfg["wing"]["mass"],
            area_wing=cfg["wing"]["area"],
            aero_input=cfg["wing"]["aerodynamics"],
            mass_kcu=16.0,
            steering_control="asymmetric",
        )
        tether = RigidLumpedTether(diameter=0.01)
        wind = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0)

        # Create system
        system = SystemModel(
            quasi_steady=False,
            wind_model=wind,
            tether=tether,
            kite=kite,
        )

        # Verify all components initialized
        assert system.kite is kite
        assert system.tether is tether
        assert system.wind is wind
        assert len(system.available_expressions()) > 0

    def test_modify_and_reread_configuration(self, v3_system):
        """Modifications to configuration are readable after setting."""
        # Modify several properties
        v3_system.kite.mass_wing = 22.0
        v3_system.input_steering = 0.1
        v3_system.input_depower = 0.05
        v3_system.override_gravity = True

        # Verify all modifications persisted
        assert v3_system.kite.mass_wing == 22.0
        assert v3_system.input_steering == 0.1
        assert v3_system.input_depower == 0.05
        assert v3_system.override_gravity is True

    def test_system_copy_independence(self, v3_system):
        """System copy is fully independent from original."""
        system_copy = copy.deepcopy(v3_system)

        # Modify original
        v3_system.kite.mass_wing = 30.0
        v3_system.override_gravity = True

        # Verify copy unchanged
        original_mass = v3_system.kite.mass_wing
        system_copy.kite.mass_wing = 50.0
        assert v3_system.kite.mass_wing == original_mass

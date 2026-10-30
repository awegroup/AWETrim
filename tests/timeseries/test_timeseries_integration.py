"""Integration tests for awetrim.timeseries module

Tests validate:
- TimeSeries base class functionality
- State management and retrieval
- Energy computation
- Cycle orchestration
- ReelinSimple interface
- Configuration handling

Per AGENTS.md @tester role:
- Test class interfaces and method signatures
- Test protocol compliance
- Use mock patterns
- Verify data types and shapes
"""

import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest

from awetrim.timeseries.timeseries import TimeSeries
from awetrim.timeseries.Cycle import Cycle
from awetrim.timeseries.reelin_phase import (
    ReelinSimple,
    SimulationResult as ReelinSimulationResult,
)

from awetrim.utils.defaults import (
    DEFAULT_PATTERN_CONFIG,
    DEFAULT_RADIAL_PARAMETERS,
    DEFAULT_WINCH_CONFIG,
)
from awetrim import State

_MINIMAL_AERO_INPUT = {
    "params": {
        "angle_pitch_tether_0": 0.0,
        "slope_angle_pitch_tether_depower": 0.0,
        "CD0": 0.05,
    }
}


@pytest.fixture
def mock_system_model():
    """Create a minimal mock SystemModel."""
    model = MagicMock()
    model.quasi_steady = True
    model.dof = 3
    model.kite = MagicMock()
    model.kite.mass_wing = 2.5
    model.kite.area_wing = 10.0
    model.wind = MagicMock()
    model.wind.speed_wind_ref_value = 12.0
    model.tether = MagicMock()
    model.tether.length = 500.0
    return model


@pytest.fixture
def pattern_config_reelin():
    """Pattern config for ReelinSimple with path_parameters."""
    return {
        "path_parameters": {
            "elevation_start_ri": np.radians(30),
            "elevation_start_ro": np.radians(30),
            "elevation_start_riro": np.radians(90),
            "distance_radial_start": 360,
            "distance_radial_end": 100,
        },
        "radial_parameters": DEFAULT_RADIAL_PARAMETERS,
    }


@pytest.fixture
def state_list():
    """Create a list of State objects."""
    states = []
    for i in range(10):
        state = State(
            t=float(i),
            s=float(i * 0.1),
            s_dot=2.0,
            input_steering=0.1 * i,
            tension_tether_ground=50000.0 + i * 1000,
            distance_radial=200.0 + i * 5,
            speed_radial=0.2,
        )
        states.append(state.to_dict())
    return states


# ============================================================================
# TIMESERIES TESTS
# ============================================================================


class TestTimeSeries:
    """Test TimeSeries base class."""

    def test_init(self, mock_system_model):
        """TimeSeries initializes with kite_model."""
        ts = TimeSeries(kite_model=mock_system_model)
        assert ts.kite_model is mock_system_model
        assert ts.states == []

    def test_states_storage(self, mock_system_model, state_list):
        """TimeSeries stores and retrieves states."""
        ts = TimeSeries(kite_model=mock_system_model)
        ts.states = state_list
        assert len(ts.states) == 10
        assert ts.states[0]["t"] == 0.0

    def test_energy_property(self, mock_system_model, state_list):
        """TimeSeries computes energy from states."""
        ts = TimeSeries(kite_model=mock_system_model)
        ts.states = state_list
        energy = ts.energy
        assert isinstance(energy, (int, float, np.number))

    def test_return_variable(self, mock_system_model, state_list):
        """return_variable extracts time series data."""
        ts = TimeSeries(kite_model=mock_system_model)
        ts.states = state_list

        times = ts.return_variable("t")
        assert isinstance(times, (list, np.ndarray))
        assert len(times) == 10

        tensions = ts.return_variable("tension_tether_ground")
        assert len(tensions) == 10


# ============================================================================
# REELIN SIMPLE TESTS
# ============================================================================


class TestReelinSimple:
    """Test ReelinSimple interface."""

    def test_init_signature(self):
        """ReelinSimple has system_model keyword argument."""
        sig = inspect.signature(ReelinSimple.__init__)
        assert "system_model" in sig.parameters

    def test_creation(self, mock_system_model, pattern_config_reelin):
        """ReelinSimple can be created with pattern_config."""
        reelin = ReelinSimple(
            system_model=mock_system_model,
            pattern_config=pattern_config_reelin,
        )
        assert reelin.system_model is mock_system_model
        assert reelin.pattern_config == pattern_config_reelin

    def test_depower_parameters(self, mock_system_model, pattern_config_reelin):
        """ReelinSimple stores depower parameters."""
        reelin = ReelinSimple(
            system_model=mock_system_model,
            pattern_config=pattern_config_reelin,
            depower_ri=0.8,
            depower_riro=0.9,
        )
        assert reelin.depower_ri == 0.8
        assert reelin.depower_riro == 0.9


# ============================================================================
# CYCLE TESTS
# ============================================================================


class TestCycle:
    """Test Cycle class."""

    def test_init_signature(self):
        """Cycle requires aero_input and sim_config."""
        sig = inspect.signature(Cycle.__init__)
        assert "aero_input" in sig.parameters
        assert "sim_config" in sig.parameters

    def test_creation_uniform_wind(self):
        """Cycle creates models with uniform wind config."""
        sim_config = {
            "wind_model": "uniform",
            "speed_wind_ref": 12.0,
            "mass_wing": 2.5,
            "area_wing": 10.0,
            "tether_diameter": 0.005,
            "dof": 3,
        }
        cycle = Cycle(_MINIMAL_AERO_INPUT, sim_config)
        assert cycle.wind_model is not None
        assert cycle.kite is not None
        assert cycle.tether is not None

    def test_methods_exist(self):
        """Cycle has create_model and run_cycle methods."""
        sim_config = {
            "wind_model": "uniform",
            "speed_wind_ref": 12.0,
            "mass_wing": 2.5,
            "area_wing": 10.0,
            "tether_diameter": 0.005,
            "dof": 3,
        }
        cycle = Cycle(_MINIMAL_AERO_INPUT, sim_config)
        assert callable(cycle.create_model)
        assert callable(cycle.run_cycle)

    def test_create_model_quasi_steady(self):
        """create_model accepts quasi_steady parameter."""
        sim_config = {
            "wind_model": "uniform",
            "speed_wind_ref": 12.0,
            "mass_wing": 2.5,
            "area_wing": 10.0,
            "tether_diameter": 0.005,
            "dof": 3,
        }
        cycle = Cycle(_MINIMAL_AERO_INPUT, sim_config)
        sig = inspect.signature(cycle.create_model)
        assert "quasi_steady" in sig.parameters


# ============================================================================
# SIMULATION RESULT TESTS
# ============================================================================


class TestSimulationResult:
    """Test ReelinSimulationResult."""

    def test_creation(self):
        """ReelinSimulationResult can be instantiated."""
        result = ReelinSimulationResult(
            solution=None,
            optimized_config={},
            phase_variables={},
            energy_objective=1000.0,
            total_time=100.0,
        )
        assert result.energy_objective == 1000.0
        assert result.total_time == 100.0

    def test_fields(self):
        """ReelinSimulationResult fields are accessible."""
        config = {"param1": 1.0}
        result = ReelinSimulationResult(
            solution=None,
            optimized_config=config,
            phase_variables={"var": [1, 2, 3]},
            energy_objective=500.0,
            total_time=50.0,
        )
        assert result.optimized_config == config
        assert result.phase_variables == {"var": [1, 2, 3]}


# ============================================================================
# STATE OBJECT TESTS
# ============================================================================


class TestState:
    """Test State object."""

    def test_creation(self):
        """State object can be created."""
        state = State(
            t=0.0,
            s=0.0,
            s_dot=2.0,
            input_steering=0.0,
            tension_tether_ground=50000.0,
            distance_radial=200.0,
            speed_radial=0.2,
        )
        assert state.t == 0.0
        assert state.s == 0.0

    def test_to_dict(self):
        """State.to_dict() converts to dictionary."""
        state = State(
            t=1.0,
            s=0.5,
            s_dot=2.0,
            input_steering=0.1,
            tension_tether_ground=50000.0,
            distance_radial=200.0,
            speed_radial=0.2,
        )
        state_dict = state.to_dict()
        assert isinstance(state_dict, dict)
        assert state_dict["t"] == 1.0
        assert state_dict["s"] == 0.5


# ============================================================================
# CONFIGURATION CONSTANTS TESTS
# ============================================================================


class TestConfigs:
    """Test default configuration constants."""

    def test_default_pattern_config(self):
        """DEFAULT_PATTERN_CONFIG is valid dict."""
        assert isinstance(DEFAULT_PATTERN_CONFIG, dict)

    def test_default_radial_parameters(self):
        """DEFAULT_RADIAL_PARAMETERS has expected structure."""
        assert isinstance(DEFAULT_RADIAL_PARAMETERS, dict)
        assert "reeling_strategy" in DEFAULT_RADIAL_PARAMETERS

    def test_default_winch_config(self):
        """DEFAULT_WINCH_CONFIG defines speed and force limits."""
        assert isinstance(DEFAULT_WINCH_CONFIG, dict)
        assert "max_speed" in DEFAULT_WINCH_CONFIG
        assert "min_speed" in DEFAULT_WINCH_CONFIG
        assert DEFAULT_WINCH_CONFIG["max_speed"] > 0
        assert DEFAULT_WINCH_CONFIG["min_speed"] < 0

    def test_minimal_aero_input(self):
        """_MINIMAL_AERO_INPUT has required params."""
        assert "params" in _MINIMAL_AERO_INPUT
        assert "CD0" in _MINIMAL_AERO_INPUT["params"]

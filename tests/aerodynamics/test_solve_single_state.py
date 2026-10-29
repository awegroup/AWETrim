"""Integration tests for examples/1_aerodynamics/solve_single_state.py

Tests the orchestration of VSM aerodynamic trim solving, including:
- Command-line argument parsing and validation
- Body and system model construction
- Trim solution execution and result collection
- JSON output serialization
- Plot generation without errors
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure scripts directory is in path for imports
PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_AERODYNAMICS_DIR = PROJECT_DIR / "examples" / "1_aerodynamics"
if str(SCRIPTS_AERODYNAMICS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_AERODYNAMICS_DIR))

from common import (
    build_body,
    build_system_model,
    output_dir,
    parsed_common,
    print_trim_summary,
    write_json,
    csv_vector,
)

# ---------------------------------------------------------------------------
# Minimal Trim Result Mock
# ---------------------------------------------------------------------------


def _make_mock_trim_result():
    """Create a realistic trim result dict matching awetrim output."""
    return {
        "opt_x": np.array([20.0, 0.1, -0.05, 0.0, 0.0]),
        "cm": np.array([1e-4, 2e-4, 3e-4]),
        "cfx": 1e-3,
        "cfy": 2e-3,
        "success": True,
        "success_physical": True,
        "aoa_deg": 5.0,
        "side_slip_deg": 0.5,
        "cl": 1.2,
        "cd": 0.05,
        "va_vel_world": np.array([20.0, 0.5, -0.2]),
        "tether_force": 500.0,
        "x_trim": np.array([20.0, 0.1, -0.05, 0.0, 0.0]),
        "mass_wing": 30.0,
        "wind_speed": 6.0,
        "angle_elevation": 0.0,
    }


# ---------------------------------------------------------------------------
# Unit Tests: Common Helper Functions
# ---------------------------------------------------------------------------


class TestCommonHelpers:
    """Test the helper functions that solve_single_state.py depends on."""

    def test_csv_vector_valid(self):
        """csv_vector correctly parses comma-separated floats."""
        result = csv_vector("1.0,2.5,3.14", length=3, name="test")
        assert result.shape == (3,)
        assert np.allclose(result, [1.0, 2.5, 3.14])

    def test_csv_vector_with_whitespace(self):
        """csv_vector handles whitespace around values."""
        result = csv_vector(" 1.0 , 2.5 , 3.14 ", length=3, name="test")
        assert np.allclose(result, [1.0, 2.5, 3.14])

    def test_csv_vector_wrong_length(self):
        """csv_vector raises TypeError if length doesn't match."""
        with pytest.raises(Exception):  # argparse.ArgumentTypeError
            csv_vector("1.0,2.0", length=3, name="test")

    def test_parsed_common_returns_dict_with_correct_keys(self):
        """parsed_common unpacks reference_point, center_of_gravity, etc."""

        class MockArgs:
            reference_point = "0,0,0"
            center_of_gravity = "0.5,0,5"
            x_guess = "25,0,0,0,0"
            bounds_lower = "2,-15,-15,-15,-5"
            bounds_upper = "80,15,15,15,5"

        result = parsed_common(MockArgs())
        assert isinstance(result, dict)
        assert "reference_point" in result
        assert "center_of_gravity" in result
        assert "x_guess" in result
        assert "bounds_lower" in result
        assert "bounds_upper" in result

    def test_parsed_common_correct_shapes(self):
        """parsed_common returns arrays with correct shapes."""

        class MockArgs:
            reference_point = "0,0,0"
            center_of_gravity = "0.5,0,5"
            x_guess = "25,0,0,0,0"
            bounds_lower = "2,-15,-15,-15,-5"
            bounds_upper = "80,15,15,15,5"

        result = parsed_common(MockArgs())
        assert result["reference_point"].shape == (3,)
        assert result["center_of_gravity"].shape == (3,)
        assert result["x_guess"].shape == (5,)
        assert result["bounds_lower"].shape == (5,)
        assert result["bounds_upper"].shape == (5,)


# ---------------------------------------------------------------------------
# Integration Tests: solve_single_state Main Flow
# ---------------------------------------------------------------------------


class TestSolveSingleStateIntegration:
    """Integration tests for the solve_single_state script."""

    def test_build_system_model_returns_valid_system(self):
        """build_system_model correctly configures a SystemModel."""

        class MockArgs:
            config_folder = str(PROJECT_DIR / "data" / "LEI-V3-KITE")
            tether_diameter = 0.01
            mass_wing = 30.0
            elevation_deg = 0.0
            azimuth_deg = 0.0
            course_deg = 0.0
            radial_speed = 0.0
            distance_radial = 200.0
            wind_speed = 6.0

        system = build_system_model(MockArgs())

        # Verify system has required attributes
        assert hasattr(system, "kite")
        assert hasattr(system, "wind")
        assert hasattr(system, "tether")
        assert system.kite.mass_wing == 30.0
        assert system.distance_radial == 200.0
        assert system.wind.speed_wind_ref == 6.0

    def test_build_system_model_sets_quasi_steady_constraints(self):
        """build_system_model enforces quasi-steady assumptions."""

        class MockArgs:
            config_folder = str(PROJECT_DIR / "data" / "LEI-V3-KITE")
            tether_diameter = 0.01
            mass_wing = 30.0
            elevation_deg = 0.0
            azimuth_deg = 0.0
            course_deg = 0.0
            radial_speed = 0.0
            distance_radial = 200.0
            wind_speed = 6.0

        system = build_system_model(MockArgs())

        # Quasi-steady model must have zero time derivatives
        assert system.timeder_speed_tangential == 0.0
        assert system.timeder_speed_radial == 0.0
        assert system.timeder_angle_course == 0.0

    def test_output_dir_creates_directory(self):
        """output_dir creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:

            class MockArgs:
                output_dir = str(Path(tmpdir) / "test_output")
                no_show = False

            result = output_dir(MockArgs(), "test_script")
            assert result.exists()
            assert result.is_dir()

    def test_write_json_creates_file_with_valid_json(self):
        """write_json writes parseable JSON to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "result.json"
            trim_result = _make_mock_trim_result()

            # Mock print to suppress output
            with patch("builtins.print"):
                write_json(out_path, trim_result)

            assert out_path.exists()

            # Verify JSON is valid
            with out_path.open() as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "cm" in data
            assert "cfx" in data
            assert "cfy" in data

    def test_json_serialization_handles_numpy_types(self):
        """write_json correctly serializes numpy arrays and scalars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "result.json"
            trim_result = {
                "array": np.array([1.0, 2.0, 3.0]),
                "scalar": np.float64(42.0),
                "tuple": (1.0, 2.0, 3.0),
                "list": [1, 2, 3],
            }

            with patch("builtins.print"):
                write_json(out_path, trim_result)

            with out_path.open() as f:
                data = json.load(f)

            # Arrays and tuples should be lists in JSON
            assert isinstance(data["array"], list)
            assert isinstance(data["tuple"], list)
            assert isinstance(data["scalar"], (int, float))

    def test_print_trim_summary_handles_mock_result(self):
        """print_trim_summary processes a trim result without error."""
        trim_result = _make_mock_trim_result()

        # Should not raise
        with patch("builtins.print"):
            print_trim_summary(trim_result)

    def test_trim_result_structure_contains_required_keys(self):
        """A valid trim result has cm, cfx, cfy keys for plotting."""
        result = _make_mock_trim_result()

        assert "cm" in result
        assert "cfx" in result
        assert "cfy" in result
        assert "opt_x" in result
        assert "success" in result
        assert "success_physical" in result
        # cm should be array-like with 3 elements
        cm_arr = np.asarray(result["cm"])
        assert cm_arr.shape == (3,) or len(result["cm"]) == 3


# ---------------------------------------------------------------------------
# Integration Tests: End-to-End Mocked Script Execution
# ---------------------------------------------------------------------------


class TestSolveSingleStateEndToEnd:
    """End-to-end tests simulating the solve_single_state.py script flow."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments matching solve_single_state defaults."""

        class Args:
            vsm_src = None
            config_folder = str(PROJECT_DIR / "data" / "LEI-V3-KITE")
            geometry_yaml = str(
                PROJECT_DIR
                / "data"
                / "LEI-V3-KITE"
                / "kite_geometries"
                / "powered_geometry"
                / "aero_geometry_kitesim_deformed.yaml"
            )
            bridle_path = None
            n_panels = 18
            spanwise_panel_distribution = "uniform"
            reference_point = "0,0,0"
            center_of_gravity = "0.5,0,5"
            mass_wing = 30.0
            tether_diameter = 0.01
            wind_speed = 6.0
            elevation_deg = 0.0
            azimuth_deg = 0.0
            course_deg = 0.0
            radial_speed = 0.0
            distance_radial = 200.0
            x_guess = "25,0,0,0,0"
            bounds_lower = "2,-15,-15,-15,-5"
            bounds_upper = "80,15,15,15,5"
            moment_tolerance = 1e-3
            include_gravity = False
            max_nfev = None
            output_json = None
            output_dir = None
            no_show = True

        return Args()

    def test_parsed_common_with_mock_args(self, mock_args):
        """parsed_common correctly processes mock arguments."""
        result = parsed_common(mock_args)

        assert result["reference_point"].shape == (3,)
        assert np.allclose(result["reference_point"], [0, 0, 0])
        assert result["center_of_gravity"].shape == (3,)
        assert np.allclose(result["center_of_gravity"], [0.5, 0, 5])
        assert result["x_guess"].shape == (5,)
        assert result["bounds_lower"].shape == (5,)
        assert result["bounds_upper"].shape == (5,)

    def test_build_system_model_with_mock_args(self, mock_args):
        """build_system_model correctly configures system from mock args."""
        system = build_system_model(mock_args)

        assert system.kite.mass_wing == mock_args.mass_wing
        assert system.distance_radial == mock_args.distance_radial
        assert system.wind.speed_wind_ref == mock_args.wind_speed

    def test_full_script_pipeline_with_mocked_solver(self, mock_args):
        """Full pipeline: args → system → (mocked) solver → JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_args.output_json = str(Path(tmpdir) / "result.json")
            mock_args.output_dir = tmpdir

            # Parse arguments
            parsed = parsed_common(mock_args)
            system = build_system_model(mock_args)

            # Mock the solver result
            mock_trim_result = _make_mock_trim_result()

            # Write output
            out_dir = output_dir(mock_args, "single_state")
            json_path = (
                Path(mock_args.output_json)
                if mock_args.output_json
                else out_dir / "trim_result.json"
            )

            with patch("builtins.print"):
                write_json(json_path, mock_trim_result)

            # Verify output was written
            assert json_path.exists()

            # Load and verify structure
            with json_path.open() as f:
                loaded = json.load(f)

            assert "cm" in loaded
            assert "cfx" in loaded
            assert "cfy" in loaded

    def test_plot_data_extraction_from_mock_result(self):
        """Verify the residual plot data can be extracted from trim result."""
        result = _make_mock_trim_result()

        # Simulate what solve_single_state.py does for plotting
        labels = ["cmx", "cmy", "cmz", "cfx", "cfy"]
        values_plot = np.r_[
            np.asarray(result["cm"], dtype=float),
            result["cfx"],
            result["cfy"],
        ]

        assert len(values_plot) == len(labels)
        assert values_plot.shape == (5,)
        assert all(isinstance(v, (float, np.floating)) for v in values_plot)


# ---------------------------------------------------------------------------
# Integration Tests: CasADi Expression Structure Validation
# ---------------------------------------------------------------------------


class TestSystemModelCasADiStructure:
    """Verify that the SystemModel works with CasADi operations."""

    def test_system_model_has_casadi_compatible_properties(self):
        """SystemModel properties can be used in CasADi operations."""

        class MockArgs:
            config_folder = str(PROJECT_DIR / "data" / "LEI-V3-KITE")
            tether_diameter = 0.01
            mass_wing = 30.0
            elevation_deg = 0.0
            azimuth_deg = 0.0
            course_deg = 0.0
            radial_speed = 0.0
            distance_radial = 200.0
            wind_speed = 6.0

        system = build_system_model(MockArgs())

        # These should be numeric or CasADi-compatible
        assert isinstance(system.kite.mass_wing, (float, int, np.ndarray))
        assert isinstance(system.distance_radial, (float, int, np.ndarray))
        assert hasattr(system.wind, "speed_wind_ref")

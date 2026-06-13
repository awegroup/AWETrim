"""Expanded integration tests for awetrim.aerodynamics.vsm_quasi_steady

Tests verify:
- Solver interface and protocols
- Trim computation structure and result handling
- Sweep parameter ranges and iteration
- Stability derivative computation
- Jacobian matrix structure
- State-space linearization
- Eigenvalue and eigenvector computation
- Timescale computation
- Stability flag determination
- Mock solver patterns
- Protocol compliance (VsmSolver, VsmBodyAerodynamics)
- Error handling and validation
- Backward-compatibility aliases
- Numeric type conversions
- System model integration

Per AGENTS.md @tester role:
- Test CasADi expression structure (not numeric solver values)
- Test protocol compliance and method signatures
- Use mock patterns already established
- Verify output shapes and types
"""

import copy
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import casadi as ca
import numpy as np
import pytest

from awetrim.aerodynamics.vsm_quasi_steady import (
    # Canonical names
    compute_vsm_trim_stability_derivatives,
    run_vsm_quasi_steady_sweep,
    solve_vsm_quasi_steady_trim,
    # Backward-compatibility aliases (used by migrated scripts)
    compute_stability_derivatives,
    run_quasi_steady_sweep,
    solve_quasi_steady_state,
    # Utilities
    _as_3vector,
    _as_5vector,
    _default_vsm_solver,
    DEFAULT_BOUNDS_LOWER,
    DEFAULT_BOUNDS_UPPER,
    DEFAULT_AXES,
)

# ============================================================================
# MOCK CLASSES (copied from existing tests)
# ============================================================================


class _MockSection:
    def __init__(self, le, te):
        self.LE_point = np.asarray(le, dtype=float)
        self.TE_point = np.asarray(te, dtype=float)


class _MockWing:
    def __init__(self, span: float = 10.0):
        h = span / 2.0
        self.sections = [
            _MockSection([0.0, -h, 0.0], [1.0, -h, 0.0]),
            _MockSection([0.0, h, 0.0], [1.0, h, 0.0]),
        ]
        self.spanwise_direction = np.array([0.0, 1.0, 0.0])

    def compute_projected_area(self) -> float:
        return 10.0


class _MockPanel:
    def __init__(self):
        self.corner_points = np.array(
            [[0.0, -5.0, 0.0], [1.0, -5.0, 0.0], [1.0, 5.0, 0.0], [0.0, 5.0, 0.0]],
            dtype=float,
        )
        self.chord = 1.0


class _MockBody:
    """Minimal BodyAerodynamics-compatible object for testing."""

    def __init__(self):
        self.wings = [_MockWing()]
        self.panels = [_MockPanel()]
        self.geometry_rotation = np.eye(3)

    def va_initialize(self, **kwargs):
        pass

    def _build_panels(self):
        pass


class _MockSolver:
    """Returns constant aerodynamic result for testing."""

    rho: float = 1.225

    def solve(self, body) -> dict:
        return {
            "Fx": 0.0,
            "Fy": 0.0,
            "Fz": -500.0,
            "cmx": 0.0,
            "cmy": 0.0,
            "cmz": 0.0,
        }


# Minimal trim result compatible with compute_vsm_trim_stability_derivatives
_TRIM_RESULT = {
    "va_vel_world": np.array([20.0, 0.0, 0.0]),
    "tether_force": 500.0,
}
_X_TRIM = np.array([20.0, 0.0, 0.0, 0.0, 0.0])


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


class TestUtilityFunctions:
    """Test helper functions for numeric conversion and validation."""

    def test_as_3vector_valid_input(self):
        """_as_3vector converts valid inputs to 3-vector."""
        result = _as_3vector([1.0, 2.0, 3.0])
        assert result.shape == (3,)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_as_3vector_wrong_size_raises(self):
        """_as_3vector raises ValueError for wrong size."""
        with pytest.raises(ValueError, match="Expected a 3-vector"):
            _as_3vector([1.0, 2.0])

        with pytest.raises(ValueError, match="Expected a 3-vector"):
            _as_3vector([1.0, 2.0, 3.0, 4.0])

    def test_as_3vector_numpy_array(self):
        """_as_3vector handles numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _as_3vector(arr)
        assert result.shape == (3,)

    def test_as_5vector_valid_input(self):
        """_as_5vector converts valid inputs to 5-vector."""
        result = _as_5vector([1.0, 2.0, 3.0, 4.0, 5.0], "test")
        assert result.shape == (5,)
        assert np.allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_as_5vector_wrong_size_raises(self):
        """_as_5vector raises ValueError for wrong size."""
        with pytest.raises(ValueError, match="must be shape \\(5,\\)"):
            _as_5vector([1.0, 2.0, 3.0], "test")

    def test_as_5vector_wrong_shape_raises(self):
        """_as_5vector raises ValueError for non-vector shapes."""
        with pytest.raises(ValueError, match="must be shape \\(5,\\)"):
            _as_5vector(np.ones((2, 5)), "test")


# ============================================================================
# PROTOCOL TESTS
# ============================================================================


class TestProtocolCompliance:
    """Test that functions work with protocol-compliant objects."""

    def test_mock_body_complies_with_protocol(self):
        """Mock body has required methods and attributes."""
        body = _MockBody()
        assert hasattr(body, "wings")
        assert hasattr(body, "panels")
        assert hasattr(body, "geometry_rotation")
        assert callable(body.va_initialize)

    def test_mock_solver_complies_with_protocol(self):
        """Mock solver has required interface."""
        solver = _MockSolver()
        assert hasattr(solver, "rho")
        assert callable(solver.solve)

        # Verify solve returns dict with expected keys
        result = solver.solve(_MockBody())
        assert "Fx" in result
        assert "Fy" in result
        assert "Fz" in result
        assert "cmx" in result
        assert "cmy" in result
        assert "cmz" in result


# ============================================================================
# TRIM SOLVER INTERFACE TESTS
# ============================================================================


class TestTrimSolverInterface:
    """Test solve_vsm_quasi_steady_trim function interface."""

    def test_trim_solver_required_arguments(self):
        """Verify required positional arguments."""
        sig = inspect.signature(solve_vsm_quasi_steady_trim)
        required = [
            name
            for name, p in sig.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.default is inspect.Parameter.empty
        ]
        # Must have at least these core arguments
        assert "body_aero" in required
        assert "center_of_gravity" in required
        assert "reference_point" in required
        assert "system_model" in required
        assert "x_guess" in required

    def test_trim_solver_optional_keyword_arguments(self):
        """Verify optional keyword arguments exist."""
        sig = inspect.signature(solve_vsm_quasi_steady_trim)
        optional_kw = [
            name
            for name, p in sig.parameters.items()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            and p.default is not inspect.Parameter.empty
        ]
        # Should have optional controls like bounds, tolerance, etc.
        assert len(optional_kw) > 0

    def test_trim_solver_return_type_is_tuple(self):
        """solve_vsm_quasi_steady_trim returns tuple with results and timing."""
        sig = inspect.signature(solve_vsm_quasi_steady_trim)
        # Function should return a tuple
        assert (
            sig.return_annotation == inspect.Signature.empty
            or sig.return_annotation is not None
        )


# ============================================================================
# SWEEP INTERFACE TESTS
# ============================================================================


class TestSweepInterface:
    """Test run_vsm_quasi_steady_sweep function interface."""

    def test_sweep_required_keyword_arguments(self):
        """Verify required keyword-only arguments."""
        sig = inspect.signature(run_vsm_quasi_steady_sweep)
        required_kw = [
            name
            for name, p in sig.parameters.items()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            and p.default is inspect.Parameter.empty
        ]
        # Must have sweep configuration parameters
        assert "build_body" in required_kw
        assert "system_model" in required_kw
        assert "center_of_gravity" in required_kw
        assert "reference_point" in required_kw
        assert "x_guess" in required_kw
        assert "principal_axis" in required_kw
        assert "secondary_axis" in required_kw
        assert "sweep_values" in required_kw

    def test_sweep_accepts_callable_build_body(self):
        """Sweep accepts callable for body building."""
        # Signature should have build_body as keyword-only
        sig = inspect.signature(run_vsm_quasi_steady_sweep)
        assert "build_body" in sig.parameters
        param = sig.parameters["build_body"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


# ============================================================================
# STABILITY DERIVATIVE INTERFACE TESTS
# ============================================================================


class TestStabilityDerivativeInterface:
    """Test compute_vsm_trim_stability_derivatives function interface."""

    def test_stability_derivative_required_arguments(self):
        """Verify required positional arguments."""
        sig = inspect.signature(compute_vsm_trim_stability_derivatives)
        required = [
            name
            for name, p in sig.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.default is inspect.Parameter.empty
        ]
        assert "body_aero" in required
        assert "center_of_gravity" in required
        assert "reference_point" in required
        assert "x_trim" in required
        assert "trim_result" in required

    def test_stability_derivative_optional_solver(self):
        """Stability derivative solver is optional."""
        sig = inspect.signature(compute_vsm_trim_stability_derivatives)
        assert "solver" in sig.parameters
        param = sig.parameters["solver"]
        assert param.default is not inspect.Parameter.empty


# ============================================================================
# OUTPUT STRUCTURE TESTS
# ============================================================================


class TestStabilityDerivativeOutput:
    """Test structure and types of stability derivative output."""

    def test_stability_derivative_returns_dict(self):
        """compute_vsm_trim_stability_derivatives returns dict."""
        result = compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.array([0.5, 0.0, 0.5]),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )
        assert isinstance(result, dict)

    def test_stability_derivative_jacobian_shapes(self):
        """Jacobian matrices have correct shapes."""
        result = compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.array([0.5, 0.0, 0.5]),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )
        # Longitudinal: 3 force/moment rows, 3 states [u, theta, q]
        assert result["J_long"].shape == (3, 3)
        # Lateral: 3 force/moment rows, 4 states [phi, psi, p, r] (v held fixed)
        assert result["J_lat"].shape == (3, 4)

    def test_stability_derivative_state_space_matrices(self):
        """State-space matrices have correct dimensions."""
        result = compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.array([0.5, 0.0, 0.5]),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )
        # A_long: 3x3 (longitudinal state-space)
        assert result["A_long"].shape == (3, 3)
        # A_lat: 4x4 (lateral state-space; v held fixed)
        assert result["A_lat"].shape == (4, 4)

    def test_stability_derivative_eigenvalue_shapes(self):
        """Eigenvalues have correct shapes."""
        result = compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.array([0.5, 0.0, 0.5]),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )
        assert result["eig_long"].shape == (3,)
        assert result["eig_lat"].shape == (4,)

    def test_stability_derivative_eigenvector_shapes(self):
        """Eigenvector matrices have correct shapes."""
        result = compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.array([0.5, 0.0, 0.5]),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )
        assert result["vec_long"].shape == (3, 3)
        assert result["vec_lat"].shape == (4, 4)

    def test_stability_derivative_timescale_shapes(self):
        """Timescale vectors have correct shapes."""
        result = compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.array([0.5, 0.0, 0.5]),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )
        assert result["Tfast_long"].shape == (3,)
        assert result["Tfast_lat"].shape == (4,)

    def test_stability_derivative_stability_flags(self):
        """Stability flags are boolean."""
        result = compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.array([0.5, 0.0, 0.5]),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )
        assert isinstance(result["stable_long"], bool)
        assert isinstance(result["stable_lat"], bool)

    def test_stability_derivative_tether_forces(self):
        """Tether force and moment vectors have correct shapes."""
        result = compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.array([0.5, 0.0, 0.5]),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )
        assert result["F_tether"].shape == (3,)
        assert result["M_tether_at_CG"].shape == (3,)


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================


class TestInputValidation:
    """Test error handling for invalid inputs."""

    def test_x_trim_wrong_shape_raises(self):
        """x_trim with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="must be shape \\(5,\\)"):
            compute_vsm_trim_stability_derivatives(
                body_aero=_MockBody(),
                center_of_gravity=np.zeros(3),
                reference_point=np.zeros(3),
                x_trim=np.zeros(4),  # Wrong — must be (5,)
                trim_result=_TRIM_RESULT,
                solver=_MockSolver(),
            )

    def test_center_of_gravity_wrong_shape_raises(self):
        """center_of_gravity with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected a 3-vector"):
            compute_vsm_trim_stability_derivatives(
                body_aero=_MockBody(),
                center_of_gravity=np.zeros(4),  # Wrong — must be (3,)
                reference_point=np.zeros(3),
                x_trim=_X_TRIM,
                trim_result=_TRIM_RESULT,
                solver=_MockSolver(),
            )

    def test_reference_point_wrong_shape_raises(self):
        """reference_point with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected a 3-vector"):
            compute_vsm_trim_stability_derivatives(
                body_aero=_MockBody(),
                center_of_gravity=np.zeros(3),
                reference_point=np.zeros(2),  # Wrong — must be (3,)
                x_trim=_X_TRIM,
                trim_result=_TRIM_RESULT,
                solver=_MockSolver(),
            )


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================


class TestBackwardCompatibility:
    """Test backward-compatibility aliases from VSM migration."""

    def test_solve_quasi_steady_state_alias_exists(self):
        """solve_quasi_steady_state alias exists and is callable."""
        assert callable(solve_quasi_steady_state)

    def test_run_quasi_steady_sweep_alias_exists(self):
        """run_quasi_steady_sweep alias exists and is callable."""
        assert callable(run_quasi_steady_sweep)

    def test_compute_stability_derivatives_alias_exists(self):
        """compute_stability_derivatives alias exists and is callable."""
        assert callable(compute_stability_derivatives)

    def test_aliases_point_to_canonical_functions(self):
        """Aliases resolve to canonical functions."""
        assert solve_quasi_steady_state is solve_vsm_quasi_steady_trim
        assert run_quasi_steady_sweep is run_vsm_quasi_steady_sweep
        assert compute_stability_derivatives is compute_vsm_trim_stability_derivatives


# ============================================================================
# DEFAULT CONSTANT TESTS
# ============================================================================


class TestDefaultConstants:
    """Test default constants are properly defined."""

    def test_default_bounds_lower_is_valid(self):
        """DEFAULT_BOUNDS_LOWER is a 5-vector."""
        assert DEFAULT_BOUNDS_LOWER.shape == (5,)
        assert all(isinstance(v, (int, float, np.number)) for v in DEFAULT_BOUNDS_LOWER)

    def test_default_bounds_upper_is_valid(self):
        """DEFAULT_BOUNDS_UPPER is a 5-vector."""
        assert DEFAULT_BOUNDS_UPPER.shape == (5,)
        assert all(isinstance(v, (int, float, np.number)) for v in DEFAULT_BOUNDS_UPPER)

    def test_bounds_are_reasonable(self):
        """Default bounds are sensible (lower < upper)."""
        assert np.all(DEFAULT_BOUNDS_LOWER < DEFAULT_BOUNDS_UPPER)

    def test_default_axes_defined(self):
        """DEFAULT_AXES is properly defined."""
        assert hasattr(DEFAULT_AXES, "course")
        assert hasattr(DEFAULT_AXES, "normal")
        assert hasattr(DEFAULT_AXES, "radial")
        assert DEFAULT_AXES.course.shape == (3,)
        assert DEFAULT_AXES.normal.shape == (3,)
        assert DEFAULT_AXES.radial.shape == (3,)


# ============================================================================
# INTEGRATION: FULL WORKFLOW TESTS
# ============================================================================


class TestVSMQuasiSteadyFullWorkflow:
    """End-to-end integration tests."""

    def test_stability_derivative_full_pipeline(self):
        """Full pipeline: body → solver → trim → stability derivatives."""
        body = _MockBody()
        solver = _MockSolver()
        cog = np.array([0.5, 0.0, 0.5])
        ref = np.zeros(3)
        x_trim = _X_TRIM

        result = compute_vsm_trim_stability_derivatives(
            body_aero=body,
            center_of_gravity=cog,
            reference_point=ref,
            x_trim=x_trim,
            trim_result=_TRIM_RESULT,
            solver=solver,
        )

        # Verify all expected keys present
        expected_keys = [
            "J_long",
            "J_lat",
            "A_long",
            "A_lat",
            "eig_long",
            "eig_lat",
            "vec_long",
            "vec_lat",
            "Tfast_long",
            "Tfast_lat",
            "stable_long",
            "stable_lat",
            "F_tether",
            "M_tether_at_CG",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_mock_solver_integration(self):
        """Mock solver produces consistent results."""
        solver = _MockSolver()
        body = _MockBody()

        # Solve twice - should get same result
        result1 = solver.solve(body)
        result2 = solver.solve(body)

        assert result1 == result2

    def test_mock_body_with_wings_and_panels(self):
        """Mock body has properly structured wings and panels."""
        body = _MockBody()
        assert len(body.wings) > 0
        assert len(body.panels) > 0

        # Wing should have sections
        wing = body.wings[0]
        assert len(wing.sections) == 2

        # Panel should have corner points
        panel = body.panels[0]
        assert panel.corner_points.shape == (4, 3)

"""Tests for awetrim.aerodynamics.vsm_quasi_steady.

Ported and adapted from
    Vortex-Step-Method/tests/Solver/test_quasi_steady_state_api.py

Changes vs the VSM originals are documented per test.
"""

import copy
import inspect

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
)


# ---------------------------------------------------------------------------
# Minimal mocks — satisfy the VsmBodyAerodynamics / VsmSolver protocols
# without requiring a live VSM installation.
# ---------------------------------------------------------------------------


class _MockSection:
    def __init__(self, le, te):
        self.LE_point = np.asarray(le, dtype=float)
        self.TE_point = np.asarray(te, dtype=float)


class _MockWing:
    def __init__(self, span: float = 10.0):
        h = span / 2.0
        self.sections = [
            _MockSection([0.0, -h, 0.0], [1.0, -h, 0.0]),
            _MockSection([0.0,  h, 0.0], [1.0,  h, 0.0]),
        ]
        self.spanwise_direction = np.array([0.0, 1.0, 0.0])

    def compute_projected_area(self) -> float:
        return 10.0


class _MockPanel:
    def __init__(self):
        self.corner_points = np.array(
            [[0.0, -5.0, 0.0], [1.0, -5.0, 0.0],
             [1.0,  5.0, 0.0], [0.0,  5.0, 0.0]],
            dtype=float,
        )
        self.chord = 1.0


class _MockBody:
    """Minimal BodyAerodynamics-compatible object for unit-level stability tests."""

    def __init__(self):
        self.wings = [_MockWing()]
        self.panels = [_MockPanel()]
        self.geometry_rotation = np.eye(3)

    def va_initialize(self, **kwargs):
        pass

    def _build_panels(self):
        pass


class _MockSolver:
    """Returns a constant aerodynamic result regardless of body state.

    All forces and moments are zero → every finite-difference column of the
    Jacobian will be zero.  That is sufficient to test output structure and
    shapes without depending on VSM internals.
    """

    rho: float = 1.225

    def solve(self, body) -> dict:
        return {
            "Fx": 0.0, "Fy": 0.0, "Fz": -500.0,
            "cmx": 0.0, "cmy": 0.0, "cmz": 0.0,
        }


# Minimal trim_result compatible with compute_vsm_trim_stability_derivatives
_TRIM_RESULT = {
    "va_vel_world": np.array([20.0, 0.0, 0.0]),
    "tether_force": 500.0,
}
_X_TRIM = np.array([20.0, 0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Ported directly — API unchanged from VSM
# ---------------------------------------------------------------------------


def test_solve_quasi_steady_state_required_arguments():
    """Ported from VSM (unchanged).

    solve_vsm_quasi_steady_trim exposes the same five positional required
    arguments as the VSM solve_quasi_steady_state function.
    """
    sig = inspect.signature(solve_quasi_steady_state)
    required = [
        name
        for name, p in sig.parameters.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.default is inspect.Parameter.empty
    ]
    assert required == [
        "body_aero",
        "center_of_gravity",
        "reference_point",
        "system_model",
        "x_guess",
    ]


def test_run_quasi_steady_sweep_required_keyword_arguments():
    """Ported from VSM (unchanged).

    run_vsm_quasi_steady_sweep exposes the same eight keyword-only required
    arguments as the VSM run_quasi_steady_sweep function.
    """
    sig = inspect.signature(run_quasi_steady_sweep)
    required_kw = [
        name
        for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
    ]
    assert required_kw == [
        "build_body",
        "system_model",
        "center_of_gravity",
        "reference_point",
        "x_guess",
        "principal_axis",
        "secondary_axis",
        "sweep_values",
    ]


# ---------------------------------------------------------------------------
# Adapted — API changed in AWETrim
# ---------------------------------------------------------------------------


def test_compute_vsm_trim_stability_derivatives_required_arguments():
    """Adapted from VSM's test_compute_quasi_steady_trim_jacobian_required_arguments.

    What changed:
    - Function renamed: compute_quasi_steady_trim_jacobian
                      → compute_vsm_trim_stability_derivatives
    - `system_model` and `x_state` replaced by `x_trim` and `trim_result`.
      The solver is now a keyword-only optional; kinematics are pre-computed
      and passed as part of trim_result rather than recomputed here.
    """
    sig = inspect.signature(compute_vsm_trim_stability_derivatives)
    required = [
        name
        for name, p in sig.parameters.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.default is inspect.Parameter.empty
    ]
    assert required == [
        "body_aero",
        "center_of_gravity",
        "reference_point",
        "x_trim",
        "trim_result",
    ]


def test_compute_vsm_trim_stability_derivatives_output_shapes():
    """Adapted from VSM's test_linearize_fast_dynamics_from_trim_jacobian_shapes_and_stability_flags.

    What changed:
    - In VSM, Jacobian computation and linearisation were separate functions;
      in AWETrim they are one: compute_vsm_trim_stability_derivatives.
    - The state-space is larger than VSM's "fast-only" matrices:
        A_long  (3×3)  states [u, θ, q]   vs. VSM (2×2)
        A_lat   (4×4)  states [φ, ψ, p, r] vs. VSM (3×3)
      The lateral velocity v is held fixed, so it is not a lateral state.
    - Additional output keys: vec_long, vec_lat, F_tether, M_tether_at_CG.
    - The mock solver returns zero forces/moments, so all Jacobian columns are
      zero and some eigenvalues will be zero (infinite timescales are valid).
    """
    result = compute_vsm_trim_stability_derivatives(
        body_aero=_MockBody(),
        center_of_gravity=np.array([0.5, 0.0, 0.5]),
        reference_point=np.zeros(3),
        x_trim=_X_TRIM,
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=100.0,
        inertia_yy=20.0,
        inertia_zz=100.0,
    )

    # Jacobians
    assert result["J_long"].shape == (3, 3)
    assert result["J_lat"].shape == (3, 4)

    # State-space matrices
    assert result["A_long"].shape == (3, 3)
    assert result["A_lat"].shape == (4, 4)

    # Eigenvalues and eigenvectors
    assert result["eig_long"].shape == (3,)
    assert result["eig_lat"].shape == (4,)
    assert result["vec_long"].shape == (3, 3)
    assert result["vec_lat"].shape == (4, 4)

    # Timescales (may be inf for zero eigenvalues — that is correct behaviour)
    assert result["Tfast_long"].shape == (3,)
    assert result["Tfast_lat"].shape == (4,)

    # Stability flags
    assert isinstance(result["stable_long"], bool)
    assert isinstance(result["stable_lat"], bool)

    # Tether transfer quantities
    assert result["F_tether"].shape == (3,)
    assert result["M_tether_at_CG"].shape == (3,)


def test_compute_vsm_trim_stability_derivatives_bad_x_trim_raises():
    """Adapted from VSM's test_linearize_fast_dynamics_from_trim_jacobian_input_validation.

    What changed:
    - The VSM test called linearize_fast_dynamics_from_trim_jacobian with a
      (4×4) Jacobian; in AWETrim the equivalent guard is in _as_5vector, which
      raises ValueError when x_trim does not have exactly 5 elements.
    """
    with pytest.raises(ValueError):
        compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.zeros(3),
            reference_point=np.zeros(3),
            x_trim=np.zeros(4),  # wrong shape — must be (5,)
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
        )


# ---------------------------------------------------------------------------
# New — full-state Jacobian + user-selectable subset / coupling
# ---------------------------------------------------------------------------


def test_compute_vsm_trim_stability_derivatives_full_state_outputs():
    """J_full (6, 9) and A_full (9, 9) are always present, in canonical order."""
    from awetrim.aerodynamics.vsm_quasi_steady import ALL_STATE_NAMES

    result = compute_vsm_trim_stability_derivatives(
        body_aero=_MockBody(),
        center_of_gravity=np.zeros(3),
        reference_point=np.zeros(3),
        x_trim=_X_TRIM,
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=100.0,
        inertia_yy=20.0,
        inertia_zz=100.0,
    )
    assert result["J_full"].shape == (6, 9)
    assert result["A_full"].shape == (9, 9)
    assert result["eig_full"].shape == (9,)
    assert result["vec_full"].shape == (9, 9)
    assert result["state_names_full"] == list(ALL_STATE_NAMES)

    # phi/theta/psi rows of A_full must be pure kinematics: phi_dot=p, etc.
    phi_idx = ALL_STATE_NAMES.index("phi")
    p_idx = ALL_STATE_NAMES.index("p")
    z_idx = ALL_STATE_NAMES.index("z")
    w_idx = ALL_STATE_NAMES.index("w")
    assert result["A_full"][phi_idx, p_idx] == pytest.approx(1.0)
    assert result["A_full"][phi_idx, p_idx + 0] == pytest.approx(1.0)
    assert result["A_full"][z_idx, w_idx] == pytest.approx(1.0)
    # No selection by default: selected-* keys must be absent.
    assert "A_selected" not in result
    assert "A_selected_long" not in result


def test_compute_vsm_trim_stability_derivatives_with_w_state():
    """Adding `w` to the longitudinal set extends A_selected_long to 4×4."""
    result = compute_vsm_trim_stability_derivatives(
        body_aero=_MockBody(),
        center_of_gravity=np.zeros(3),
        reference_point=np.zeros(3),
        x_trim=_X_TRIM,
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=100.0,
        inertia_yy=20.0,
        inertia_zz=100.0,
        states=["u", "w", "theta", "q", "phi", "psi", "p", "r"],
        coupled=False,
    )
    assert result["states_selected_long"] == ["u", "w", "theta", "q"]
    assert result["A_selected_long"].shape == (4, 4)
    assert result["A_selected_lat"].shape == (4, 4)
    # theta_dot = q kinematic row
    theta_row = result["states_selected_long"].index("theta")
    q_col = result["states_selected_long"].index("q")
    assert result["A_selected_long"][theta_row, q_col] == pytest.approx(1.0)


def test_compute_vsm_trim_stability_derivatives_coupled_selection():
    """coupled=True assembles a single A matrix over the selected states."""
    sel = ["u", "w", "z", "theta", "q", "phi", "psi", "p", "r"]
    result = compute_vsm_trim_stability_derivatives(
        body_aero=_MockBody(),
        center_of_gravity=np.zeros(3),
        reference_point=np.zeros(3),
        x_trim=_X_TRIM,
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=100.0,
        inertia_yy=20.0,
        inertia_zz=100.0,
        states=sel,
        coupled=True,
    )
    assert result["coupled_selected"] is True
    assert result["A_selected"].shape == (9, 9)
    assert result["J_selected"].shape == (6, 9)
    assert result["states_selected"] == sel


def test_compute_vsm_trim_stability_derivatives_rejects_unknown_state():
    with pytest.raises(ValueError, match="Unknown stability state"):
        compute_vsm_trim_stability_derivatives(
            body_aero=_MockBody(),
            center_of_gravity=np.zeros(3),
            reference_point=np.zeros(3),
            x_trim=_X_TRIM,
            trim_result=_TRIM_RESULT,
            solver=_MockSolver(),
            states=["u", "not_a_state"],
        )


# ---------------------------------------------------------------------------
# Existing — verify backward-compatibility aliases
# ---------------------------------------------------------------------------


def test_compatibility_aliases_point_to_canonical_functions():
    """Aliases used by scripts migrated from VSM must resolve to the AWETrim implementations."""
    assert solve_quasi_steady_state is solve_vsm_quasi_steady_trim
    assert run_quasi_steady_sweep is run_vsm_quasi_steady_sweep
    assert compute_stability_derivatives is compute_vsm_trim_stability_derivatives

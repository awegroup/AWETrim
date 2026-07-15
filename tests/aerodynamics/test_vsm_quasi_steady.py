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
    turn_radius_vs_steer_moment,
    # Backward-compatibility aliases (used by migrated scripts)
    compute_stability_derivatives,
    run_quasi_steady_sweep,
    solve_quasi_steady_state,
    # Internals exercised directly for the Omega_C gyroscopic coupling
    _gyroscopic_rate_coupling,
    _course_transport_rate_axes,
    _compose_attitude_rotation,
    _strip_theory_added_mass,
    corotating_state_transform,
    ALL_STATE_NAMES,
    DEFAULT_AXES,
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
# Steering / turn-map API (applied moment + turn_radius_vs_steer_moment)
# ---------------------------------------------------------------------------


def test_solve_trim_accepts_applied_moment_nm():
    """The steering injection is an optional, backward-compatible parameter."""
    sig = inspect.signature(solve_vsm_quasi_steady_trim)
    assert "applied_moment_nm" in sig.parameters
    assert sig.parameters["applied_moment_nm"].default is None


def test_turn_radius_vs_steer_moment_signature():
    sig = inspect.signature(turn_radius_vs_steer_moment)
    params = list(sig.parameters)
    assert params[:6] == [
        "body_aero", "center_of_gravity", "reference_point", "system_model",
        "steer_moments_nm", "x_guess",
    ]
    assert "steer_gain_nm_per_us" in sig.parameters
    assert sig.parameters["include_gravity"].default is False


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
    """J_full (6, 10) and A_full (10, 10) are always present, in canonical
    order — the 10 states include the lateral velocity ``v``."""
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
    assert result["J_full"].shape == (6, 10)
    assert result["A_full"].shape == (10, 10)
    assert result["eig_full"].shape == (10,)
    assert result["vec_full"].shape == (10, 10)
    assert result["state_names_full"] == list(ALL_STATE_NAMES)
    assert "v" in ALL_STATE_NAMES

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


def test_compute_vsm_trim_stability_derivatives_gyroscopic_coupling():
    """A_full carries the gyroscopic term omega x I_B omega on the rate rows.

    Since the B-point rework the gyroscopic term is part of the differenced
    right-hand side (it lives in ``eval_force_moment``), so it reaches A_full
    through J_full rather than through an analytic addition. With the mock
    solver returning a constant aerodynamic result, a ZERO CG offset (so the
    mass matrix is block-diagonal, I_B == I_cg, and no offset-force terms
    appear), and ``Omega_C = -course_rate * e_radial`` (spin rate
    ``w = -course_rate`` about the radial axis), the classical linearised
    Euler closed forms must be recovered exactly (central differences are
    exact for the quadratic term):
        A[p, q] = w (Iyy - Izz) / Ixx = course_rate * (Izz - Iyy) / Ixx
        A[q, p] = w (Izz - Ixx) / Iyy = course_rate * (Ixx - Izz) / Iyy
    with the r row (and the p/q <-> r entries) untouched.
    """
    course_rate = 0.3
    ixx, iyy, izz = 80.0, 20.0, 100.0
    result = compute_vsm_trim_stability_derivatives(
        body_aero=_MockBody(),
        center_of_gravity=np.zeros(3),
        reference_point=np.zeros(3),
        x_trim=np.array([20.0, 0.0, 0.0, 0.0, course_rate]),
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=ixx,
        inertia_yy=iyy,
        inertia_zz=izz,
    )
    names = list(result["state_names_full"])
    p, q, r = names.index("p"), names.index("q"), names.index("r")
    A = result["A_full"]
    assert A[p, q] == pytest.approx(course_rate * (izz - iyy) / ixx)
    assert A[q, p] == pytest.approx(course_rate * (ixx - izz) / iyy)
    # Radial-only Omega_C does not couple the yaw rate, and has no self-damping.
    assert A[p, r] == pytest.approx(0.0)
    assert A[q, r] == pytest.approx(0.0)
    assert A[r, p] == pytest.approx(0.0)
    assert A[r, q] == pytest.approx(0.0)
    assert A[p, p] == pytest.approx(0.0)
    assert A[q, q] == pytest.approx(0.0)

    # A zero course rate recovers the torque-free rate rows (no coupling).
    result0 = compute_vsm_trim_stability_derivatives(
        body_aero=_MockBody(),
        center_of_gravity=np.zeros(3),
        reference_point=np.zeros(3),
        x_trim=np.array([20.0, 0.0, 0.0, 0.0, 0.0]),
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=ixx,
        inertia_yy=iyy,
        inertia_zz=izz,
    )
    A0 = result0["A_full"]
    assert A0[p, q] == pytest.approx(0.0)
    assert A0[q, p] == pytest.approx(0.0)


def test_compute_vsm_trim_stability_derivatives_bpoint_coupling():
    """A nonzero CG offset couples the translational and rotational channels.

    The coupled B-point mass matrix ``[[m 1, -m c_x], [m c_x, I_B]]`` and the
    offset inertial terms (centripetal force, gravity/transport moments) make
    the force columns feed the rate rows and vice versa. A_full must equal the
    documented assembly ``solve(M6, J_full)`` plus kinematic rows, and the
    u-row must pick up rate-column entries that a point-mass model cannot
    have.
    """
    course_rate = 0.3
    result = compute_vsm_trim_stability_derivatives(
        body_aero=_MockBody(),
        center_of_gravity=np.array([0.5, 0.0, 0.5]),
        reference_point=np.zeros(3),
        x_trim=np.array([20.0, 0.0, 0.0, 0.0, course_rate]),
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=100.0,
        inertia_yy=20.0,
        inertia_zz=100.0,
    )
    names = list(result["state_names_full"])
    A = np.asarray(result["A_full"], dtype=float)
    J = np.asarray(result["J_full"], dtype=float)
    accel = np.linalg.solve(np.asarray(result["mass_matrix_axes"], float), J)
    assert A[names.index("u"), :] == pytest.approx(accel[0, :])
    assert A[names.index("w"), :] == pytest.approx(accel[2, :])
    assert A[names.index("p"), :] == pytest.approx(accel[3, :])
    # I_B carries the parallel-axis shift of the CG offset.
    c = np.asarray(result["cg_offset_trim_world"], dtype=float)
    expected_ib = np.diag([100.0, 20.0, 100.0]) + 15.0 * (
        float(c @ c) * np.eye(3) - np.outer(c, c)
    )
    assert np.asarray(result["inertia_b_axes"]) == pytest.approx(expected_ib)
    # The gyroscopic/centripetal rate columns now reach the u-row through the
    # coupled solve: at least one rate-column entry must be nonzero.
    rate_cols = [names.index(s) for s in ("p", "q", "r")]
    assert np.max(np.abs(A[names.index("u"), rate_cols])) > 0.0


def test_strip_theory_added_mass_flat_wing_closed_form():
    """Single flat panel (chord 1, span 10, normal +z, mid-chord at x=0.5).

    Strip theory entrains ``m = rho pi c^2/4 * span`` along the normal only:
    the translational block is ``m e_z e_z^T`` (no in-plane entrainment), the
    pitch inertia about the origin is ``m (c/2)^2``, and the heave-pitch
    coupling is ``-m c/2`` — the rank-1 mass matrix of a point mass at the
    mid-chord constrained to move along the panel normal.
    """
    rho = 1.225
    m = rho * np.pi / 4.0 * 1.0**2 * 10.0
    Ma = _strip_theory_added_mass(_MockBody(), np.zeros(3), rho=rho)

    expected = np.zeros((6, 6))
    expected[2, 2] = m
    expected[2, 4] = expected[4, 2] = -0.5 * m
    expected[4, 4] = 0.25 * m
    assert Ma == pytest.approx(expected)
    # Symmetric and positive semi-definite by construction.
    assert Ma == pytest.approx(Ma.T)
    assert float(np.min(np.linalg.eigvalsh(Ma))) >= -1e-12


def test_compute_vsm_trim_stability_derivatives_added_mass_flag():
    """``include_added_mass`` augments the mass matrix and nothing else.

    Off (default): historical rigid-only behaviour, zero recorded block,
    model "none". On: ``mass_matrix_axes`` gains exactly the strip-theory
    matrix (zero trim attitude -> no rotation) and ``A_full`` still equals
    the documented ``solve(M6 + M_a, J_full)`` assembly.
    """
    kwargs = dict(
        body_aero=_MockBody(),
        center_of_gravity=np.array([0.5, 0.0, 0.5]),
        reference_point=np.zeros(3),
        x_trim=np.array([20.0, 0.0, 0.0, 0.0, 0.3]),
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=100.0,
        inertia_yy=20.0,
        inertia_zz=100.0,
    )
    rigid = compute_vsm_trim_stability_derivatives(**kwargs)
    with_ma = compute_vsm_trim_stability_derivatives(
        **kwargs, include_added_mass=True
    )

    assert rigid["added_mass_model"] == "none"
    assert np.asarray(rigid["added_mass_matrix_axes"]) == pytest.approx(
        np.zeros((6, 6))
    )

    Ma = _strip_theory_added_mass(_MockBody(), np.zeros(3), rho=_MockSolver.rho)
    assert with_ma["added_mass_model"] == "strip_theory"
    assert np.asarray(with_ma["added_mass_matrix_axes"]) == pytest.approx(Ma)
    assert np.asarray(with_ma["mass_matrix_axes"]) == pytest.approx(
        np.asarray(rigid["mass_matrix_axes"]) + Ma
    )

    names = list(with_ma["state_names_full"])
    A = np.asarray(with_ma["A_full"], dtype=float)
    accel = np.linalg.solve(
        np.asarray(with_ma["mass_matrix_axes"], dtype=float),
        np.asarray(with_ma["J_full"], dtype=float),
    )
    assert A[names.index("u"), :] == pytest.approx(accel[0, :])
    assert A[names.index("q"), :] == pytest.approx(accel[4, :])

    # The exposed per-attitude mass matrix is the one nonlinear_rhs solves:
    # at zero perturbation (identity axes, zero trim attitude) it equals the
    # assembled trim matrix, and the trim RHS has an anchored force channel.
    M_world = with_ma["mass_matrix_world_fn"]()
    assert M_world == pytest.approx(np.asarray(with_ma["mass_matrix_axes"]))
    rhs0 = np.asarray(with_ma["rhs_world_at_trim"], dtype=float)
    assert rhs0[:3] == pytest.approx(np.zeros(3))


def test_gyroscopic_rate_coupling_full_omega_c_couples_p_and_r():
    """The great-circle normal component of Omega_C couples p <-> r.

    G = -I^{-1} ([Omega_C]_x I - [I Omega_C]_x). For a diagonal inertia and
    Omega_C = (0, Omega_y, Omega_z) in the (course, normal, radial) basis the
    closed form is
        G[p, q] = -Omega_z (Izz - Iyy) / Ixx,
        G[p, r] = -Omega_y (Izz - Iyy) / Ixx,
        G[q, p] = -Omega_z (Ixx - Izz) / Iyy,
        G[r, p] = +Omega_y (Ixx - Iyy) / Izz,
    so the radial component couples p <-> q and the normal (great-circle)
    component couples p <-> r, with no q <-> r coupling and a zero diagonal.
    """
    ixx, iyy, izz = 80.0, 20.0, 100.0
    omega_y, omega_z = 0.25, -0.3
    G = _gyroscopic_rate_coupling(
        np.array([0.0, omega_y, omega_z]), np.diag([ixx, iyy, izz])
    )
    # p <-> r from the normal (great-circle) component.
    assert G[0, 2] == pytest.approx(-omega_y * (izz - iyy) / ixx)
    assert G[2, 0] == pytest.approx(+omega_y * (ixx - iyy) / izz)
    # p <-> q from the radial (turn) component.
    assert G[0, 1] == pytest.approx(-omega_z * (izz - iyy) / ixx)
    assert G[1, 0] == pytest.approx(-omega_z * (ixx - izz) / iyy)
    # No q <-> r coupling and no self-damping (Omega has no course component).
    assert G[1, 2] == pytest.approx(0.0)
    assert G[2, 1] == pytest.approx(0.0)
    assert np.allclose(np.diag(G), 0.0)


def test_compute_vsm_trim_stability_derivatives_rotates_inertia_by_trim():
    """The principal inertia is rotated into the stability frame by the trim attitude.

    In the course frame the kite's principal axes are tilted from the stability
    axes by the trim attitude, so I_stab = R diag(I) R^T picks up off-diagonal
    products of inertia. Disabling the rotation recovers the diagonal set.
    """
    ixx, iyy, izz = 100.0, 20.0, 120.0
    roll, pitch, yaw = 3.0, -4.0, 8.0  # degrees
    common = dict(
        body_aero=_MockBody(),
        center_of_gravity=np.array([0.5, 0.0, 0.5]),
        reference_point=np.zeros(3),
        x_trim=np.array([20.0, roll, pitch, yaw, 0.0]),
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
        inertia_xx=ixx,
        inertia_yy=iyy,
        inertia_zz=izz,
    )

    result = compute_vsm_trim_stability_derivatives(**common)
    I_stab = np.asarray(result["inertia_stability"])
    assert result["inertia_rotated_by_trim"] is True
    R = _compose_attitude_rotation(
        roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw, axes=DEFAULT_AXES
    )
    assert np.allclose(I_stab, R @ np.diag([ixx, iyy, izz]) @ R.T)
    assert np.allclose(I_stab, I_stab.T)  # symmetric
    # Nonzero trim attitude -> genuine products of inertia (off-diagonal terms).
    assert not np.allclose(I_stab, np.diag(np.diag(I_stab)))

    result_diag = compute_vsm_trim_stability_derivatives(
        **common, rotate_inertia_by_trim=False
    )
    assert result_diag["inertia_rotated_by_trim"] is False
    assert np.allclose(result_diag["inertia_stability"], np.diag([ixx, iyy, izz]))


def test_compute_vsm_trim_stability_derivatives_accepts_full_inertia_tensor():
    """``inertia_cg`` carries the full CG tensor, overriding the scalars.

    A diagonal ``inertia_cg`` must reproduce the principal-scalar path
    exactly; a tensor with a roll-yaw product of inertia must carry it into
    ``inertia_stability`` (rotated by the trim attitude like the geometry).
    """
    ixx, iyy, izz, ixz = 100.0, 20.0, 120.0, 8.0
    roll, pitch, yaw = 3.0, -4.0, 8.0  # degrees
    common = dict(
        body_aero=_MockBody(),
        center_of_gravity=np.array([0.5, 0.0, 0.5]),
        reference_point=np.zeros(3),
        x_trim=np.array([20.0, roll, pitch, yaw, 0.0]),
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
        mass=15.0,
    )

    scalar = compute_vsm_trim_stability_derivatives(
        **common, inertia_xx=ixx, inertia_yy=iyy, inertia_zz=izz
    )
    tensor_diag = compute_vsm_trim_stability_derivatives(
        **common, inertia_cg=np.diag([ixx, iyy, izz])
    )
    assert np.allclose(tensor_diag["inertia_stability"], scalar["inertia_stability"])
    assert np.allclose(tensor_diag["A_full"], scalar["A_full"])

    full = np.array(
        [[ixx, 0.0, ixz], [0.0, iyy, 0.0], [ixz, 0.0, izz]], dtype=float
    )
    R = _compose_attitude_rotation(
        roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw, axes=DEFAULT_AXES
    )
    result = compute_vsm_trim_stability_derivatives(**common, inertia_cg=full)
    assert np.allclose(result["inertia_stability"], R @ full @ R.T)

    # The principal scalars are ignored when the full tensor is given.
    ignored = compute_vsm_trim_stability_derivatives(
        **common, inertia_cg=full, inertia_xx=1.0, inertia_yy=1.0, inertia_zz=1.0
    )
    assert np.allclose(ignored["inertia_stability"], R @ full @ R.T)


def test_corotating_state_transform_structure():
    """T maps frozen-axes to co-rotating-axes perturbation states.

    Velocity rows pick up [v_trim]_x on the attitude columns, rate rows pick
    up [Omega_C]_x; everything else is identity. T is unipotent, so
    inv(T) = 2I - T exactly, and any similarity T A inv(T) preserves the
    spectrum.
    """
    v_trim = np.array([20.0, 0.0, -2.0])
    omega_c = np.array([0.0, 0.1, -0.3])
    T = corotating_state_transform(v_trim, omega_c)
    n = len(ALL_STATE_NAMES)
    assert T.shape == (n, n)

    idx = {s: k for k, s in enumerate(ALL_STATE_NAMES)}
    att = [idx["phi"], idx["theta"], idx["psi"]]

    def _skew_ref(v):
        return np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
        )

    assert np.allclose(
        T[np.ix_([idx["u"], idx["v"], idx["w"]], att)], _skew_ref(v_trim)
    )
    assert np.allclose(
        T[np.ix_([idx["p"], idx["q"], idx["r"]], att)], _skew_ref(omega_c)
    )
    # All other entries are the identity.
    T_stripped = T.copy()
    T_stripped[np.ix_([idx["u"], idx["v"], idx["w"]], att)] = 0.0
    T_stripped[np.ix_([idx["p"], idx["q"], idx["r"]], att)] = 0.0
    assert np.allclose(T_stripped, np.eye(n))
    # Unipotent: the attitude coupling is nilpotent.
    assert np.allclose(T @ (2.0 * np.eye(n) - T), np.eye(n))


def test_compute_vsm_trim_stability_derivatives_corotating_outputs():
    """The result dict carries the co-rotating transform, consistent with the
    recorded v_kite_trim_axes / omega_c_axes, and the similarity it defines
    leaves the spectrum untouched."""
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
    # Minimal trim_result has no kite_vel_world and no system_model: the
    # fallback is the tangential speed along the course axis.
    v_axes = np.asarray(result["v_kite_trim_axes"], dtype=float)
    assert np.allclose(v_axes, [_X_TRIM[0], 0.0, 0.0])

    T = np.asarray(result["T_corotating_from_frozen"], dtype=float)
    assert np.allclose(
        T, corotating_state_transform(v_axes, result["omega_c_axes"])
    )

    A_frozen = np.asarray(result["A_full"], dtype=float)
    A_corot = T @ A_frozen @ np.linalg.inv(T)
    assert np.allclose(
        np.sort_complex(np.linalg.eigvals(A_corot)),
        np.sort_complex(np.asarray(result["eig_full"], dtype=complex)),
    )


def test_compute_vsm_trim_stability_derivatives_rejects_bad_inertia_cg():
    """Wrong shape and asymmetric tensors are rejected before any solve."""
    common = dict(
        body_aero=_MockBody(),
        center_of_gravity=np.zeros(3),
        reference_point=np.zeros(3),
        x_trim=_X_TRIM,
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
    )
    with pytest.raises(ValueError, match="3x3"):
        compute_vsm_trim_stability_derivatives(**common, inertia_cg=np.zeros(3))
    with pytest.raises(ValueError, match="symmetric"):
        compute_vsm_trim_stability_derivatives(
            **common,
            inertia_cg=np.array(
                [[100.0, 0.0, 5.0], [0.0, 20.0, 0.0], [0.0, 0.0, 120.0]]
            ),
        )


def test_course_transport_rate_axes_reduced_without_system_model():
    """Without a system model (or with full=False) the radial-only Omega_C is used."""
    axes_body, axes_world, is_full = _course_transport_rate_axes(
        None, DEFAULT_AXES, course_rate=0.3, speed_tangential=20.0, full=True
    )
    assert is_full is False
    assert np.allclose(axes_body, [0.0, 0.0, -0.3])
    assert np.allclose(axes_world, [0.0, 0.0, -0.3])


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
# New — numerical-hygiene diagnostics of the finite-difference solves
# ---------------------------------------------------------------------------


def test_stability_derivatives_hygiene_diagnostics_graceful_defaults():
    """Mocks without gamma support / alpha_at_ac / panel polars opt out.

    The warm-start machinery must not require the solver to accept a
    ``gamma_distribution`` keyword, and the linearisation-point stall check
    must degrade to NaN/None when the solver or body exposes no stall
    information.
    """
    result = compute_vsm_trim_stability_derivatives(
        body_aero=_MockBody(),
        center_of_gravity=np.zeros(3),
        reference_point=np.zeros(3),
        x_trim=_X_TRIM,
        trim_result=_TRIM_RESULT,
        solver=_MockSolver(),
    )
    assert result["perturbation_solves_converged"] is True
    assert result["n_unconverged_perturbation_solves"] == 0
    assert result["gamma_warm_start_used"] is False
    assert np.isnan(result["stall_margin_min_deg_at_trim"])
    assert result["n_stalled_panels_at_trim"] is None


class _MockStallPanel(_MockPanel):
    def __init__(self, onset_rad: float):
        super().__init__()
        alpha = np.deg2rad(np.arange(-5.0, 26.0, 1.0))
        cl = 1.5 - 5.0 * (alpha - onset_rad) ** 2  # interior peak at onset
        self.panel_polar_data = np.column_stack(
            [alpha, cl, np.full(alpha.size, 0.02), np.zeros(alpha.size)]
        )


class _MockGammaSolver(_MockSolver):
    """Accepts a warm start and reports circulation + effective alphas."""

    def __init__(self):
        self.warm_started_calls = 0
        self.calls = 0

    def solve(self, body, gamma_distribution=None) -> dict:
        self.calls += 1
        if gamma_distribution is not None:
            self.warm_started_calls += 1
        res = dict(super().solve(body))
        res["gamma_distribution"] = np.zeros(1)
        res["gamma_converged"] = True
        res["alpha_at_ac"] = np.array([[np.deg2rad(8.0)]])  # (n, 1) like VSM
        return res


def test_stability_derivatives_warm_start_and_trim_stall_check():
    body = _MockBody()
    body.panels = [_MockStallPanel(onset_rad=np.deg2rad(12.0))]
    solver = _MockGammaSolver()
    result = compute_vsm_trim_stability_derivatives(
        body_aero=body,
        center_of_gravity=np.zeros(3),
        reference_point=np.zeros(3),
        x_trim=_X_TRIM,
        trim_result=_TRIM_RESULT,
        solver=solver,
    )
    assert result["gamma_warm_start_used"] is True
    # Baseline solve is cold; every finite-difference solve is warm-started.
    assert solver.warm_started_calls == solver.calls - 1
    assert result["perturbation_solves_converged"] is True
    # alpha_eff 8 deg vs onset 12 deg -> 4 deg margin, no stalled panels.
    assert result["stall_margin_min_deg_at_trim"] == pytest.approx(4.0)
    assert result["n_stalled_panels_at_trim"] == 0


# ---------------------------------------------------------------------------
# Existing — verify backward-compatibility aliases
# ---------------------------------------------------------------------------


def test_compatibility_aliases_point_to_canonical_functions():
    """Aliases used by scripts migrated from VSM must resolve to the AWETrim implementations."""
    assert solve_quasi_steady_state is solve_vsm_quasi_steady_trim
    assert run_quasi_steady_sweep is run_vsm_quasi_steady_sweep
    assert compute_stability_derivatives is compute_vsm_trim_stability_derivatives

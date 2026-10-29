"""Tests for awetrim.kinematics.Kinematics.ParametrizedKinematics.

Consolidated from the three former files:
  test_parametrized_kinematics_symbolic.py   — vtau relation
  test_parametrized_kinematics_more.py       — dot_vtau and chi expressions
  test_parametrized_kinematics_all_eqs.py    — full kinematic equation suite
"""

import casadi as ca

from awetrim.kinematics.Kinematics import ParametrizedKinematics


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


class DummyPattern:
    def elevation(self, r, s):
        return 0.2 + 0.001 * r + 0.01 * s

    def azimuth(self, r, s):
        return 0.1 + 0.002 * r + 0.02 * s


class DummyKiteModel:
    def __init__(self, distance_radial, speed_radial):
        self.distance_radial = distance_radial
        self.speed_radial = speed_radial


class DummyPhase:
    def __init__(self, s, kite_model, s_dot, s_ddot):
        self.s = s
        self.kite_model = kite_model
        self.s_dot = s_dot
        self.s_ddot = s_ddot


def _make_pk():
    """Return (pk, symbols) for a ParametrizedKinematics with symbolic inputs."""
    s = ca.MX.sym("s")
    r = ca.MX.sym("r")
    vr = ca.MX.sym("vr")
    s_dot = ca.MX.sym("s_dot")
    s_ddot = ca.MX.sym("s_ddot")

    pattern = DummyPattern()
    kite_model = DummyKiteModel(r, vr)
    phase = DummyPhase(s, kite_model, s_dot, s_ddot)
    pk = ParametrizedKinematics(pattern, phase)

    return pk, (s, r, vr, s_dot, s_ddot)


_NUMERIC_VALS = (0.5, 80.0, 0.8, 0.25, 0.01)


def _assert_close(val1, val2, tol=1e-8):
    assert abs(float(val1) - float(val2)) < tol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vtau_relation():
    """vk² = vr² + vtau² must hold for all valid inputs."""
    pk, syms = _make_pk()
    s, r, vr, s_dot, s_ddot = syms

    expr = pk.vk**2 - pk.vr**2 - pk.vtau**2
    f = ca.Function("check", [s, r, vr, s_dot, s_ddot], [expr])

    val = float(f(0.5, 50.0, 1.0, 0.2, 0.0)[0])
    assert abs(val) < 1e-6


def test_dot_vtau_and_chi_expressions():
    """dot_vtau and chi match their manually derived equivalents."""
    pk, syms = _make_pk()
    s, r, vr, s_dot, s_ddot = syms
    pattern = DummyPattern()

    expr_dot_vtau_manual = pk.sqrt_A * (
        pk.s_dot**2 * pk.dr_ds + pk.s_ddot * pk.r
    ) + pk.s_dot * pk.r * pk.dot_A / (2 * pk.sqrt_A)

    phi_expr = pattern.azimuth(r, s)
    beta_expr = pattern.elevation(r, s)
    dphi_manual = ca.gradient(phi_expr, s) + ca.gradient(phi_expr, r) * vr / s_dot
    dbeta_manual = ca.gradient(beta_expr, s) + ca.gradient(beta_expr, r) * vr / s_dot
    expr_chi_manual = ca.atan2(dphi_manual * ca.cos(beta_expr), dbeta_manual)

    f_dot = ca.Function("f_dot", list(syms), [pk.dot_vtau - expr_dot_vtau_manual])
    f_chi = ca.Function("f_chi", list(syms), [pk.chi - expr_chi_manual])

    _assert_close(f_dot(*_NUMERIC_VALS)[0], 0.0)
    _assert_close(f_chi(*_NUMERIC_VALS)[0], 0.0)


def test_all_kinematics_equations():
    """Every ParametrizedKinematics output matches its manual derivation."""
    pk, syms = _make_pk()
    s, r, vr, s_dot, s_ddot = syms
    pattern = DummyPattern()

    phi_expr = pattern.azimuth(r, s)
    beta_expr = pattern.elevation(r, s)

    dphi_ds = ca.gradient(phi_expr, s) + ca.gradient(phi_expr, r) * vr / s_dot
    dbeta_ds = ca.gradient(beta_expr, s) + ca.gradient(beta_expr, r) * vr / s_dot
    dr_ds = vr / s_dot

    dR_ds_manual = ca.vertcat(
        r * dphi_ds * ca.cos(beta_expr), r * dbeta_ds, dr_ds
    )
    vk_manual = ca.norm_2(dR_ds_manual) * s_dot
    vtau_manual = ca.sqrt(vk_manual**2 - vr**2)

    dr_ds2_manual = ca.gradient(dr_ds, s)
    dbeta_ds2_manual = (
        ca.gradient(dbeta_ds, s) + ca.gradient(dbeta_ds, r) * vr / s_dot
    )
    dphi_ds2_manual = (
        ca.gradient(dphi_ds, s) + ca.gradient(dphi_ds, r) * vr / s_dot
    )

    sqrt_A_manual = vtau_manual / (s_dot * r)
    dot_A_manual = (
        2
        * s_dot
        * (
            dbeta_ds * dbeta_ds2_manual
            + dphi_ds * dphi_ds2_manual * ca.cos(beta_expr) ** 2
            - dphi_ds**2 * dbeta_ds * ca.sin(beta_expr) * ca.cos(beta_expr)
        )
    )
    dot_vr_manual = dr_ds2_manual * s_dot**2 + s_ddot * dr_ds
    dot_vtau_manual = sqrt_A_manual * (
        s_dot**2 * dr_ds + s_ddot * r
    ) + s_dot * r * dot_A_manual / (2 * sqrt_A_manual)
    chi_manual = ca.atan2(dphi_ds * ca.cos(beta_expr), dbeta_ds)
    dot_chi_manual = (
        ca.gradient(chi_manual, s) * s_dot + ca.gradient(chi_manual, r) * vr
    )

    inputs = list(syms)
    checks = {
        "dR_ds":     (pk.dR_ds,     dR_ds_manual),
        "vk":        (pk.vk,        vk_manual),
        "vtau":      (pk.vtau,      vtau_manual),
        "dr_ds2":    (pk.dr_ds2,    dr_ds2_manual),
        "dbeta_ds2": (pk.dbeta_ds2, dbeta_ds2_manual),
        "sqrt_A":    (pk.sqrt_A,    sqrt_A_manual),
        "dot_A":     (pk.dot_A,     dot_A_manual),
        "dot_vr":    (pk.dot_vr,    dot_vr_manual),
        "dot_vtau":  (pk.dot_vtau,  dot_vtau_manual),
        "chi":       (pk.chi,       chi_manual),
        "dot_chi":   (pk.dot_chi,   dot_chi_manual),
    }

    for name, (impl, manual) in checks.items():
        f = ca.Function(f"f_{name}", inputs, [impl - manual])
        _assert_close(f(*_NUMERIC_VALS)[0], 0.0)

    # extract_function must return a callable
    fk = pk.extract_function("vk")
    assert callable(fk)


# ---------------------------------------------------------------------------
# Geodesic curvature of the pattern (turn radius on the tether sphere)
# ---------------------------------------------------------------------------


class _SmallCirclePattern:
    """Small circle of angular radius rho about (phi_c, beta_c) on the unit sphere."""

    def __init__(self, rho, phi_c, beta_c):
        import numpy as np

        c = np.array(
            [np.cos(beta_c) * np.cos(phi_c), np.cos(beta_c) * np.sin(phi_c), np.sin(beta_c)]
        )
        e1 = np.cross([0.0, 0.0, 1.0], c)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(c, e1)
        self.rho, self.c, self.e1, self.e2 = rho, ca.DM(c), ca.DM(e1), ca.DM(e2)

    def _p(self, s):
        return ca.cos(self.rho) * self.c + ca.sin(self.rho) * (
            ca.cos(s) * self.e1 + ca.sin(s) * self.e2
        )

    def azimuth(self, r, s):
        p = self._p(s)
        return ca.atan2(p[1], p[0])

    def elevation(self, r, s):
        return ca.asin(self._p(s)[2])


class _ParallelPattern:
    def __init__(self, beta0):
        self.beta0 = beta0

    def azimuth(self, r, s):
        return s

    def elevation(self, r, s):
        return self.beta0 + 0.0 * s


class _MeridianPattern:
    def azimuth(self, r, s):
        return 0.3 + 0.0 * s

    def elevation(self, r, s):
        return 0.1 + 0.5 * s


def _curvature_function(pattern):
    s = ca.MX.sym("s")
    r = ca.MX.sym("r")
    vr = ca.MX.sym("vr")
    s_dot = ca.MX.sym("s_dot")
    pk = ParametrizedKinematics(pattern, DummyPhase(s, DummyKiteModel(r, vr), s_dot, 0.0))
    return ca.Function("kappa", [s, r, vr, s_dot], [pk.curvature_geodesic])


def test_curvature_geodesic_small_circle_is_cot_rho():
    """A small circle of angular radius rho anywhere on the sphere has geodesic
    curvature cot(rho); with r = 300 m and rho = 11.35/300 the physical turn
    radius r/|kappa| = r tan(rho) ~ 11.355 m."""
    import numpy as np

    rho = 11.35 / 300.0
    f = _curvature_function(_SmallCirclePattern(rho, 0.4, 0.5))
    for sv in np.linspace(0.0, 2 * np.pi, 7):
        kappa = float(f(sv, 300.0, 1.0, 0.5)[0])
        assert abs(abs(kappa) - 1.0 / np.tan(rho)) < 1e-6 * (1.0 / np.tan(rho))
        assert abs(300.0 / abs(kappa) - 300.0 * np.tan(rho)) < 1e-6


def test_curvature_geodesic_parallel_and_meridian():
    """A parallel at latitude beta0 has |kappa| = tan(beta0) (its curvature is
    pure transport), a meridian (great circle) has kappa = 0."""
    import numpy as np

    f_par = _curvature_function(_ParallelPattern(0.5))
    assert abs(abs(float(f_par(1.0, 200.0, 0.0, 1.0)[0])) - np.tan(0.5)) < 1e-9
    f_mer = _curvature_function(_MeridianPattern())
    assert abs(float(f_mer(0.7, 200.0, 0.0, 1.0)[0])) < 1e-9


def test_curvature_numerator_matches_kappa_times_dsigma_cubed():
    """The division-free polynomial numerator equals kappa * sigma'^3 (used by
    the NLP turn-radius rows), checked on a generic r-dependent pattern."""
    pk, syms = _make_pk()
    expr = pk.curvature_numerator - pk.curvature_geodesic * pk.dsigma_ds**3
    f = ca.Function("chk", list(syms), [expr, pk.curvature_geodesic])
    for vals in (_NUMERIC_VALS, (1.3, 120.0, -0.5, 0.7, 0.0), (2.0, 300.0, 2.0, 0.05, 0.1)):
        diff, kappa = f(*vals)
        assert abs(float(diff)) < 1e-9 * max(1.0, abs(float(kappa)))

"""The ``xp`` section-force kernels reproduce VSM's own force assembly.

These are the single-source guard for :mod:`awetrim.aerodynamics.panel_kernels`:
each kernel is pinned to the numbers VSM's ``BodyAerodynamics.compute_results``
/ ``compute_line_aerodynamic_force`` produce, in the NumPy path and in the
CasADi path (evaluated on ``DM`` inputs), on a rotating body with sideslip so
no term is trivially zero. A synthetic bridle exercises the line kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")
VSM_Wing = pytest.importorskip("VSM.core.WingGeometry").Wing
from VSM.core.BodyAerodynamics import BodyAerodynamics  # noqa: E402
from VSM.core.Solver import Solver  # noqa: E402

from awetrim.aerodynamics import panel_kernels as pk  # noqa: E402
from awetrim.aerodynamics.apparent_wind import apparent_wind_at  # noqa: E402


def _polar():
    alpha = np.deg2rad(np.arange(-10.0, 31.0, 1.0))
    cl = 2 * np.pi * alpha
    cd = 0.01 + 0.02 * alpha**2
    cm = -0.05 - 0.1 * alpha
    return np.column_stack((alpha, cl, cd, cm))


@pytest.fixture(scope="module")
def solved_body():
    """Tapered, twisted, anhedral wing with a bridle: VSM solved once."""
    n = 10
    wing = VSM_Wing(n_panels=n, spanwise_panel_distribution="uniform")
    for y in np.linspace(-4.0, 4.0, n + 1):
        chord = 1.2 - 0.05 * abs(y)
        twist = np.deg2rad(2.0 * y / 4.0)
        z = 0.1 * y**2
        le = np.array([0.0, y, z])
        te = le + chord * np.array([np.cos(twist), 0.0, -np.sin(twist)])
        wing.add_section(le, te, _polar())
    bridle = [
        [np.array([0.3, -2.0, 0.0]), np.array([0.4, 0.0, -4.0]), 0.004],
        [np.array([0.3, 2.0, 0.0]), np.array([0.4, 0.0, -4.0]), 0.004],
        [np.array([0.9, 0.0, 0.0]), np.array([0.4, 0.0, -4.0]), 0.003],
    ]
    body = BodyAerodynamics([wing], bridle_line_system=bridle)
    omega = np.array([0.05, 0.3, 0.1])
    body.va_initialize(
        Umag=15.0, angle_of_attack=8.0, side_slip=4.0,
        body_rates=float(np.linalg.norm(omega)), body_axis=omega / np.linalg.norm(omega),
        reference_point=np.zeros(3), rates_in_body_frame=True,
    )
    solver = Solver(gamma_loop_type="casadi_newton", allowed_error=1e-10)
    results = solver.solve(body)
    assert results["gamma_converged"]
    return body, solver, results


@pytest.mark.parametrize("use_casadi", [False, True])
def test_panel_forces_and_totals_match_vsm(solved_body, use_casadi):
    body, s, res = solved_body
    xp = casadi if use_casadi else np
    X = (lambda a: casadi.DM(a)) if use_casadi else (lambda a: a)
    gamma = np.asarray(res["gamma_distribution"], dtype=float)
    v_rel = s.va_array + np.column_stack([a @ gamma for a in (s.AIC_x, s.AIC_y, s.AIC_z)])

    alpha, umag = pk.relative_flow(X(v_rel), X(s.x_airf_array), X(s.y_airf_array), X(s.z_airf_array), xp)
    alpha_np = np.asarray(alpha).ravel()
    # the kernel's alpha/umag ARE VSM's
    alpha_vsm, umag_vsm, _, _ = s.compute_aerodynamic_quantities(gamma)
    np.testing.assert_allclose(alpha_np, alpha_vsm, atol=1e-13)
    np.testing.assert_allclose(np.asarray(umag).ravel(), umag_vsm, atol=1e-12)

    cl = np.array([p.compute_cl(a) for p, a in zip(s.panels, alpha_np)])
    cdcm = np.array([p.compute_cd_cm(a) for p, a in zip(s.panels, alpha_np)])
    force, moment_local = pk.panel_forces(
        alpha, umag, X(cl), X(cdcm[:, 0]), X(cdcm[:, 1]), X(s.chord_array), X(s.width_array),
        X(s.x_airf_array), X(s.y_airf_array), X(s.z_airf_array),
        body.wings[0].spanwise_direction, s.rho, xp,
    )
    np.testing.assert_allclose(np.asarray(force), np.asarray(res["F_distribution"]), atol=1e-11)

    ac = np.array([p.aerodynamic_center for p in s.panels])
    f_tot, m_tot = pk.total_force_and_moment(force, moment_local, X(ac), xp)
    f_tot = np.asarray(f_tot).ravel()
    m_tot = np.asarray(m_tot).ravel()
    va_free, omega_b = np.asarray(body.va), np.asarray(body.body_rates)
    for line in body._bridle_line_system:
        p1, p2 = pk.order_line_endpoints(line[0], line[1])
        mid = 0.5 * (np.asarray(line[0]) + np.asarray(line[1]))
        va_seg = apparent_wind_at(va_free, omega_b, mid, np.zeros(3))
        f_seg = pk.line_force(X(va_seg), p1, p2, line[2], s.rho, 1.1, 0.01, xp)
        f_tot = f_tot + np.asarray(f_seg).ravel()
        m_tot = m_tot + np.cross(mid, np.asarray(f_seg).ravel())
    np.testing.assert_allclose(f_tot, [res["Fx"], res["Fy"], res["Fz"]], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(m_tot, [res["Mx"], res["My"], res["Mz"]], rtol=1e-12, atol=1e-9)

    # VSM's lift/drag/side bookkeeping (local inflow frame per panel); on this
    # rotating body it is NOT the total force projected on the reference inflow.
    bridle = []
    for line in body._bridle_line_system:
        p1, p2 = pk.order_line_endpoints(line[0], line[1])
        mid = 0.5 * (np.asarray(line[0]) + np.asarray(line[1]))
        va_seg = apparent_wind_at(va_free, omega_b, mid, np.zeros(3))
        bridle.append(pk.line_force(X(va_seg), p1, p2, line[2], s.rho, 1.1, 0.01, xp))
    lift, drag, side = pk.lift_drag_side_vsm(force, X(s.va_array), X(va_free), body.wings[0].spanwise_direction, xp, bridle)
    np.testing.assert_allclose([float(lift), float(drag), float(side)], [res["lift"], res["drag"], res["side"]], rtol=1e-12, atol=1e-9)
    assert abs(float(drag) - float(np.dot(f_tot, va_free / np.linalg.norm(va_free)))) > 1e-3 * abs(float(drag))


def test_line_force_matches_vsm_kernel(solved_body):
    body, s, _ = solved_body
    va = np.array([12.0, -2.0, 3.0])
    for line in body._bridle_line_system:
        expected = body.compute_line_aerodynamic_force(va, line, rho=s.rho)
        p1, p2 = pk.order_line_endpoints(line[0], line[1])
        np.testing.assert_allclose(pk.line_force(va, p1, p2, line[2], s.rho), expected, atol=1e-12)
        sym = pk.line_force(casadi.DM(va), p1, p2, line[2], s.rho, xp=casadi)
        np.testing.assert_allclose(np.asarray(sym).ravel(), expected, atol=1e-12)


def test_apparent_wind_and_kcu_force_casadi_paths_match_numpy():
    from awetrim.aerodynamics.kcu_drag import force_drag_kcu

    va, omega, r, r0 = np.array([10.0, 1.0, -2.0]), np.array([0.1, -0.2, 0.3]), np.array([0.5, 0.2, -1.0]), np.zeros(3)
    expected = apparent_wind_at(va, omega, r, r0)
    sym = apparent_wind_at(casadi.DM(va), casadi.DM(omega), r, r0, xp=casadi)
    np.testing.assert_allclose(np.asarray(sym).ravel(), expected, atol=1e-13)

    axis = np.array([0.0, 0.0, 1.0])
    expected = force_drag_kcu(va, axis, 1.225, 0.15, 0.33)
    sym = force_drag_kcu(casadi.DM(va), axis, 1.225, 0.15, 0.33, xp=casadi)
    np.testing.assert_allclose(np.asarray(sym).ravel(), expected, atol=1e-12)


def test_kernels_are_differentiable_in_casadi():
    """The CasADi path builds a graph whose Jacobian matches finite differences
    of the NumPy path -- the property the CasADi trim relies on."""
    n = 3
    rng = np.random.default_rng(1)
    x_airf = np.tile([0.0, 0.0, 1.0], (n, 1))
    y_airf = np.tile([1.0, 0.0, 0.0], (n, 1))
    z_airf = np.tile([0.0, 1.0, 0.0], (n, 1))
    chord, width = np.full(n, 1.0), np.full(n, 0.5)
    v0 = np.column_stack([np.full(n, 10.0), rng.normal(0, 0.5, n), np.full(n, 1.5)])
    cl = np.full(n, 0.8)
    cd, cm = np.full(n, 0.02), np.full(n, -0.05)

    def total(v_rel_flat, xp):
        v_rel = xp.reshape(v_rel_flat, n, 3) if xp is casadi else v_rel_flat.reshape(n, 3)
        alpha, umag = pk.relative_flow(v_rel, x_airf, y_airf, z_airf, xp)
        force, moment_local = pk.panel_forces(alpha, umag, cl, cd, cm, chord, width, x_airf, y_airf, z_airf, [0.0, 1.0, 0.0], 1.225, xp)
        f, m = pk.total_force_and_moment(force, moment_local, y_airf * 0.25, xp)
        return xp.vertcat(f, m) if xp is casadi else np.concatenate([f, m])

    v = casadi.MX.sym("v", 3 * n)
    # CasADi reshape is column-major; hand it the matching flat layout.
    fn = casadi.Function("f", [v], [casadi.jacobian(total(v, casadi), v)])
    jac = np.asarray(fn(v0.ravel(order="F")))
    eps = 1e-6
    jac_fd = np.zeros_like(jac)
    flat = v0.ravel(order="F")
    for j in range(flat.size):
        e = np.zeros_like(flat)
        e[j] = eps
        jac_fd[:, j] = (total((flat + e).reshape(n, 3, order="F"), np) - total((flat - e).reshape(n, 3, order="F"), np)) / (2 * eps)
    np.testing.assert_allclose(jac, jac_fd, atol=1e-6, rtol=1e-6)

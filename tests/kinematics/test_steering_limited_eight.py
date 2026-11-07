"""The kite-sized seed: curvature kernels, course bounds, the spline fit."""
import casadi as ca
import numpy as np
import pytest

from awetrim.kinematics.parametrized_patterns import (
    count_self_crossings,
    fit_bspline_pattern_to_trajectory,
    geodesic_curvature_numerator,
    heading_rate_turn,
    periodic_bspline_basis_matrices,
)
from awetrim.kinematics.steering_limited_eight import (
    BlendedTurnLimit,
    CourseTurnLimit,
    fit_spline_to_turn_limit,
    spline_turn_geometry,
)


def circle_path(rho, direction=1.0, el_c=0.5, r0=200.0, M=40):
    """A small circle of angular radius ``rho``; ``direction`` +1 flies it
    with the course increasing."""
    s = np.linspace(0.0, 1.0, 800, endpoint=False)
    psi = direction * 2.0 * np.pi * s
    # Centre on the x axis tilted up by el_c, built in 3D so it is exact.
    q = np.stack(
        [np.cos(rho) * np.ones_like(psi), np.sin(rho) * np.sin(psi), np.sin(rho) * np.cos(psi)],
        axis=1,
    )
    rot = np.array(
        [[np.cos(el_c), 0.0, -np.sin(el_c)], [0.0, 1.0, 0.0], [np.sin(el_c), 0.0, np.cos(el_c)]]
    )
    q = q @ rot.T
    _, C_phi, C_beta = fit_bspline_pattern_to_trajectory(
        spline_type="periodic", M=M, s_init=0.0, s_final=1.0,
        az_target=np.arctan2(q[:, 1], q[:, 0]), el_target=np.arcsin(q[:, 2]),
        s_samples=s, downloops=True,
    )
    return {
        "r0": r0, "M": M, "C_phi": np.asarray(C_phi).ravel().tolist(),
        "C_beta": np.asarray(C_beta).ravel().tolist(),
        "s_init": 0.0, "s_final": 1.0, "downloops": True,
    }


@pytest.mark.parametrize("direction, sign", [(1.0, 1.0), (-1.0, -1.0)])
def test_turn_geometry_of_a_small_circle(direction, sign):
    # A small circle of angular radius rho has geodesic curvature cot(rho).
    rho = np.radians(10.0)
    r0 = 200.0
    _, radius, signs = spline_turn_geometry(circle_path(rho, direction, r0=r0), r0)
    assert radius == pytest.approx(r0 * np.tan(rho), rel=5e-3)  # spline ripple
    assert np.all(signs == sign)


def test_curvature_kernel_numpy_and_casadi_agree():
    values = (0.4, 0.3, -0.2, 0.1, 0.5, 0.7)
    numeric = geodesic_curvature_numerator(*values, np)
    symbolic = geodesic_curvature_numerator(*(ca.DM(v) for v in values), ca)
    assert float(symbolic) == pytest.approx(numeric)


def test_heading_rate_turn_removes_the_meridian_convergence():
    assert heading_rate_turn(0.3, 20.0, 200.0, 0.5, 0.0) == pytest.approx(0.3)
    assert heading_rate_turn(0.3, 20.0, 200.0, 0.5, np.pi / 2.0) == pytest.approx(
        0.3 - 0.1 * np.tan(0.5)
    )


def asymmetric_required(course, sign):
    """Heading down needs twice the radius of heading up; turn sign matters."""
    base = 20.0 + 10.0 * (1.0 - np.cos(np.asarray(course, dtype=float)))
    return base * np.where(np.asarray(sign) > 0, 1.0, 1.1)


def test_course_turn_limit_is_mirror_symmetric():
    limit = CourseTurnLimit(asymmetric_required, r0=200.0, margin=1.2)
    chi = np.radians([20.0, 75.0, 140.0, 200.0, 310.0])
    plus = limit.required_radius(chi, np.ones_like(chi))
    minus = limit.required_radius(-chi, -np.ones_like(chi))
    assert plus == pytest.approx(minus, rel=1e-9)
    # The stricter of the two directions, margin included.
    assert plus == pytest.approx(1.2 * 1.1 * (20.0 + 10.0 * (1.0 - np.cos(chi))), rel=2e-3)


def test_course_turn_limit_treats_no_turn_as_a_straight_line():
    def required(course, sign):
        return np.full_like(np.asarray(course, dtype=float), np.inf)

    limit = CourseTurnLimit(required, r0=200.0)
    assert limit.scale < 0.1  # curvature bound ~0, not NaN


def test_fit_rejects_an_up_loop_and_an_odd_m():
    limit = CourseTurnLimit(asymmetric_required, r0=200.0)
    with pytest.raises(ValueError, match="down-loop"):
        fit_spline_to_turn_limit({"r0": 200.0, "downloops": False}, limit)
    with pytest.raises(ValueError, match="even"):
        fit_spline_to_turn_limit({"r0": 200.0}, limit, M=15)


@pytest.mark.slow
def test_fit_holds_the_limit_on_the_spline_and_stays_an_eight():
    s = np.linspace(0.0, 1.0, 400, endpoint=False)
    psi = 2.0 * np.pi * s
    _, C_phi, C_beta = fit_bspline_pattern_to_trajectory(
        spline_type="periodic", M=10, s_init=0.0, s_final=1.0,
        az_target=0.32 * np.sin(psi), el_target=0.35 + 0.11 * np.sin(2.0 * psi),
        s_samples=s, downloops=True,
    )
    seed = {
        "r0": 200.0, "M": 10, "C_phi": np.asarray(C_phi).ravel().tolist(),
        "C_beta": np.asarray(C_beta).ravel().tolist(),
        "s_init": 0.0, "s_final": 1.0, "downloops": True,
    }
    limit = CourseTurnLimit(asymmetric_required, r0=200.0, margin=1.2)
    course, radius, sign = spline_turn_geometry(seed, 200.0)
    assert np.min(radius / limit.required_radius(course, sign)) < 1.0  # needs the fit

    fitted, report = fit_spline_to_turn_limit(seed, limit, M=16, el_min=0.2)

    assert report["converged"]
    assert report["ratio"] >= 0.99
    assert report["self_crossings"] == 1
    M, phi, beta = fitted["M"], np.array(fitted["C_phi"]), np.array(fitted["C_beta"])
    assert M == 16
    assert phi[M // 2:] == pytest.approx(-phi[: M // 2])
    assert beta[M // 2:] == pytest.approx(beta[: M // 2])
    B0 = periodic_bspline_basis_matrices(np.linspace(0.0, 1.0, 800, endpoint=False), M)[0]
    assert (B0 @ beta).min() >= 0.2 - 1e-3
    assert count_self_crossings(B0 @ phi, B0 @ beta) == 1


def test_blended_limit_goes_from_first_to_second_along_the_path():
    first = CourseTurnLimit(lambda course, sign: np.full_like(course, 20.0), r0=200.0, margin=1.0)
    second = CourseTurnLimit(lambda course, sign: np.full_like(course, 40.0), r0=200.0, margin=1.0)
    limit = BlendedTurnLimit(first, second, lambda u: np.asarray(u) >= 0.5)
    course = np.zeros(2)
    sign = np.ones(2)
    u = np.array([0.25, 0.75])

    assert limit.required_radius(course, sign, u) == pytest.approx([20.0, 40.0], rel=1e-3)
    assert limit.scale == first.scale
    up, lo = limit.bounds(ca.DM(course), u)
    assert np.asarray(up).ravel() == pytest.approx(200.0 / np.array([20.0, 40.0]), rel=1e-3)
    assert np.asarray(lo).ravel() == pytest.approx(200.0 / np.array([20.0, 40.0]), rel=1e-3)


@pytest.mark.slow
def test_fit_without_mirror_keeps_m_and_holds_a_blended_limit():
    seed = circle_path(0.2, direction=-1.0, M=20)
    tight = CourseTurnLimit(lambda course, sign: np.full_like(course, 30.0), r0=200.0)
    wide = CourseTurnLimit(lambda course, sign: np.full_like(course, 45.0), r0=200.0)
    limit = BlendedTurnLimit(tight, wide, lambda u: np.clip(2.0 * np.asarray(u) - 0.5, 0.0, 1.0))

    fitted, report = fit_spline_to_turn_limit(seed, limit, M=None, mirror=False, n_samples=800)

    assert fitted["M"] == 20
    assert report["ratio"] >= 0.99
    phi = np.array(fitted["C_phi"])
    assert phi[10:] != pytest.approx(-phi[:10])  # not the mirror rows

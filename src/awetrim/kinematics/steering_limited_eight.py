"""A figure-eight seed whose every turn the kite can fly.

A fixed seed pattern is sized for one kite. On a kite that turns less per unit
steering, its turns are tighter than the kite can hold at its steering limit,
the seed's forward march cannot trim it, and the optimizer starts from a broken
seed (LEI-V3 simulation-only ROM on the server's down-loop: seed turns 14 m,
the kite holds 25-36 m on those courses at 6 m/s). Scaling a fixed shape does
not help: the needed radius depends on the COURSE (heading down about twice
heading up), so a scaled eight is oversized everywhere else before its
descending turns are wide enough.

The requirement is therefore put on the SPLINE the optimizer receives, not on a
curve it is fitted to (a 1 m fit turns 10-25 % tighter than its curve):
:func:`fit_spline_to_turn_limit` moves the periodic-spline control points the
least (IPOPT, from a seed such as the cycle file's) so that at every sample the
signed geodesic curvature stays inside the kite's course-dependent bounds,
``-K_-(chi) <= kappa_g <= K_+(chi)``, ``K = r0 / (margin * R_kite)``
(:class:`CourseTurnLimit`, from ``TurnRadiusTable.radius`` of
``awetrim.system.turn_radius``). The mirror symmetry about the centre meridian
is exact (half the coefficients are variables,
:func:`parametrized_patterns.symmetrize_periodic_coefficients`), the height
floor and an anti-cusp speed floor are rows of the same problem. Curvature is
:func:`parametrized_patterns.geodesic_curvature_numerator`, the kernel
``Kinematics`` uses, so the check (:func:`spline_turn_geometry`) and the
optimizer agree on what a turn is.
"""
import casadi as ca
import numpy as np

from awetrim.kinematics.parametrized_patterns import (
    count_self_crossings,
    fit_bspline_pattern_to_trajectory,
    geodesic_curvature_numerator,
    periodic_bspline_basis_matrices,
    symmetrize_periodic_coefficients,
)

#: Radius [m] standing in for "no steady turn this way" (curvature bound ~0).
NO_TURN_RADIUS_M = 1.0e4
#: Trust region [rad] of every control point around the seed's: without it
#: IPOPT can leave an infeasible start for a turn-feasible but unrelated
#: basin (an eight stretched over the zenith).
MAX_MOVE = 0.35
#: Elevation ceiling [rad] of the fitted eight.
MAX_ELEVATION = np.radians(60.0)


def spline_turn_geometry(path_parameters, r0, n=800):
    """(course [rad, 0 = up], turn radius [m], curvature sign) along a
    periodic spline's path parameters, ``n`` samples over one period, from
    the exact basis derivatives (radius ``r0 / |kappa_g|``)."""
    M = int(path_parameters["M"])
    span = float(path_parameters.get("s_final", 1.0)) - float(path_parameters.get("s_init", 0.0))
    direction = 1.0 if path_parameters.get("downloops", True) else -1.0
    u = np.linspace(0.0, 1.0, int(n), endpoint=False)
    B0, B1, B2 = periodic_bspline_basis_matrices(u, M)
    C_phi = np.asarray(path_parameters["C_phi"], dtype=float)
    C_beta = np.asarray(path_parameters["C_beta"], dtype=float)
    # d/ds = (direction / span) d/du; the curvature sign follows the travel.
    k1, k2 = direction / span, (direction / span) ** 2
    beta = B0 @ C_beta
    dphi, dbeta = k1 * (B1 @ C_phi), k1 * (B1 @ C_beta)
    ddphi, ddbeta = k2 * (B2 @ C_phi), k2 * (B2 @ C_beta)
    dsigma = np.sqrt(dbeta**2 + (dphi * np.cos(beta)) ** 2 + 1e-12)
    kappa = geodesic_curvature_numerator(beta, dphi, dbeta, ddphi, ddbeta, dsigma, np) / dsigma**3
    course = np.mod(np.arctan2(dphi * np.cos(beta), dbeta), 2.0 * np.pi)
    with np.errstate(divide="ignore"):
        radius = float(r0) / np.abs(kappa)
    return course, radius, np.sign(kappa)


class CourseTurnLimit:
    """The kite's unit-sphere curvature bounds per course, mirror-symmetric.

    ``required(course, sign)`` -> radius [m] the kite needs on ``course``
    [rad, 0 = up] turning with geodesic-curvature ``sign`` (inf/NaN = no
    steady turn). A symmetric eight flies each turn mirrored on the other
    lobe, so the bound for ``+`` on course chi is the stricter of
    ``required(chi, +1)`` and ``required(-chi, -1)``, and ``K_-(chi) =
    K_+(-chi)``. ``K = r0 / (margin * R)``; :attr:`upper` / :attr:`lower`
    are CasADi functions of the course (cubic B-spline interpolants on a
    tiled grid: periodic, smooth across the atan2 branch cut).
    """

    def __init__(self, required, *, r0, margin=1.2, n=73):
        self.r0, self.margin = float(r0), float(margin)
        chi = np.linspace(-np.pi, np.pi, int(n))
        ones = np.ones_like(chi)
        radius = np.fmax(
            np.asarray(required(chi, ones), dtype=float),
            np.asarray(required(-chi, -ones), dtype=float),
        )
        radius = np.where(np.isfinite(radius), radius, NO_TURN_RADIUS_M)
        self.chi, self.radius_positive = chi, radius
        k_plus = self.r0 / (self.margin * radius)
        k_minus = k_plus[::-1]  # K_-(chi) = K_+(-chi) on the symmetric grid
        grid = np.r_[chi[:-1] - 2.0 * np.pi, chi[:-1], chi + 2.0 * np.pi]

        def tile(k):
            return np.r_[k[:-1], k[:-1], k]

        self.upper = ca.interpolant("turn_limit_upper", "bspline", [grid], tile(k_plus))
        self.lower = ca.interpolant("turn_limit_lower", "bspline", [grid], tile(k_minus))
        self.scale = float(k_plus.max())

    def required_radius(self, course, sign):
        """The bound as a radius [m] (margin included), NumPy, for checks."""
        course = np.mod(np.asarray(course, dtype=float) + np.pi, 2.0 * np.pi) - np.pi
        row = course.reshape(1, -1)
        k = np.where(
            np.asarray(sign) > 0,
            np.asarray(self.upper.map(row.size)(row)).ravel(),
            np.asarray(self.lower.map(row.size)(row)).ravel(),
        )
        return self.r0 / np.maximum(k, 1e-12)


def resample_periodic_path(path_parameters, M, n_fit=600):
    """The same periodic path on ``M`` coefficients (least squares)."""
    s = np.linspace(0.0, 1.0, int(n_fit), endpoint=False)
    B = periodic_bspline_basis_matrices(s, int(path_parameters["M"]))[0]
    az = B @ np.asarray(path_parameters["C_phi"], dtype=float)
    el = B @ np.asarray(path_parameters["C_beta"], dtype=float)
    _, C_phi, C_beta = fit_bspline_pattern_to_trajectory(
        spline_type="periodic", M=int(M), s_init=0.0, s_final=1.0,
        az_target=az, el_target=el, s_samples=s, downloops=True,
    )
    return {**path_parameters, "M": int(M),
            "C_phi": np.asarray(C_phi).ravel().tolist(),
            "C_beta": np.asarray(C_beta).ravel().tolist()}


def fit_spline_to_turn_limit(
    path_parameters,
    limit,
    *,
    M=16,
    el_min=None,
    el_max=MAX_ELEVATION,
    max_move=MAX_MOVE,
    n_samples=1600,
    speed_floor=0.25,
    precision=6,
):
    """The mirror-symmetric periodic spline nearest ``path_parameters``
    whose every turn is within ``limit`` (a :class:`CourseTurnLimit`).

    ``path_parameters`` is a one-eight down-loop periodic spline (s in [0,
    1], e.g. the server's cycle file), resampled onto ``M`` (even)
    coefficients. Unknowns: half the coefficients (``C_phi[k + M/2] =
    -C_phi[k]``, ``C_beta[k + M/2] = C_beta[k]``); objective: the squared
    coefficient move; rows at ``n_samples`` points: ``-K_-(chi) <= kappa_g
    <= K_+(chi)`` (normalized), ``el_min <= beta <= el_max`` [rad] and ``|q'|^2 >=
    speed_floor`` x its seed minimum (without it the bound can be met by
    stalling the parametrization into a cusp); every coefficient stays
    within ``max_move`` [rad] of the seed's. Returns ``(path_parameters,
    report)``; ``report`` has ``converged`` (IPOPT success), ``ratio`` (min
    over a 4x finer grid of spline radius / required radius, >= 1 = holds),
    ``turn_radius_min_m``, ``max_move`` (rad, largest coefficient move),
    ``self_crossings`` (1 for an eight), azimuth/elevation extent.
    """
    if not path_parameters.get("downloops", True):
        raise ValueError("fit_spline_to_turn_limit expects a down-loop (downloops: True) path")
    M = int(M)
    if M % 2:
        raise ValueError(f"M must be even for the mirror rows, got {M}")
    r0 = float(path_parameters["r0"])
    seed = resample_periodic_path(path_parameters, M)
    phi0, beta0 = symmetrize_periodic_coefficients(seed["C_phi"], seed["C_beta"])
    h = M // 2
    n = int(n_samples)

    u = np.linspace(0.0, 1.0, n, endpoint=False)
    B0n, B1n, B2n = periodic_bspline_basis_matrices(u, M)
    B0, B1, B2 = ca.DM(B0n), ca.DM(B1n), ca.DM(B2n)
    a = ca.MX.sym("a", h)
    b = ca.MX.sym("b", h)
    C_phi = ca.vertcat(a, -a)
    C_beta = ca.vertcat(b, b)
    beta = ca.mtimes(B0, C_beta)
    dphi, dbeta = ca.mtimes(B1, C_phi), ca.mtimes(B1, C_beta)
    ddphi, ddbeta = ca.mtimes(B2, C_phi), ca.mtimes(B2, C_beta)
    speed2 = dbeta**2 + (dphi * ca.cos(beta)) ** 2
    dsigma = ca.sqrt(speed2)
    kappa = geodesic_curvature_numerator(beta, dphi, dbeta, ddphi, ddbeta, dsigma, ca) / dsigma**3
    course = ca.atan2(dphi * ca.cos(beta), dbeta)
    k_up = limit.upper.map(n)(course.T).T
    k_lo = limit.lower.map(n)(course.T).T

    seed_speed2 = (B1n @ beta0) ** 2 + ((B1n @ phi0) * np.cos(B0n @ beta0)) ** 2
    floor = float(speed_floor) * float(seed_speed2.min())
    rows = [(kappa - k_up) / limit.scale, (-kappa - k_lo) / limit.scale, 1.0 - speed2 / floor]
    if el_min is not None:
        rows.append(float(el_min) - beta)
    if el_max is not None:
        rows.append(beta - float(el_max))
    g = ca.vertcat(*rows)
    x = ca.vertcat(a, b)
    x_ref = np.r_[phi0[:h], beta0[:h]]
    solver = ca.nlpsol(
        "turn_limit_fit", "ipopt",
        {"x": x, "f": ca.sumsqr(x - x_ref), "g": g},
        {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
         "ipopt.max_iter": 500, "ipopt.tol": 1e-8},
    )
    sol = solver(x0=x_ref, lbx=x_ref - float(max_move), ubx=x_ref + float(max_move),
                 ubg=np.zeros(g.shape[0]))
    xs = np.asarray(sol["x"]).ravel()
    stats = solver.stats()

    a_s, b_s = xs[:h], xs[h:]
    out = {**seed,
           "C_phi": np.round(np.r_[a_s, -a_s], int(precision)).tolist(),
           "C_beta": np.round(np.r_[b_s, b_s], int(precision)).tolist(),
           "s_init": 0.0, "s_final": 1.0, "downloops": True}
    course_c, radius_c, sign_c = spline_turn_geometry(out, r0, n=4 * n)
    ratio = radius_c / limit.required_radius(course_c, sign_c)
    B0c = periodic_bspline_basis_matrices(np.linspace(0.0, 1.0, 4 * n, endpoint=False), M)[0]
    az_c, el_c = B0c @ np.asarray(out["C_phi"]), B0c @ np.asarray(out["C_beta"])
    report = {
        "converged": bool(stats["success"]),
        "status": stats["return_status"],
        "iterations": int(stats.get("iter_count", -1)),
        "ratio": float(ratio.min()),
        "course_binding": float(course_c[int(np.argmin(ratio))]),
        "turn_radius_min_m": float(radius_c.min()),
        "max_move": float(np.max(np.abs(xs - x_ref))),
        "self_crossings": int(count_self_crossings(az_c, el_c)),
        "azimuth_max": float(az_c.max()),
        "elevation_range": (float(el_c.min()), float(el_c.max())),
        "M": M,
    }
    return out, report

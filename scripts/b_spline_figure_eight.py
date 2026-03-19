import casadi as ca
import numpy as np
from awetrim.utils.color_palette import set_plot_style, get_color_list

set_plot_style()
colors = get_color_list()


def _tp3(x):
    """truncated power (x_+)^3 using fmax; works for MX/SX."""
    return ca.fmax(x, 0.0) ** 3


def cubic_cardinal_B3(t):
    """
    Cardinal cubic B-spline basis kernel with support [-2,2].
    Using truncated power representation:
    B3(t) = ( (t+2)_+^3 -4(t+1)_+^3 +6(t)_+^3 -4(t-1)_+^3 + (t-2)_+^3 ) / 6
    """
    return (
        _tp3(t + 2) - 4 * _tp3(t + 1) + 6 * _tp3(t) - 4 * _tp3(t - 1) + _tp3(t - 2)
    ) / 6.0


def build_periodic_cubic_bspline_function(M, dim=1, name="per_bspline"):
    """
    Build a CasADi function S = spline(C, u) for a uniform periodic cubic B-spline.

    - M: number of control points (periodic)
    - dim: output dimension (1 for scalar, 2 for [phi,beta] etc.)
    - C: (M, dim)
    - u: scalar in [0,1] (you map s -> u outside)

    Returns:
      spline_fun(C, u) -> S (1, dim)
    """
    C = ca.MX.sym("C", M, dim)
    u = ca.MX.sym("u")  # assumed in [0,1]

    x = u * M  # in [0, M]

    S = ca.MX.zeros(1, dim)

    # Sum from i=-2..M+1; wrap coefficient index with python int modulo
    for i in range(-2, M + 2):
        idx = i % M  # integer, safe for MX indexing
        t = x - i
        w = cubic_cardinal_B3(t)  # scalar
        S += w * C[idx, :].T  # (1,dim) += scalar*(1,dim)

    return ca.Function(name, [C, u], [S], ["C", "u"], ["S"])


class PeriodicBSplinePatternAngles:
    def __init__(self, M, C_phi, C_beta, s_init, s_final, downloops=True):
        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0

        self.spline1 = build_periodic_cubic_bspline_function(M, dim=1, name="spl1")

        # store control points as (M,1)
        self.C_phi = C_phi
        self.C_beta = C_beta

    def _u(self, s):
        # map s -> u in [0,1] (no wrapping here; you can constrain s in [s_init,s_final])
        return self.omega * (s - self.s_init) / (self.s_final - self.s_init)

    def azimuth(self, r, s):
        u = self._u(s)
        return self.spline1(self.C_phi, u)[0]  # scalar

    def elevation(self, r, s):
        u = self._u(s)
        return self.spline1(self.C_beta, u)[0]  # scalar


import matplotlib.pyplot as plt


def figure_targets(
    s, A_phi=0.8, beta0=0.45, A_beta=0.35, downloops=True, pattern_type="lissajous"
):
    s = np.asarray(s).ravel()
    omega = 1.0 if downloops else -1.0
    if pattern_type == "lissajous":
        phi = A_phi * np.sin(omega * s)
        beta = beta0 + A_beta * np.sin(omega * 2 * s)
    elif pattern_type == "helix":
        phi = A_phi * np.sin(omega * s)
        beta = beta0 + A_beta * np.cos(omega * s)
    return phi, beta


def cubic_B3_np(t):
    tp = lambda x: np.maximum(x, 0.0) ** 3
    return (tp(t + 2) - 4 * tp(t + 1) + 6 * tp(t) - 4 * tp(t - 1) + tp(t - 2)) / 6.0


import casadi as ca
import numpy as np


def open_uniform_knots(M, p=3):
    """Open-uniform (clamped) knot vector on [0,1] for M control points, degree p."""
    if M < p + 1:
        raise ValueError(f"Need M >= p+1. Got M={M}, p={p}.")
    n_knots = M + p + 1
    n_int = n_knots - 2 * (p + 1)  # number of interior knots
    if n_int > 0:
        interior = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
        T = np.r_[np.zeros(p + 1), interior, np.ones(p + 1)]
    else:
        T = np.r_[np.zeros(p + 1), np.ones(p + 1)]
    return T


def build_open_cubic_bspline_function(M, dim=1, name="open_bspline", p=3):
    """
    Open (non-periodic) clamped cubic B-spline S = spline(C,u)
    - C: (M, dim)
    - u in [0,1]
    Returns: S (1,dim)
    """
    T_np = open_uniform_knots(M, p=p)
    T = ca.DM(T_np)  # constants inside CasADi graph

    C = ca.MX.sym("C", M, dim)
    u = ca.MX.sym("u")

    # clamp u to [0,1] and handle u==1 safely for half-open intervals
    u0 = ca.fmin(ca.fmax(u, 0.0), 1.0)
    u_eval = ca.if_else(u0 == 1.0, ca.DM(1.0 - 1e-12), u0)

    # degree-0 basis N_i,0(u)
    N = [None] * M
    for i in range(M):
        left = T[i]
        right = T[i + 1]
        N[i] = ca.if_else(ca.logic_and(u_eval >= left, u_eval < right), 1.0, 0.0)

    # Cox–de Boor recursion up to degree p
    for k in range(1, p + 1):
        Nk = [0] * M
        for i in range(M):
            # left term
            den1 = T[i + k] - T[i]
            term1 = ca.if_else(den1 != 0, (u_eval - T[i]) / den1 * N[i], 0.0)

            # right term uses N[i+1]
            if i + 1 < M:
                den2 = T[i + k + 1] - T[i + 1]
                term2 = ca.if_else(
                    den2 != 0, (T[i + k + 1] - u_eval) / den2 * N[i + 1], 0.0
                )
            else:
                term2 = 0.0

            Nk[i] = term1 + term2
        N = Nk

    # Evaluate spline
    S = ca.MX.zeros(1, dim)
    for i in range(M):
        S += N[i] * C[i, :].T

    # enforce exact endpoint at u==1: S(1)=last control point (clamped convention)
    S = ca.if_else(u0 == 1.0, C[M - 1, :].T, S)

    return ca.Function(name, [C, u], [S], ["C", "u"], ["S"])


def bspline_open_basis_matrix(u_grid, M, p=3):
    u = np.asarray(u_grid).ravel()
    Np = u.size
    T = open_uniform_knots(M, p=p)

    u0 = np.clip(u, 0.0, 1.0)
    u_eval = np.where(u0 == 1.0, np.nextafter(1.0, 0.0), u0)

    # degree-0
    B = np.zeros((Np, M))
    for i in range(M):
        B[:, i] = ((T[i] <= u_eval) & (u_eval < T[i + 1])).astype(float)

    # recursion
    for k in range(1, p + 1):
        Bk = np.zeros_like(B)
        for i in range(M):
            den1 = T[i + k] - T[i]
            if den1 != 0:
                Bk[:, i] += (u_eval - T[i]) / den1 * B[:, i]
            if i + 1 < M:
                den2 = T[i + k + 1] - T[i + 1]
                if den2 != 0:
                    Bk[:, i] += (T[i + k + 1] - u_eval) / den2 * B[:, i + 1]
        B = Bk

    # exact endpoint
    at_one = u0 == 1.0
    if np.any(at_one):
        B[at_one, :] = 0.0
        B[at_one, -1] = 1.0

    return B


def periodic_bspline_basis_matrix(u_grid, M):
    """
    Returns B (N,M) with B[n,j] = basis weight for control point j at u_grid[n].
    Implements the same periodic sum i=-2..M+1 with idx=i%M.
    """
    u_grid = np.asarray(u_grid).ravel()
    N = u_grid.size
    x = u_grid * M

    B = np.zeros((N, M))
    for i in range(-2, M + 2):
        idx = i % M
        t = x - i
        B[:, idx] += cubic_B3_np(t)
    return B


def fit_periodic_bspline_ctrl_points(
    M,
    s_init,
    s_final,
    N_fit,
    A_phi,
    beta0,
    A_beta,
    downloops=True,
    pattern_type="lissajous",
):
    s_grid = np.linspace(s_init, s_final, N_fit, endpoint=True)
    u_grid = (s_grid - s_init) / (s_final - s_init)

    phi_t, beta_t = figure_targets(
        s_grid,
        A_phi=A_phi,
        beta0=beta0,
        A_beta=A_beta,
        downloops=downloops,
        pattern_type=pattern_type,
    )

    B = periodic_bspline_basis_matrix(u_grid, M)

    c_phi, *_ = np.linalg.lstsq(B, phi_t, rcond=None)
    c_beta, *_ = np.linalg.lstsq(B, beta_t, rcond=None)

    C_phi = ca.DM(c_phi).reshape((M, 1))
    C_beta = ca.DM(c_beta).reshape((M, 1))
    return C_phi, C_beta


def fit_open_bspline_ctrl_points(
    M,
    s_init,
    s_final,
    N_fit,
    A_phi,
    beta0,
    A_beta,
    downloops=True,
    pattern_type="lissajous",
):
    s_grid = np.linspace(s_init, s_final, N_fit, endpoint=True)
    u_grid = (s_grid - s_init) / (s_final - s_init)

    phi_t, beta_t = figure_targets(
        s_grid,
        A_phi=A_phi,
        beta0=beta0,
        A_beta=A_beta,
        downloops=downloops,
        pattern_type=pattern_type,
    )

    B = bspline_open_basis_matrix(u_grid, M, p=3)

    c_phi, *_ = np.linalg.lstsq(B, phi_t, rcond=None)
    c_beta, *_ = np.linalg.lstsq(B, beta_t, rcond=None)

    C_phi = ca.DM(c_phi).reshape((M, 1))
    C_beta = ca.DM(c_beta).reshape((M, 1))
    return C_phi, C_beta


def open_uniform_knots(M, p=3):
    """Open-uniform (clamped) knot vector on [0,1] for M control points, degree p."""
    if M < p + 1:
        raise ValueError(f"Need M >= p+1. Got M={M}, p={p}.")
    n_knots = M + p + 1
    n_int = n_knots - 2 * (p + 1)  # number of interior knots
    if n_int > 0:
        interior = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
        T = np.r_[np.zeros(p + 1), interior, np.ones(p + 1)]
    else:
        T = np.r_[np.zeros(p + 1), np.ones(p + 1)]
    return T


def build_open_cubic_bspline_function(M, dim=1, name="open_bspline", p=3):
    """
    Open (non-periodic) clamped cubic B-spline S = spline(C,u)
    - C: (M, dim)
    - u in [0,1]
    Returns: S (1,dim)
    """
    T_np = open_uniform_knots(M, p=p)
    T = ca.DM(T_np)  # constants inside CasADi graph

    C = ca.MX.sym("C", M, dim)
    u = ca.MX.sym("u")

    # clamp u to [0,1] and handle u==1 safely for half-open intervals
    u0 = ca.fmin(ca.fmax(u, 0.0), 1.0)
    u_eval = ca.if_else(u0 == 1.0, ca.DM(1.0 - 1e-12), u0)

    # degree-0 basis N_i,0(u)
    N = [None] * M
    for i in range(M):
        left = T[i]
        right = T[i + 1]
        N[i] = ca.if_else(ca.logic_and(u_eval >= left, u_eval < right), 1.0, 0.0)

    # Cox–de Boor recursion up to degree p
    for k in range(1, p + 1):
        Nk = [0] * M
        for i in range(M):
            # left term
            den1 = T[i + k] - T[i]
            term1 = ca.if_else(den1 != 0, (u_eval - T[i]) / den1 * N[i], 0.0)

            # right term uses N[i+1]
            if i + 1 < M:
                den2 = T[i + k + 1] - T[i + 1]
                term2 = ca.if_else(
                    den2 != 0, (T[i + k + 1] - u_eval) / den2 * N[i + 1], 0.0
                )
            else:
                term2 = 0.0

            Nk[i] = term1 + term2
        N = Nk

    # Evaluate spline
    S = ca.MX.zeros(1, dim)
    for i in range(M):
        S += N[i] * C[i, :].T

    # enforce exact endpoint at u==1: S(1)=last control point (clamped convention)
    S = ca.if_else(u0 == 1.0, C[M - 1, :].T, S)

    return ca.Function(name, [C, u], [S], ["C", "u"], ["S"])


def bspline_open_basis_matrix(u_grid, M, p=3):
    u = np.asarray(u_grid).ravel()
    Np = u.size
    T = open_uniform_knots(M, p=p)

    u0 = np.clip(u, 0.0, 1.0)
    u_eval = np.where(u0 == 1.0, np.nextafter(1.0, 0.0), u0)

    # degree-0
    B = np.zeros((Np, M))
    for i in range(M):
        B[:, i] = ((T[i] <= u_eval) & (u_eval < T[i + 1])).astype(float)

    # recursion
    for k in range(1, p + 1):
        Bk = np.zeros_like(B)
        for i in range(M):
            den1 = T[i + k] - T[i]
            if den1 != 0:
                Bk[:, i] += (u_eval - T[i]) / den1 * B[:, i]
            if i + 1 < M:
                den2 = T[i + k + 1] - T[i + 1]
                if den2 != 0:
                    Bk[:, i] += (T[i + k + 1] - u_eval) / den2 * B[:, i + 1]
        B = Bk

    # exact endpoint
    at_one = u0 == 1.0
    if np.any(at_one):
        B[at_one, :] = 0.0
        B[at_one, -1] = 1.0

    return B


def demo_plot():

    s_init = 0
    s_final = 2 * np.pi + s_init
    downloops = True
    pattern_type = "lissajous"  # "lissajous" or "helix"
    # choose #control points
    M = 10

    # target Lissajous
    A_phi, beta0, A_beta = 0.32, 0.3, 0.1

    # fit init control points
    C_phi0, C_beta0 = fit_periodic_bspline_ctrl_points(
        M,
        s_init,
        s_final,
        N_fit=400,
        A_phi=A_phi,
        beta0=beta0,
        A_beta=A_beta,
        downloops=downloops,
        pattern_type=pattern_type,
    )

    print("Fitted control points (phi):", C_phi0.T)
    print("Fitted control points (beta):", C_beta0.T)

    pattern = PeriodicBSplinePatternAngles(M, C_phi0, C_beta0, s_init, s_final)

    # CasADi functions (symbolic s)
    s = ca.MX.sym("s")
    r = ca.MX.sym("r")
    phi_fun = ca.Function("phi", [s], [pattern.azimuth(r, s)])
    beta_fun = ca.Function("beta", [s], [pattern.elevation(r, s)])

    s_plot = np.linspace(s_init, s_final, 100, endpoint=True)
    phi_fit = np.array([float(phi_fun(si)) for si in s_plot])
    beta_fit = np.array([float(beta_fun(si)) for si in s_plot])

    phi_true, beta_true = figure_targets(
        s_plot,
        A_phi=A_phi,
        beta0=beta0,
        A_beta=A_beta,
        downloops=downloops,
        pattern_type=pattern_type,
    )

    # plot
    # plt.figure(figsize=(6, 3.6))
    # plt.plot(s_plot, phi_true, label="target φ(s)")
    # plt.plot(s_plot, phi_fit, "--", label="periodic B-spline φ(s)")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()

    # plt.figure(figsize=(6, 3.6))
    # plt.plot(s_plot, beta_true, label="target β(s)")
    # plt.plot(s_plot, beta_fit, "--", label="periodic B-spline β(s)")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()

    plt.figure(figsize=(5, 4))
    plt.plot(np.degrees(phi_true), np.degrees(beta_true), label="Baseline Lissajous")
    plt.plot(
        np.degrees(phi_fit),
        np.degrees(beta_fit),
        ".",
        label="Periodic B-spline",
    )
    phi_ctrl = np.r_[C_phi0.full().flatten(), C_phi0.full().flatten()[0]]
    beta_ctrl = np.r_[C_beta0.full().flatten(), C_beta0.full().flatten()[0]]
    plt.plot(
        np.degrees(phi_ctrl),
        np.degrees(beta_ctrl),
        "--o",
        label="Control points",
        # color=colors[2],
    )
    plt.grid(True)
    plt.legend()
    plt.xlabel(r"$\phi$ ($^{\circ}$)")
    plt.ylabel(r"$\beta$ ($^{\circ}$)")
    plt.tight_layout()
    plt.savefig("results/figures/torque2026/bspline_fit_example.pdf")
    plt.show()

    s_sym = ca.MX.sym("s")
    phi_spline = pattern.azimuth(0, s_sym)
    dphi_ds = ca.gradient(phi_spline, s_sym)
    print(phi_spline)


if __name__ == "__main__":
    demo_plot()

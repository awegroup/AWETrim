import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from awetrim.utils.color_palette import set_plot_style, get_color_list

set_plot_style()
colors = get_color_list()


# =============================================================================
# Target trajectory
# =============================================================================


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
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")

    return phi, beta


# =============================================================================
# Periodic cubic B-spline
# =============================================================================


def _tp3(x):
    """Truncated power (x_+)^3 using fmax; works for MX/SX."""
    return ca.fmax(x, 0.0) ** 3


def cubic_cardinal_B3(t):
    """
    Cardinal cubic B-spline basis kernel with support [-2, 2]:
    B3(t) = ((t+2)_+^3 - 4(t+1)_+^3 + 6(t)_+^3 - 4(t-1)_+^3 + (t-2)_+^3)/6
    """
    return (
        _tp3(t + 2) - 4 * _tp3(t + 1) + 6 * _tp3(t) - 4 * _tp3(t - 1) + _tp3(t - 2)
    ) / 6.0


def cubic_B3_np(t):
    tp = lambda x: np.maximum(x, 0.0) ** 3
    return (tp(t + 2) - 4 * tp(t + 1) + 6 * tp(t) - 4 * tp(t - 1) + tp(t - 2)) / 6.0


def build_periodic_cubic_bspline_function(M, dim=1, name="periodic_bspline"):
    """
    Uniform periodic cubic B-spline S = spline(C, u)
    - C: (M, dim)
    - u: scalar in [0,1]
    Returns: S (1, dim)
    """
    C = ca.MX.sym("C", M, dim)
    u = ca.MX.sym("u")

    x = u * M
    S = ca.MX.zeros(1, dim)

    for i in range(-2, M + 2):
        idx = i % M
        t = x - i
        w = cubic_cardinal_B3(t)
        S += w * C[idx, :].T

    return ca.Function(name, [C, u], [S], ["C", "u"], ["S"])


def periodic_bspline_basis_matrix(u_grid, M):
    """
    Returns B (N,M) with B[n,j] = basis weight for control point j at u_grid[n].
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


# =============================================================================
# Open (clamped) cubic B-spline
# =============================================================================


def open_uniform_knots(M, p=3):
    """Open-uniform (clamped) knot vector on [0,1] for M control points, degree p."""
    if M < p + 1:
        raise ValueError(f"Need M >= p+1. Got M={M}, p={p}.")

    n_knots = M + p + 1
    n_int = n_knots - 2 * (p + 1)

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
    T = ca.DM(T_np)

    C = ca.MX.sym("C", M, dim)
    u = ca.MX.sym("u")

    u0 = ca.fmin(ca.fmax(u, 0.0), 1.0)
    u_eval = ca.if_else(u0 == 1.0, ca.DM(1.0 - 1e-12), u0)

    N = [None] * M
    for i in range(M):
        left = T[i]
        right = T[i + 1]
        N[i] = ca.if_else(ca.logic_and(u_eval >= left, u_eval < right), 1.0, 0.0)

    for k in range(1, p + 1):
        Nk = [0] * M
        for i in range(M):
            den1 = T[i + k] - T[i]
            term1 = ca.if_else(den1 != 0, (u_eval - T[i]) / den1 * N[i], 0.0)

            if i + 1 < M:
                den2 = T[i + k + 1] - T[i + 1]
                term2 = ca.if_else(
                    den2 != 0,
                    (T[i + k + 1] - u_eval) / den2 * N[i + 1],
                    0.0,
                )
            else:
                term2 = 0.0

            Nk[i] = term1 + term2

        N = Nk

    S = ca.MX.zeros(1, dim)
    for i in range(M):
        S += N[i] * C[i, :].T

    S = ca.if_else(u0 == 1.0, C[M - 1, :].T, S)

    return ca.Function(name, [C, u], [S], ["C", "u"], ["S"])


def bspline_open_basis_matrix(u_grid, M, p=3):
    u = np.asarray(u_grid).ravel()
    Np = u.size
    T = open_uniform_knots(M, p=p)

    u0 = np.clip(u, 0.0, 1.0)
    u_eval = np.where(u0 == 1.0, np.nextafter(1.0, 0.0), u0)

    B = np.zeros((Np, M))
    for i in range(M):
        B[:, i] = ((T[i] <= u_eval) & (u_eval < T[i + 1])).astype(float)

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

    at_one = u0 == 1.0
    if np.any(at_one):
        B[at_one, :] = 0.0
        B[at_one, -1] = 1.0

    return B


# =============================================================================
# Pattern classes
# =============================================================================


class PeriodicBSplinePatternAngles:
    def __init__(self, M, C_phi, C_beta, s_init, s_final, downloops=True):
        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0

        self.spline1 = build_periodic_cubic_bspline_function(
            self.M, dim=1, name=f"periodic_bspline_{self.M}"
        )

        self.C_phi = ca.DM(C_phi).reshape((self.M, 1))
        self.C_beta = ca.DM(C_beta).reshape((self.M, 1))

    def _u(self, s):
        return self.omega * (s - self.s_init) / (self.s_final - self.s_init)

    def azimuth(self, r, s):
        u = self._u(s)
        return self.spline1(self.C_phi, u)[0]

    def elevation(self, r, s):
        u = self._u(s)
        return self.spline1(self.C_beta, u)[0]


class OpenBSplinePatternAngles:
    def __init__(self, M, C_phi, C_beta, s_init, s_final, downloops=True):
        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0

        self.spline1 = build_open_cubic_bspline_function(
            self.M, dim=1, name=f"open_bspline_{self.M}"
        )

        self.C_phi = ca.DM(C_phi).reshape((self.M, 1))
        self.C_beta = ca.DM(C_beta).reshape((self.M, 1))

    def _u(self, s):
        return self.omega * (s - self.s_init) / (self.s_final - self.s_init)

    def azimuth(self, r, s):
        u = self._u(s)
        return self.spline1(self.C_phi, u)[0]

    def elevation(self, r, s):
        u = self._u(s)
        return self.spline1(self.C_beta, u)[0]


# =============================================================================
# Generic fitting function
# =============================================================================


def fit_bspline_pattern_to_trajectory(
    spline_type,
    M,
    s_init,
    s_final,
    az_target,
    el_target,
    s_samples,
    downloops=True,
):
    """
    Fit either a periodic or open cubic B-spline to target az/el data.

    Parameters
    ----------
    spline_type : str
        "periodic" or "open"
    M : int
        Number of control points
    s_init, s_final : float
        Path parameter interval
    az_target, el_target : array_like
        Target azimuth/elevation values sampled on s_samples
    s_samples : array_like
        Sampling points in [s_init, s_final]
    downloops : bool
        Same convention as your existing class

    Returns
    -------
    pattern : PeriodicBSplinePatternAngles or OpenBSplinePatternAngles
    C_phi, C_beta : ca.DM
        Fitted control points
    """
    s_samples = np.asarray(s_samples).ravel()
    az_target = np.asarray(az_target).ravel()
    el_target = np.asarray(el_target).ravel()

    if not (s_samples.size == az_target.size == el_target.size):
        raise ValueError("s_samples, az_target, and el_target must have same length.")

    omega = 1.0 if downloops else -1.0
    u_grid = omega * (s_samples - s_init) / (s_final - s_init)

    if spline_type == "periodic":
        B = periodic_bspline_basis_matrix(u_grid, M)
    elif spline_type == "open":
        B = bspline_open_basis_matrix(u_grid, M, p=3)
    else:
        raise ValueError("spline_type must be 'periodic' or 'open'.")

    c_phi, *_ = np.linalg.lstsq(B, az_target, rcond=None)
    c_beta, *_ = np.linalg.lstsq(B, el_target, rcond=None)

    C_phi = ca.DM(c_phi).reshape((M, 1))
    C_beta = ca.DM(c_beta).reshape((M, 1))

    if spline_type == "periodic":
        pattern = PeriodicBSplinePatternAngles(
            M, C_phi, C_beta, s_init, s_final, downloops=downloops
        )
    else:
        pattern = OpenBSplinePatternAngles(
            M, C_phi, C_beta, s_init, s_final, downloops=downloops
        )

    return pattern, C_phi, C_beta


# =============================================================================
# Evaluation helpers
# =============================================================================


def evaluate_pattern(pattern, s_plot):
    s_sym = ca.MX.sym("s")
    r_sym = ca.MX.sym("r")

    phi_fun = ca.Function("phi_eval", [s_sym], [pattern.azimuth(r_sym, s_sym)])
    beta_fun = ca.Function("beta_eval", [s_sym], [pattern.elevation(r_sym, s_sym)])

    phi = np.array([float(phi_fun(si)) for si in s_plot])
    beta = np.array([float(beta_fun(si)) for si in s_plot])

    return phi, beta


def rms(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return np.sqrt(np.mean((a - b) ** 2))


# =============================================================================
# Demo
# =============================================================================


def demo_plot():
    s_init = 0.0
    s_final = 2 * np.pi
    downloops = True
    pattern_type = "lissajous"
    M = 10

    A_phi, beta0, A_beta = 0.32, 0.30, 0.10

    s_fit = np.linspace(s_init, s_final, 400, endpoint=True)
    phi_target, beta_target = figure_targets(
        s_fit,
        A_phi=A_phi,
        beta0=beta0,
        A_beta=A_beta,
        downloops=downloops,
        pattern_type=pattern_type,
    )

    periodic_pattern, C_phi_per, C_beta_per = fit_bspline_pattern_to_trajectory(
        spline_type="periodic",
        M=M,
        s_init=s_init,
        s_final=s_final,
        az_target=phi_target,
        el_target=beta_target,
        s_samples=s_fit,
        downloops=downloops,
    )

    open_pattern, C_phi_open, C_beta_open = fit_bspline_pattern_to_trajectory(
        spline_type="open",
        M=M,
        s_init=s_init,
        s_final=s_final,
        az_target=phi_target,
        el_target=beta_target,
        s_samples=s_fit,
        downloops=downloops,
    )

    s_plot = np.linspace(s_init, s_final, 300, endpoint=True)

    phi_true, beta_true = figure_targets(
        s_plot,
        A_phi=A_phi,
        beta0=beta0,
        A_beta=A_beta,
        downloops=downloops,
        pattern_type=pattern_type,
    )

    phi_per, beta_per = evaluate_pattern(periodic_pattern, s_plot)
    phi_open, beta_open = evaluate_pattern(open_pattern, s_plot)

    print("Periodic phi RMS [deg]:", np.degrees(rms(phi_per, phi_true)))
    print("Periodic beta RMS [deg]:", np.degrees(rms(beta_per, beta_true)))
    print("Open phi RMS [deg]:", np.degrees(rms(phi_open, phi_true)))
    print("Open beta RMS [deg]:", np.degrees(rms(beta_open, beta_true)))

    # -------------------------------------------------------------------------
    # 1) phi(s)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(s_plot, np.degrees(phi_true), label="Target azimuth")
    plt.plot(s_plot, np.degrees(phi_per), "--", label="Periodic spline")
    plt.plot(s_plot, np.degrees(phi_open), ":", label="Open spline")
    plt.grid(True)
    plt.xlabel(r"$s$")
    plt.ylabel(r"$\phi$ ($^\circ$)")
    plt.legend()
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # 2) beta(s)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(s_plot, np.degrees(beta_true), label="Target elevation")
    plt.plot(s_plot, np.degrees(beta_per), "--", label="Periodic spline")
    plt.plot(s_plot, np.degrees(beta_open), ":", label="Open spline")
    plt.grid(True)
    plt.xlabel(r"$s$")
    plt.ylabel(r"$\beta$ ($^\circ$)")
    plt.legend()
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # 3) trajectory in az-el plane
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(np.degrees(phi_true), np.degrees(beta_true), label="Target")
    plt.plot(np.degrees(phi_per), np.degrees(beta_per), "--", label="Periodic spline")
    plt.plot(np.degrees(phi_open), np.degrees(beta_open), ":", label="Open spline")

    # periodic control polygon
    phi_ctrl_per = np.r_[C_phi_per.full().flatten(), C_phi_per.full().flatten()[0]]
    beta_ctrl_per = np.r_[C_beta_per.full().flatten(), C_beta_per.full().flatten()[0]]
    plt.plot(
        np.degrees(phi_ctrl_per),
        np.degrees(beta_ctrl_per),
        "--o",
        markersize=4,
        label="Periodic control points",
    )

    # open control polygon
    plt.plot(
        np.degrees(C_phi_open.full().flatten()),
        np.degrees(C_beta_open.full().flatten()),
        ":s",
        markersize=4,
        label="Open control points",
    )

    plt.grid(True)
    plt.xlabel(r"$\phi$ ($^\circ$)")
    plt.ylabel(r"$\beta$ ($^\circ$)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("results/figures/torque2026/bspline_fit_comparison.pdf")
    plt.show()


if __name__ == "__main__":
    demo_plot()

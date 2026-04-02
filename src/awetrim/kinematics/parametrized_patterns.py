import casadi as ca
from abc import ABC, abstractmethod
import numpy as np
from math import comb
import matplotlib.pyplot as plt


class ParametrizedPatterns(ABC):
    def __init__(self, **kwargs):
        self.optimization_vars = {}  # Dictionary to store symbolic MX variables
        for key, value in kwargs.items():
            setattr(self, key, value)
            if isinstance(value, ca.MX):  # If value is symbolic, store it separately
                self.optimization_vars[key] = value

    def x(self, r, s):
        return r * ca.cos(self.azimuth(r, s)) * ca.cos(self.elevation(r, s))

    def y(self, r, s):
        return r * ca.sin(self.azimuth(r, s)) * ca.cos(self.elevation(r, s))

    def z(self, r, s):
        return r * ca.sin(self.elevation(r, s))

    def curvature(self, r_array, s_array):

        # --- Get scalar fields as expressions of s (t is fixed here) ---
        # If your methods are r(s,t), phi(s), beta(s), call accordingly.
        # The user code showed self.r(t) and gradient(..., s), so we mimic that.

        s = ca.MX.sym("s")
        r = ca.MX.sym("r")  # expression that depends on s
        phi = self.azimuth(r, s)  # expression that depends on s
        beta = self.elevation(r, s)  # expression that depends on s

        # --- Cartesian curve r_vec(s) ---
        x = r * ca.cos(beta) * ca.cos(phi)
        y = r * ca.cos(beta) * ca.sin(phi)
        z = r * ca.sin(beta)
        r_vec = ca.vertcat(x, y, z)  # 3x1

        # --- First and second derivatives wrt s ---
        print(r_vec)
        r_s = ca.jacobian(r_vec, s)  # 3x1
        r_ss = ca.jacobian(r_s, s)  # 3x1

        # --- Curvature and radius ---
        # (use a tiny epsilon to avoid division by zero in degenerate cases)
        eps = 1e-12
        cross_rs_rss = ca.cross(r_s, r_ss)  # 3x1
        num = ca.norm_2(cross_rs_rss)  # ||r_s x r_ss||
        den = ca.power(ca.norm_2(r_s), 3) + eps  # ||r_s||^3
        kappa = num / den
        rho = 1.0 / (kappa + eps)

        kappa_fun = ca.Function("kappa_fun", [r, s], [kappa], {"allow_free": True})
        kappa = kappa_fun(r_array, s_array)

        return kappa

    def radius_curvature(self, r, s):
        return 1.0 / (self.curvature(r, s) + 1e-12)


class Helix(ParametrizedPatterns):
    def __init__(self, omega, r0, amp0, vr, beta0, kappa=1, kbeta=0):
        super().__init__(
            omega=omega,
            r0=r0,
            amp0=amp0,
            vr=vr,
            beta0=beta0,
            kappa=kappa,
            kbeta=kbeta,
        )

    def beta(self, t):
        return self.beta0 * (1 + self.kbeta * (self.r0 / self.r(t) - 1))

    def az_amp(self, t):
        return self.az_amp0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def beta_amp(self, t):
        return self.beta_amp0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def r(self, t):
        return self.r0 + self.vr * t

    def azimuth(self, t, s):
        return self.az_amp(t) * ca.cos(self.omega * s)

    def elevation(self, t, s):
        return self.beta_amp(t) * ca.sin(self.omega * s)


class LissajousAngles(ParametrizedPatterns):
    def __init__(self, omega, r0, az_amp0, beta_amp0, vr, beta0, kappa=0, kbeta=0):
        super().__init__(
            omega=omega,
            r0=r0,
            az_amp0=az_amp0,
            beta_amp0=beta_amp0,
            vr=vr,
            beta0=beta0,
            kappa=kappa,
            kbeta=kbeta,
        )

    def beta(self, t):
        return self.beta0 * (1 + self.kbeta * (self.r0 / self.r(t) - 1))

    def r(self, t):
        return self.r0 + self.vr * t

    def az_amp(self, t):
        return self.az_amp0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def beta_amp(self, t):
        return self.beta_amp0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def azimuth(self, t, s):
        return self.az_amp(t) * ca.cos(self.omega * s)

    def elevation(self, t, s):
        return self.beta_amp(t) * ca.sin(self.omega * s) * ca.cos(
            self.omega * s
        ) + self.beta(t)


def create_pattern_from_dict(
    pattern_type,
    parameters,
) -> ParametrizedPatterns:

    required_params = {
        "helix": ["omega", "r0", "d0", "vr", "beta0", "kappa"],
        "lissajous": ["omega", "r0", "a0", "h0", "vr", "beta0", "kappa"],
        "lissajous_angles": [
            "omega",
            "r0",
            "az_amp0",
            "beta_amp0",
            "vr",
            "beta0",
            "kappa",
        ],
        "figure_eight": ["omega", "r0", "ry", "rz", "vr", "beta0", "ky", "kz", "kappa"],
        "figure_eight_angles": [
            "omega",
            "r0",
            "az_amp0",
            "beta_amp0",
            "vr",
            "beta0",
            "ky",
            "kz",
            "kappa",
        ],
        "cst_lissajous": [
            "r0",
            "az_amp0",
            "beta_amp0",
            "beta0",
            "beta_coeffs",
            "az_coeffs",
        ],
        "spline": ["r0", "r1", "C_az", "C_el", "s_norm_az", "s_norm_el"],
        "cst_helix": [
            "r0",
            "az_amp0",
            "beta_amp0",
            "beta0",
            "phi0",
            "beta_coeffs",
            "az_coeffs",
        ],
        "reel_in_simple": ["elevation_start_ri", "elevation_start_riro"],
        "transition_simple": [
            "elevation_start_riro",
            "elevation_start_ro",
            "azimuth_start_riro",
            "azimuth_start_ro",
        ],
        "spline_periodic": ["M", "C_phi", "C_beta", "s_init", "s_final"],
        "spline_open": ["M", "C_phi", "C_beta", "s_init", "s_final", "r0"],
    }

    if pattern_type not in required_params:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    missing_params = [
        param for param in required_params[pattern_type] if param not in parameters
    ]
    if missing_params:
        raise ValueError(
            f"Missing required parameters in 'parameters' for '{pattern_type}': {', '.join(missing_params)}"
        )

    # Instantiate the appropriate pattern class
    pattern_classes = {
        "helix": Helix,
        "lissajous_angles": LissajousAngles,
        "reel_in_simple": Reelin_Simple,
        "transition_simple": Transition_Simple,
        "spline_periodic": PeriodicBSpline,
        "spline_open": OpenBSpline,
    }

    return pattern_classes[pattern_type](**parameters)


class Reelin_Simple(ParametrizedPatterns):
    def __init__(
        self,
        elevation_start_ri,
        elevation_start_riro,
    ):  # <- only flags
        super().__init__(
            elevation_start_ri=elevation_start_ri,
            elevation_start_riro=elevation_start_riro,
        )

    def elevation(self, r, s):
        return self.elevation_start_ri + s * (
            self.elevation_start_riro - self.elevation_start_ri
        )

    def azimuth(self, r, s):
        return 0


class Transition_Simple(ParametrizedPatterns):
    def __init__(
        self,
        elevation_start_riro,
        elevation_start_ro,
        azimuth_start_riro=0,
        azimuth_start_ro=0,
    ):  # <- only flags
        super().__init__(
            elevation_start_riro=elevation_start_riro,
            elevation_start_ro=elevation_start_ro,
            azimuth_start_riro=azimuth_start_riro,
            azimuth_start_ro=azimuth_start_ro,
        )

    def elevation(self, r, s):
        return self.elevation_start_riro + s * (
            self.elevation_start_ro - self.elevation_start_riro
        )

    def azimuth(self, r, s):
        return self.azimuth_start_riro + s * (
            self.azimuth_start_ro - self.azimuth_start_riro
        )


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


def cubic_B3_np(t):
    tp = lambda x: np.maximum(x, 0.0) ** 3
    return (tp(t + 2) - 4 * tp(t + 1) + 6 * tp(t) - 4 * tp(t - 1) + tp(t - 2)) / 6.0


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
    """Basis matrix B[n,j] for periodic cubic B-splines evaluated on u_grid."""
    u_grid = np.asarray(u_grid).ravel()
    N = u_grid.size
    x = u_grid * M

    B = np.zeros((N, M))
    for i in range(-2, M + 2):
        idx = i % M
        t = x - i
        B[:, idx] += cubic_B3_np(t)

    return B


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


class PeriodicBSpline(ParametrizedPatterns):

    def __init__(self, M, C_phi, C_beta, s_init, s_final, r0=None, downloops=True):
        super().__init__(
            M=M, C_phi=C_phi, C_beta=C_beta, s_init=s_init, s_final=s_final, r0=r0
        )
        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0

        self.spline = build_periodic_cubic_bspline_function(
            self.M, dim=1, name=f"periodic_bspline_{self.M}"
        )

        self.C_phi = C_phi
        self.C_beta = C_beta

    def _u(self, s):
        return self.omega * (s - self.s_init) / (self.s_final - self.s_init)

    def _eval_spline_vec(self, C, u):
        if u.is_scalar():
            return self.spline(C, u)[0]

        u_col = ca.reshape(u, u.numel(), 1)
        N = int(u_col.numel())
        spl_map = self.spline.map(N)
        S = spl_map(C, u_col)
        return S.T

    def azimuth(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_phi, u)

    def elevation(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_beta, u)


class OpenBSpline(ParametrizedPatterns):
    def __init__(self, M, C_phi, C_beta, s_init, s_final, downloops=True, r0=None):
        super().__init__(
            M=M, C_phi=C_phi, C_beta=C_beta, s_init=s_init, s_final=s_final, r0=r0
        )
        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0
        self.r0 = r0  # optional reference radius for compatibility

        self.spline = build_open_cubic_bspline_function(
            self.M, dim=1, name=f"open_bspline_{self.M}"
        )

        self.C_phi = C_phi
        self.C_beta = C_beta

    def _u(self, s):
        return self.omega * (s - self.s_init) / (self.s_final - self.s_init)

    def _eval_spline_vec(self, C, u):
        if u.is_scalar():
            return self.spline(C, u)[0]

        u_col = ca.reshape(u, u.numel(), 1)
        N = int(u_col.numel())
        spl_map = self.spline.map(N)
        S = spl_map(C, u_col)
        return S.T

    def azimuth(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_phi, u)

    def elevation(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_beta, u)


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
    Fit a periodic or open cubic B-spline to target azimuth/elevation samples.
    Returns the fitted pattern instance and control points.
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
        pattern = PeriodicBSpline(
            M, C_phi, C_beta, s_init, s_final, downloops=downloops
        )
    else:
        pattern = OpenBSpline(M, C_phi, C_beta, s_init, s_final, downloops=downloops)

    return pattern, C_phi, C_beta

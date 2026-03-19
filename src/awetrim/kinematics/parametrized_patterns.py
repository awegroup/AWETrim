import casadi as ca
from abc import ABC, abstractmethod
import numpy as np
from math import comb
import matplotlib.pyplot as plt
from awetrim.utils.my_defaults import DEFAULT_SPLINE_PATTERN_CONFIG


class ParametrizedPatterns(ABC):

    def __init__(self, **kwargs):
        self.optimization_vars = (
            {}
        )  # Dictionary to store symbolic optimization variables
        for key, value in kwargs.items():
            setattr(self, key, value)
            if isinstance(value, ca.MX):  # If value is symbolic, store it separately
                self.optimization_vars[key] = value

    def x(self, t, s):
        return self.xd(t, s) * ca.cos(self.beta(t)) - self.zd(t, s) * ca.sin(
            self.beta(t)
        )

    def z(self, t, s):
        return self.xd(t, s) * ca.sin(self.beta(t)) + self.zd(t, s) * ca.cos(
            self.beta(t)
        )

    def y(self, t, s):
        return self.yd(t, s)

    def azimuth(self, t, s):
        return ca.atan2(self.y(t, s), self.x(t, s))

    def elevation(self, t, s):
        return ca.atan2(self.z(t, s), ca.sqrt(self.x(t, s) ** 2 + self.y(t, s) ** 2))

    def curvature(self, t_array, s_array):

        # --- Get scalar fields as expressions of s (t is fixed here) ---
        # If your methods are r(s,t), phi(s), beta(s), call accordingly.
        # The user code showed self.r(t) and gradient(..., s), so we mimic that.
        t = ca.MX.sym("t")
        s = ca.MX.sym("s")
        r = self.r(t)  # expression that depends on s
        phi = self.azimuth(t, s)  # expression that depends on s
        beta = self.elevation(t, s)  # expression that depends on s

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

        kappa_fun = ca.Function("kappa_fun", [t, s], [kappa])
        kappa = kappa_fun(t_array, s_array)

        return kappa

    def radius_curvature(self, t, s):
        return 1.0 / (self.curvature(t, s) + 1e-12)


class Helix(ParametrizedPatterns):

    def __init__(self, omega, r0, d0, vr, beta0, kappa=1, kbeta=0):
        super().__init__(
            omega=omega, r0=r0, d0=d0, vr=vr, beta0=beta0, kappa=kappa, kbeta=kbeta
        )

    def beta(self, t):
        return self.beta0 * (1 + self.kbeta * (self.r0 / self.r(t) - 1))

    def d(self, t):
        return self.d0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def r(self, t):
        return self.r0 + self.vr * t

    def yd(self, t, s):
        return self.d(t) / 2 * ca.sin(self.omega * s)

    def zd(self, t, s):
        return self.d(t) / 2 * ca.cos(self.omega * s)

    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(t, s)
        zd = self.zd(t, s)
        return ca.sqrt(r**2 - yd**2 - zd**2)


class HelixAngles(ParametrizedPatterns):
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


class Lissajous(ParametrizedPatterns):

    def __init__(self, omega, r0, a0, h0, vr, beta0, kappa=0, kbeta=0):
        super().__init__(
            omega=omega,
            r0=r0,
            a0=a0,
            h0=h0,
            vr=vr,
            beta0=beta0,
            kappa=kappa,
            kbeta=kbeta,
        )

    def beta(self, t):
        return self.beta0 * (1 + self.kbeta * (self.r0 / self.r(t) - 1))

    def a(self, t):
        return self.a0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def h(self, t):
        return self.h0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def r(self, t):
        return self.r0 + self.vr * t

    def yd(self, t, s):
        return self.a(t) * ca.cos(self.omega * s)

    def zd(self, t, s):
        return self.h(t) * ca.sin(2 * self.omega * s)

    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(t, s)
        zd = self.zd(t, s)
        return ca.sqrt(r**2 - yd**2 - zd**2)


class FigureEight(ParametrizedPatterns):

    def __init__(self, omega, r0, ry, rz, vr, beta0, ky=1, kz=1, kappa=0, kbeta=0):
        super().__init__(
            omega=omega,
            r0=r0,
            ry0=ry,
            rz0=rz,
            vr=vr,
            ky=ky,
            kz=kz,
            kappa=kappa,
            beta0=beta0,
            kbeta=kbeta,
        )

    def beta(self, t):
        return self.beta0 * (1 + self.kbeta * (self.r0 / self.r(t) - 1))

    def r(self, t):
        return self.r0 + self.vr * t

    def ry(self, t):
        return self.ry0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def rz(self, t):
        return self.rz0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def yd(self, t, s):
        return (
            self.ry(t)
            * ca.cos(self.omega * s)
            / (1 + self.ky * ca.sin(self.omega * s) ** 2)
        )

    def zd(self, t, s):
        return (
            self.rz(t)
            * ca.sin(self.omega * s)
            * ca.cos(self.omega * s)
            / (1 + self.kz * ca.sin(self.omega * s) ** 2)
        )

    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(t, s)
        zd = self.zd(t, s)
        return ca.sqrt(r**2 - yd**2 - zd**2)


class ParametrizedPatternsAngles(ParametrizedPatterns):
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


class FigureEightAngles(ParametrizedPatternsAngles):

    def __init__(
        self, omega, r0, az_amp0, beta_amp0, vr, beta0, ky=1, kz=1, kappa=0, kbeta=0
    ):
        super().__init__(
            omega=omega,
            r0=r0,
            az_amp0=az_amp0,
            beta_amp0=beta_amp0,
            vr=vr,
            ky=ky,
            kz=kz,
            kappa=kappa,
            beta0=beta0,
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
        return (
            self.az_amp(t)
            * ca.cos(self.omega * s)
            / (1 + self.ky * ca.sin(self.omega * s) ** 2)
        )

    def elevation(self, t, s):
        return self.beta_amp(t) * ca.sin(self.omega * s) * ca.cos(self.omega * s) / (
            1 + self.kz * ca.sin(self.omega * s) ** 2
        ) + self.beta(t)


from awetrim.kinematics.reelin_parametrization import ReelInBezier


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
        "lissajous": Lissajous,
        "lissajous_angles": LissajousAngles,
        "figure_eight": FigureEight,
        "figure_eight_angles": FigureEightAngles,
        "cst_lissajous": CST_Lissajous,
        "spline": CasadiSpline,
        "cst_helix": CST_Helix,
        "reel_in_simple": Reelin_Simple,
        "transition_simple": Transition_Simple,
        "spline_periodic": PeriodicBSpline,
    }

    return pattern_classes[pattern_type](**parameters)


class CST_Lissajous(ParametrizedPatternsAngles):
    def __init__(
        self,
        az_amp0,
        beta_amp0,
        beta0,
        beta_coeffs,
        az_coeffs,
        kappa=0.0,
        kbeta=0.0,
        width_phi=0.45,
        width_beta=0.45,
        left_first=True,
        normalize_bumps=False,
        repeat_phi=True,
        repeat_beta=True,
        downloops=True,
        **kwargs,
    ):  # <- only flags
        super().__init__(
            az_amp0=az_amp0,
            beta_amp0=beta_amp0,
            beta0=beta0,
            kappa=kappa,
            kbeta=kbeta,
            beta_coeffs=beta_coeffs,
            az_coeffs=az_coeffs,
            width_phi=width_phi,
            width_beta=width_beta,
            left_first=left_first,
            normalize_bumps=normalize_bumps,
            **kwargs,
        )

        self.omega = 1.0 if downloops else -1.0
        # Base weight vectors
        self.az_coeffs = ca.vertcat(az_coeffs)
        self.beta_coeffs = ca.vertcat(beta_coeffs)
        P_phi = int(self.az_coeffs.numel())
        P_beta = int(self.beta_coeffs.numel())

        # Total number of bumps = len(weights) or 2× if repeating
        self.K_phi = 2 * P_phi if repeat_phi else P_phi
        self.K_beta = 2 * P_beta if repeat_beta else P_beta

        self.width_phi, self.width_beta = float(width_phi), float(width_beta)
        self.normalize_bumps = bool(normalize_bumps)
        self.sgn = -1.0 if left_first else +1.0

    def beta_center(self, r):
        return self.beta0 * (self.r0 / (self.r0 + (r - self.r0) * self.kbeta))

    def az_amp(self, r):
        return self.az_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    def beta_amp(self, r):
        return self.beta_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    @staticmethod
    def _mod1(x):
        return x - ca.floor(x)

    @staticmethod
    def _p(x):
        # 2nd basis polynomial of 4th-order Bernstein: p(x)=6x^2(1-x)^2
        return 6.0 * (x**2) * ((1.0 - x) ** 2)

    @staticmethod
    def _gate01(x):
        # hard gate: 1 if 0<=x<=1 else 0
        return ca.if_else(ca.logic_and(x >= 0.0, x <= 1.0), 1.0, 0.0)

    def _bump_pair(self, u, s0, width, normalize=False):
        """
        Periodic bump without inner mod1:
        - right bump active for u in [s0, s0+width]
        - left bump is the wrapped copy (shifted by +1)
        """
        x_right = (u - s0) / width
        x_left = (u - s0 + 1.0) / width  # wrap-around copy

        pright = self._gate01(x_right) * self._p(x_right)
        pleft = self._gate01(x_left) * self._p(x_left)

        bump = pright + pleft
        # mean_bump = ca.sum1(bump) / bump.numel()
        # bump = bump - mean_bump  # zero-mean on the [0,1] period for any s0
        return bump / width if normalize else bump

    def _build_shape_repeat(self, u, K, width, base_vec):
        """N(u) = 1 + Σ_{k=0..K-1} w_{k mod P} * bump_pair(u; s0=k/K, width)."""
        P = int(base_vec.numel())
        N = 1.0
        for k in range(K):
            wk = base_vec[k % P]

            # Start-aligned bumps (matches your current a=k/K interpretation)
            s0 = k / K

            # If instead you want the PEAK at k/K (since p peaks at x=0.5), use:
            # s0 = k / K - width / 2.0

            N = N + wk * self._bump_pair(
                u, s0=s0, width=width, normalize=self.normalize_bumps
            )
        return N

    def _u(self, s):
        # keep ONE wrap here
        return self._mod1(self.omega * s / (2.0 * ca.pi))

    def azimuth(self, r, s):
        a_phi = self.az_amp(r)
        phi_class = self.sgn * a_phi * ca.sin(self.omega * s)
        u = self._u(s)
        N_phi = self._build_shape_repeat(u, self.K_phi, self.width_phi, self.az_coeffs)
        return phi_class * N_phi

    def elevation(self, r, s):
        c_beta = self.beta_center(r)
        b_beta = self.beta_amp(r)
        beta_class = c_beta + b_beta * ca.sin(2.0 * self.omega * s)
        u = self._u(s)
        N_beta = self._build_shape_repeat(
            u, self.K_beta, self.width_beta, self.beta_coeffs
        )
        return beta_class * N_beta


class Bspline(ParametrizedPatternsAngles):

    # =======================================
    """NO COURSE ANGLE ENFORCEMENT YET"""
    # =======================================

    def __init__(
        self,
        p=3,
        n_ctrl=8,
        r0=300,
        r1=150,
        crs0=(11 / 6) * np.pi,
        crsf=np.pi / 2,
        phi0=0,
        phif=0,
        beta0=0,
        betaf=0,
        C_interior=None,
        u_vals=None,
        U_interior=None,
    ):

        # Fixed attributes
        self.p = p
        self.n_ctrl = n_ctrl
        self.r0 = r0
        self.r1 = r1
        self.crs0 = crs0
        self.crsf = crsf
        self.phi0 = phi0
        self.phif = phif
        self.beta0 = beta0
        self.betaf = betaf
        self.dim = 2  # azimuth, elevation

        # Knot vector
        self.n_knots = self.n_ctrl + self.p + 1
        self.n_interior_knots = self.n_knots - 2 * (self.p + 1)
        if self.n_interior_knots < 0:
            raise ValueError("Too few control points for spline order")

        self.U_interior = (
            np.linspace(0.15, 0.85, self.n_interior_knots + 2)[1:-1]
            if U_interior is None
            else U_interior
        )
        self.U = np.concatenate(
            ([0] * (self.p + 1), self.U_interior, [1] * (self.p + 1))
        )

        self.C_interior = (
            np.ones((self.n_ctrl - 2, self.dim)) if C_interior is None else C_interior
        )
        # Full control points (first & last fixed, interior symbolic)
        self.C = np.vstack(
            [
                np.array([self.phi0, self.beta0]),
                self.C_interior,
                np.array([self.phif, self.betaf]),
            ]
        )

        # Sampling
        self.u_vals = np.linspace(0, 1, 100) if u_vals is None else u_vals

        self.spline_func = self.build_bspline_symbolic()

    # -------------------------------
    # B-spline basis symbolic function
    # -------------------------------
    def Nvec_symbolic(self):
        u_sym = ca.MX.sym("u")
        U_sym = ca.MX.sym("U", self.n_ctrl + self.p + 1)
        n_ctrl = self.n_ctrl
        p = self.p

        def N(i, k, u):
            if k == 0:
                return ca.if_else(
                    ca.logic_and(U_sym[i] <= u, u <= U_sym[i + 1]), 1.0, 0.0
                )
            left = ca.if_else(
                U_sym[i + k] > U_sym[i],
                (u - U_sym[i]) / (U_sym[i + k] - U_sym[i]) * N(i, k - 1, u),
                0,
            )
            right = ca.if_else(
                U_sym[i + k + 1] > U_sym[i + 1],
                (U_sym[i + k + 1] - u)
                / (U_sym[i + k + 1] - U_sym[i + 1])
                * N(i + 1, k - 1, u),
                0,
            )
            return left + right

        Nvec_sym = ca.vertcat(*[N(i, p, u_sym) for i in range(n_ctrl)]).T
        N_func = ca.Function("N_func", [u_sym, U_sym], [Nvec_sym], ["u", "U"], ["Nvec"])
        return N_func

    # -------------------------------
    # Build symbolic spline S(u) = N(u,U)*C
    # -------------------------------
    def build_bspline_symbolic(self, return_derivative=True):
        C_sym = ca.MX.sym("C", self.n_ctrl, self.dim)
        u_sym = ca.MX.sym("u")
        U_sym = ca.MX.sym("U", self.n_ctrl + self.p + 1)

        N_func = self.Nvec_symbolic()
        S_sym = ca.mtimes(N_func(u_sym, U_sym), C_sym)
        dS_sym = ca.jacobian(S_sym, u_sym).T if return_derivative else None

        return ca.Function(
            "spline_func",
            [C_sym, u_sym, U_sym],
            [S_sym, dS_sym],
            ["C", "u", "U"],
            ["S", "dS"],
        )

    def evaluate_spline(self, r, s):
        """Evaluate spline and derivatives simultaneously for efficiency"""
        return self.spline_func(C=self.C, u=s, U=self.U)

    def azimuth(self, r, s):
        res = self.evaluate_spline(r, s)
        return res["S"][0]

    def elevation(self, r, s):
        res = self.evaluate_spline(r, s)
        return res["S"][1]

    # Convenience evaluators that do not require the caller to supply r or a prebuilt s-grid
    def _normalize_s(self, s, s_in_radians):
        if s_in_radians:
            if isinstance(s, (ca.MX, ca.SX, ca.DM)):
                u = s / (2.0 * ca.pi)
                return u - ca.floor(u)  # wrap to [0,1)
            s_arr = np.asarray(s)
            u = (s_arr / (2.0 * np.pi)) % 1.0
            return float(u) if np.ndim(u) == 0 else u
        return s

    def azimuth_at(self, s, s_in_radians=False):
        u = self._normalize_s(s, s_in_radians)
        res = self.spline_func(C=self.C, u=u, U=self.U)
        vals = res["S"][0]
        return float(vals) if vals.numel() == 1 else vals

    def elevation_at(self, s, s_in_radians=False):
        u = self._normalize_s(s, s_in_radians)
        res = self.spline_func(C=self.C, u=u, U=self.U)
        vals = res["S"][1]
        return float(vals) if vals.numel() == 1 else vals

    def azimuth_derivative(self, r, s):
        res = self.evaluate_spline(r, s)
        return res["dS"][0]

    def elevation_derivative(self, r, s):
        res = self.evaluate_spline(r, s)
        return res["dS"][1]

    def curvature_using_bspline_derivatives(self, r_array, s_array):
        """Compute curvature using B-spline derivatives directly"""
        import casadi as ca

        s = ca.MX.sym("s")
        r = ca.MX.sym("r")

        # Get spline values and derivatives
        spline_result = self.spline_func(C=self.C, u=s, U=self.U)
        phi = spline_result["S"][0]  # azimuth
        beta = spline_result["S"][1]  # elevation
        dphi_ds = spline_result["dS"][0]  # azimuth derivative
        dbeta_ds = spline_result["dS"][1]  # elevation derivative

        # Cartesian position
        x = r * ca.cos(beta) * ca.cos(phi)
        y = r * ca.cos(beta) * ca.sin(phi)
        z = r * ca.sin(beta)

        # First derivatives using chain rule (more stable than jacobian)
        dx_ds = r * (
            -ca.sin(beta) * dbeta_ds * ca.cos(phi)
            - ca.cos(beta) * ca.sin(phi) * dphi_ds
        )
        dy_ds = r * (
            -ca.sin(beta) * dbeta_ds * ca.sin(phi)
            + ca.cos(beta) * ca.cos(phi) * dphi_ds
        )
        dz_ds = r * ca.cos(beta) * dbeta_ds

        # Second derivatives (still need jacobian, but only once)
        d2x_ds2 = ca.jacobian(dx_ds, s)
        d2y_ds2 = ca.jacobian(dy_ds, s)
        d2z_ds2 = ca.jacobian(dz_ds, s)

        # Curvature calculation
        r_s = ca.vertcat(dx_ds, dy_ds, dz_ds)
        r_ss = ca.vertcat(d2x_ds2, d2y_ds2, d2z_ds2)

        eps = 1e-12
        cross_rs_rss = ca.cross(r_s, r_ss)
        num = ca.norm_2(cross_rs_rss)
        den = ca.power(ca.norm_2(r_s), 3) + eps
        kappa = num / den

        kappa_fun = ca.Function("kappa_stable", [r, s], [kappa])
        return kappa_fun(r_array, s_array)


import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from math import comb


class CasadiSpline(ParametrizedPatternsAngles):

    # =======================================
    """NO COURSE ANGLE ENFORCEMENT YET"""
    # =======================================

    def __init__(
        self, r0=None, r1=None, C_az=None, C_el=None, s_norm_az=None, s_norm_el=None
    ):

        if r0 is None:
            self.r0 = 322  # m
        else:
            self.r0 = r0

        if r1 is None:
            self.r1 = 240  # m
        else:
            self.r1 = r1

        # Default interior points
        if C_az is None:
            self.C_az = np.deg2rad(
                np.array([-60, -45, -20, 0, 20, 35, 45, 50, 40, 20], dtype=float)
            )
        else:
            self.C_az = C_az

        if C_el is None:
            self.C_el = np.deg2rad(
                np.array([10, 20, 35, 45, 55, 60, 55, 45, 30, 15], dtype=float)
            )
        else:
            self.C_el = C_el

        # ---------- Chord-length parameterization on [0,1] ----------
        if s_norm_az is not None:
            self.s_norm_az = s_norm_az
        else:
            pts = np.vstack([self.C_az, self.C_el]).T
            d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
            s = np.hstack([[0.0], np.cumsum(d)])
            if s[-1] == 0.0:
                s[-1] = 1.0
            self.s_norm_az = s / s[-1]

        if s_norm_el is not None:
            self.s_norm_el = s_norm_el
        else:
            pts = np.vstack([self.C_az, self.C_el]).T
            d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
            s = np.hstack([[0.0], np.cumsum(d)])
            if s[-1] == 0.0:
                s[-1] = 1.0
            self.s_norm_el = s / s[-1]

        self.opts = opts = {"degree": [3]}

        self.build()

    def build(self):
        self.spline_phi = ca.interpolant(
            "spline_phi", "bspline", [self.s_norm_az], self.C_az, self.opts
        )
        self.spline_beta = ca.interpolant(
            "spline_beta", "bspline", [self.s_norm_el], self.C_el, self.opts
        )

    # helpers to evaluate from Python (vectorized)
    def azimuth(self, r, s):
        return self.spline_phi(s)

    def elevation(self, r, s):
        return self.spline_beta(s)


if __name__ == "__main__":
    obj = CasadiSpline()

    s = np.linspace(0, 1, 100)
    az = obj.azimuth(1, s)
    el = obj.elevation(1, s)

    plt.figure()
    plt.plot(s, az)
    plt.plot(s, el)
    plt.show()

    plt.figure()
    plt.plot(az, el)
    plt.show()

    pattern = create_pattern_from_dict(
        "spline",
        {
            "r0": 300,
            "r1": 150,
            "C_az": np.deg2rad(
                np.array([-60, -45, -20, 0, 20, 35, 45, 50, 40, 20], dtype=float)
            ),
            "C_el": np.deg2rad(
                np.array([10, 20, 35, 45, 55, 60, 55, 45, 30, 15], dtype=float)
            ),
            "s_norm_az": np.linspace(0, 1, 10),
            "s_norm_el": np.linspace(0, 1, 10),
        },
    )

    pattern2 = create_pattern_from_dict(
        DEFAULT_SPLINE_PATTERN_CONFIG["pattern_type"],
        DEFAULT_SPLINE_PATTERN_CONFIG["parameters"],
    )


class CST_Helix(ParametrizedPatternsAngles):
    def __init__(
        self,
        r0,
        az_amp0,
        beta_amp0,
        beta0,
        beta_coeffs,
        az_coeffs,
        omega=1.0,
        kappa=0.0,
        kbeta=0.0,
        width_phi=0.5,
        width_beta=0.5,
        phi0=0,
        left_first=True,
        normalize_bumps=False,
        repeat_phi=True,
        repeat_beta=True,
        **kwargs,
    ):  # <- only flags
        super().__init__(
            omega=omega,
            r0=r0,
            az_amp0=az_amp0,
            beta_amp0=beta_amp0,
            beta0=beta0,
            kappa=kappa,
            kbeta=kbeta,
            beta_coeffs=beta_coeffs,
            az_coeffs=az_coeffs,
            width_phi=width_phi,
            width_beta=width_beta,
            left_first=left_first,
            normalize_bumps=normalize_bumps,
            phi0=phi0,
        )

        # Base weight vectors
        self.az_coeffs = ca.vertcat(az_coeffs)
        self.beta_coeffs = ca.vertcat(beta_coeffs)
        self.phi_center = phi0
        P_phi = int(self.az_coeffs.numel())
        P_beta = int(self.beta_coeffs.numel())

        # Total number of bumps = len(weights) or 2× if repeating
        self.K_phi = 2 * P_phi if repeat_phi else P_phi
        self.K_beta = 2 * P_beta if repeat_beta else P_beta

        self.width_phi, self.width_beta = float(width_phi), float(width_beta)
        self.normalize_bumps = bool(normalize_bumps)
        self.sgn = -1.0 if left_first else +1.0

    def beta_center(self, r):
        return self.beta0 * (self.r0 / (self.r0 + (r - self.r0) * self.kbeta))

    def az_amp(self, r):
        return self.az_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    def beta_amp(self, r):
        return self.beta_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    @staticmethod
    def _mod1(x):
        return x - ca.floor(x)

    @staticmethod
    def _p(x):
        # 2nd basis polynomial of 4th-order Bernstein: p(x)=6x^2(1-x)^2
        return 6.0 * (x**2) * ((1.0 - x) ** 2)

    @staticmethod
    def _gate01_smooth_sigmoid(x, k=50.0):
        # k controls sharpness. Larger k -> closer to hard gate.
        s_left = 1.0 / (1.0 + ca.exp(-k * (x - 0.0)))
        s_right = 1.0 / (1.0 + ca.exp(-k * (1.0 - x)))
        return s_left * s_right

    @staticmethod
    def _gate01(x):
        # hard gate: 1 if 0<=x<=1 else 0
        return ca.if_else(ca.logic_and(x >= 0.0, x <= 1.0), 1.0, 0.0)

    def _bump_pair(self, u, s0, width, normalize=False):
        """
        Periodic bump without inner mod1:
        - right bump active for u in [s0, s0+width]
        - left bump is the wrapped copy (shifted by +1)
        """
        x_right = (u - s0) / width
        x_left = (u - s0 + 1.0) / width  # wrap-around copy

        pright = self._gate01(x_right) * self._p(x_right)
        pleft = self._gate01(x_left) * self._p(x_left)

        bump = pright + pleft
        return bump / width if normalize else bump

    def _build_shape_repeat(self, u, K, width, base_vec):
        """N(u) = 1 + Σ_{k=0..K-1} w_{k mod P} * bump_pair(u; s0=k/K, width)."""
        P = int(base_vec.numel())
        N = 1.0
        for k in range(K):
            wk = base_vec[k % P]

            # Start-aligned bumps (matches your current a=k/K interpretation)
            s0 = k / K

            # If instead you want the PEAK at k/K (since p peaks at x=0.5), use:
            # s0 = k / K - width / 2.0

            N = N + wk * self._bump_pair(
                u, s0=s0, width=width, normalize=self.normalize_bumps
            )
        return N

    def _u(self, s):
        # keep ONE wrap here
        return self._mod1(self.omega * s / (2.0 * ca.pi))

    def azimuth(self, r, s):
        c_phi = self.phi_center
        a_phi = self.az_amp(r)
        phi_class = self.sgn * a_phi * ca.sin(self.omega * s)
        u = self._u(s)
        N_phi = self._build_shape_repeat(u, self.K_phi, self.width_phi, self.az_coeffs)
        return c_phi + phi_class * N_phi  # c_phi = 0

    def elevation(self, r, s):
        c_beta = self.beta_center(r)
        b_beta = self.beta_amp(r)
        beta_class = c_beta + b_beta * ca.cos(self.omega * s)
        u = self._u(s)
        N_beta = self._build_shape_repeat(
            u, self.K_beta, self.width_beta, self.beta_coeffs
        )
        return (beta_class) * N_beta


class Reelin_Simple(ParametrizedPatternsAngles):
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


class Transition_Simple(ParametrizedPatternsAngles):
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


class PeriodicBSpline(ParametrizedPatternsAngles):

    def __init__(self, r0, M, C_phi, C_beta, s_init, s_final, downloops=True):
        super().__init__(
            r0=r0, M=M, C_phi=C_phi, C_beta=C_beta, s_init=s_init, s_final=s_final
        )

        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0

        self.spline = build_periodic_cubic_bspline_function(M, dim=1, name="spl")

        self.C_phi = C_phi
        self.C_beta = C_beta

    def _u(self, s):
        return self.omega * (s - self.s_init) / (self.s_final - self.s_init)
        # return s

    def _eval_spline_vec(self, C, u):
        """
        Evaluate spline for scalar u or vector u (column/row).
        Returns:
        - scalar (1x1) if u scalar
        - column (N x 1) if u vector length N
        """
        # If u is scalar -> normal call
        if u.is_scalar():
            return self.spline(C, u)[0]

        # Ensure u is a column vector (N x 1)
        u_col = ca.reshape(u, u.numel(), 1)

        # Map the scalar spline over N points
        N = int(u_col.numel())
        spl_map = self.spline.map(N)  # maps over repeated calls
        S = spl_map(C, u_col)  # shape: (1, N) because output is (1,1) stacked

        # Convert to (N x 1)
        return S.T

    def azimuth(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_phi, u)

    def elevation(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_beta, u)

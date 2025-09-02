import casadi as ca
from abc import ABC, abstractmethod
import numpy as np


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


from picawe.kinematics.reelin_parametrization import ReelInBezier


def create_pattern_from_dict(
    config: dict, optimize: bool = False
) -> ParametrizedPatterns:
    pattern_type = config.get("pattern_type").lower()
    params = config.get("parameters", {})
    optimization_params = config.get("optimization_parameters", {})

    print(params)

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
            "omega",
            "r0",
            "az_amp0",
            "beta_amp0",
            "beta0",
            "beta_coeffs",
            "az_coeffs",
        ],
        "reel_in": [
            "r0",
            "r1",
        ],
    }

    if pattern_type not in required_params:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    missing_params = [
        param for param in required_params[pattern_type] if param not in params
    ]
    if missing_params:
        raise ValueError(
            f"Missing required parameters in 'parameters' for '{pattern_type}': {', '.join(missing_params)}"
        )

    # Replace optimized parameters with symbolic variables
    final_params = params.copy()
    if optimize:
        for param in optimization_params:
            if param in required_params[pattern_type]:

                val = np.atleast_1d(params[param])  # guarantees array, even for scalar
                if len(val) > 1:
                    final_params[param] = ca.MX.sym(param, len(val))
                else:
                    final_params[param] = ca.MX.sym(param)

    # Instantiate the appropriate pattern class
    pattern_classes = {
        "helix": Helix,
        "lissajous": Lissajous,
        "lissajous_angles": LissajousAngles,
        "figure_eight": FigureEight,
        "figure_eight_angles": FigureEightAngles,
        "cst_lissajous": CST_Lissajous,
        "reel_in": ReelInBezier,
    }

    return pattern_classes[pattern_type](**final_params)


class CST_Lissajous(ParametrizedPatternsAngles):
    def __init__(
        self,
        omega,
        r0,
        az_amp0,
        beta_amp0,
        beta0,
        beta_coeffs,
        az_coeffs,
        kappa=0.0,
        kbeta=0.0,
        width_phi=0.5,
        width_beta=0.5,
        left_first=True,
        normalize_bumps=False,
        repeat_phi=False,
        repeat_beta=False,
        k_vr=6300,
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
        )

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

    def _bump(self, u, a, width, normalize=False):
        delta = self._mod1(u - a)
        s = delta / width
        val = 6.0 * (s**2) * ((1.0 - s) ** 2)
        inside = ca.if_else(delta <= width, 1.0, 0.0)
        bump = inside * val
        return bump / width if normalize else bump

    def _build_shape_repeat(self, u, K, width, base_vec):
        """N(u) = 1 + Σ_{k=0..K-1} w_{k mod P} * bump(u; a=k/K, width)."""
        P = int(base_vec.numel())
        N = 1.0
        for k in range(K):
            wk = base_vec[k % P]
            a = k / K
            N = N + wk * self._bump(u, a=a, width=width, normalize=self.normalize_bumps)
        return N

    def _u(self, s):  # unit-phase for shaping
        return self._mod1(self.omega * s / (2.0 * ca.pi))

    def azimuth(self, r, s):
        a_phi = self.az_amp(r)
        phi_class = self.sgn * a_phi * ca.sin(self.omega * s)
        u = self._u(s)
        N_phi = self._build_shape_repeat(u, self.K_phi, self.width_phi, self.az_coeffs)
        return phi_class * N_phi  # c_phi = 0

    def elevation(self, r, s):
        c_beta = self.beta_center(r)
        b_beta = self.beta_amp(r)
        beta_class = c_beta + b_beta * ca.sin(2.0 * self.omega * s)
        u = self._u(s)
        N_beta = self._build_shape_repeat(
            u, self.K_beta, self.width_beta, self.beta_coeffs
        )
        return (beta_class) * N_beta

import casadi as ca
from abc import ABC, abstractmethod
import numpy as np
from math import comb
import matplotlib.pyplot as plt



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

def create_pattern_from_dict(
    config: dict, optimize: bool = False
) -> ParametrizedPatterns:
    pattern_type = config.get("pattern_type").lower()
    params = config.get("parameters", {})
    optimization_params = config.get("optimization_parameters", {})

    # print(params)

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
        "spline": [
            "p",
            "n_ctrl", 
            "r0", 
            "r1", 
            "crs0", 
            "crsf", 
            "phi0", 
            "phif", 
            "beta0", 
            "betaf", 
            "C_interior", 
            "u_vals", 
            "U_interior",
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
        "spline": Bspline,
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

class Bspline(ParametrizedPatternsAngles): 
    
    # =======================================
    """ NO COURSE ANGLE ENFORCEMENT YET """
    # =======================================

    def __init__(self, 
                 p=3, 
                 n_ctrl=8, 
                 r0=300, 
                 r1=150, 
                 crs0=(11/6)*np.pi, 
                 crsf=np.pi/2, 
                 phi0=0, 
                 phif=0, 
                 beta0=0, 
                 betaf=0, 
                 C_interior=None, 
                 u_vals=None, 
                 U_interior=None):

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
        self.n_interior_knots = self.n_knots - 2*(self.p+1)
        if self.n_interior_knots < 0:
            raise ValueError("Too few control points for spline order")

        self.U_interior = np.linspace(0.15, 0.85, self.n_interior_knots+2)[1:-1] if U_interior is None else U_interior
        self.U = np.concatenate(([0]*(self.p+1), self.U_interior, [1]*(self.p+1)))

        self.C_interior = np.ones((self.n_ctrl-2, self.dim)) if C_interior is None else C_interior
        # Full control points (first & last fixed, interior symbolic)
        self.C = np.vstack([np.array([self.phi0, self.beta0]),
                            self.C_interior,
                            np.array([self.phif, self.betaf])])

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
                return ca.if_else(ca.logic_and(U_sym[i] <= u, u <= U_sym[i+1]), 1.0, 0.0)
            left = ca.if_else(U_sym[i+k] > U_sym[i],
                              (u - U_sym[i]) / (U_sym[i+k]-U_sym[i]) * N(i, k-1, u),
                              0)
            right = ca.if_else(U_sym[i+k+1] > U_sym[i+1],
                               (U_sym[i+k+1]-u)/(U_sym[i+k+1]-U_sym[i+1]) * N(i+1, k-1, u),
                               0)
            return left + right

        Nvec_sym = ca.vertcat(*[N(i, p, u_sym) for i in range(n_ctrl)]).T
        N_func = ca.Function("N_func", [u_sym, U_sym], [Nvec_sym], ["u","U"], ["Nvec"])
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

        return ca.Function("spline_func",
                           [C_sym, u_sym, U_sym],
                           [S_sym, dS_sym],
                           ["C","u","U"],
                           ["S","dS"])

    def evaluate_spline(self, r, s):
        """Evaluate spline and derivatives simultaneously for efficiency"""
        return self.spline_func(C=self.C, u=s, U=self.U)

    def azimuth(self, r, s):
        res = self.evaluate_spline(r, s)
        return res["S"][0]

    def elevation(self, r, s):
        res = self.evaluate_spline(r, s)
        return res["S"][1]

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
        phi = spline_result["S"][0]      # azimuth
        beta = spline_result["S"][1]     # elevation
        dphi_ds = spline_result["dS"][0] # azimuth derivative  
        dbeta_ds = spline_result["dS"][1] # elevation derivative
        
        # Cartesian position
        x = r * ca.cos(beta) * ca.cos(phi)
        y = r * ca.cos(beta) * ca.sin(phi)
        z = r * ca.sin(beta)
        
        # First derivatives using chain rule (more stable than jacobian)
        dx_ds = r * (-ca.sin(beta) * dbeta_ds * ca.cos(phi) - ca.cos(beta) * ca.sin(phi) * dphi_ds)
        dy_ds = r * (-ca.sin(beta) * dbeta_ds * ca.sin(phi) + ca.cos(beta) * ca.cos(phi) * dphi_ds)
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
    """
    N-point Bézier spline (2D: azimuth and elevation)
    Fully symbolic using CasADi MX.
    Accepts scalar or vector s directly.
    """

    def __init__(self, r0=None, r1=None, C_az=None, C_el=None, s_norm_az=None, s_norm_el=None):

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
            self.C_az = np.deg2rad(np.array([-60, -45, -20, 0, 20, 35, 45, 50, 40, 20], dtype=float))
        else:
            self.C_az = C_az 

        if C_el is None:
            self.C_el = np.deg2rad(np.array([10, 20, 35, 45, 55, 60, 55, 45, 30, 15], dtype=float))
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
        self.spline_phi = ca.interpolant("spline_phi", "bspline", [self.s_norm_az], self.C_az, self.opts)
        self.spline_beta = ca.interpolant("spline_beta", "bspline", [self.s_norm_el], self.C_el, self.opts)

    
    # helpers to evaluate from Python (vectorized)
    def azimuth(self, r, s):
        return np.array(self.spline_phi(s).full()).ravel()
    
    def elevation(self, r, s):
        return np.array(self.spline_beta(s).full()).ravel()

if __name__ == "__main__":
    obj = CasadiSpline()

    s = np.linspace(0,1,100)
    az = obj.azimuth(1, s)
    el = obj.elevation(1, s)

    plt.figure()
    plt.plot(s,az)
    plt.plot(s,el)
    plt.show()

    plt.figure()
    plt.plot(az, el)
    plt.show()

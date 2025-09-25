# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from picawe.kinematics.Kinematics import ParametrizedKinematics
from picawe.system.system_model import SystemModel
from picawe.kinematics.ReelInBspline_fitting import ReelInBspline_fitting as ribfit

# =========================================================
# Base class: angles-only pattern (radians) + numeric eval
# =========================================================
class ParametrizedPatternsAngles:
    def azimuth(self, s):  # radians, CasADi MX/SX
        raise NotImplementedError

    def elevation(self, s):  # radians, CasADi MX/SX
        raise NotImplementedError

    def eval_angles(self, u_vec):
        u_vec = np.asarray(u_vec).reshape(-1)
        N = u_vec.size
        s = ca.MX.sym("s")
        phi_s, beta_s = self.azimuth(s), self.elevation(s)
        f_ab = ca.Function("f_ab", [s], [phi_s, beta_s]).map(N)
        phi_row, beta_row = f_ab(ca.DM(u_vec).T)
        return np.array(phi_row).ravel(), np.array(beta_row).ravel()

    def eval_xyz(self, u_vec, r_vec):
        u_vec = np.asarray(u_vec).reshape(-1)
        r_vec = np.asarray(r_vec).reshape(-1)
        assert u_vec.shape == r_vec.shape, "u_vec and r_vec must have same length"
        phi, beta = self.eval_angles(u_vec)
        x = r_vec * np.cos(beta) * np.cos(phi)
        y = r_vec * np.cos(beta) * np.sin(phi)
        z = r_vec * np.sin(beta)
        return x, y, z

# -------------------------------
    # something0 or somethingf means the start or end 0 for start and f for final
    # p - point eg. p0 start point
    # v - velocity
    # crs - course
    # idx - index
    # cyc - cycle
    # ri - reel-in
    # ro - reel-out
    # sph - spherical
    # cart - cartesian
# -------------------------------

# -------------------------------
# B-spline class compatible with ParametrizedPatternsAngles
# -------------------------------
class ReelInBspline(ParametrizedPatternsAngles):
    """
    B-spline in φ(u), β(u) (spherical) or x(u),y(u),z(u) (cartesian),
    compatible with ParametrizedPatternsAngles interface.
    """

    def __init__(self, 
                 p=3, 
                 n_ctrl=8, 
                 r0=300, 
                 rf=None, 
                 crs0=(11/6)*np.pi, 
                 crsf=np.pi/2, 
                 phi0=0, 
                 phif=0, 
                 beta0=0, 
                 betaf=0, 
                 C_interior=None, 
                 u_vals=None, 
                 U_interior=None,
                 mode="spherical"):

        self.dim = 2 if mode == "spherical" else 3
        self.p = p
        self.n_ctrl = n_ctrl

        self.r0 = r0
        self.rf = rf if rf is not None else 0.0
        self.crs0 = crs0
        self.crsf = crsf
        self.phi0 = phi0
        self.phif = phif
        self.beta0 = beta0
        self.betaf = betaf

        # knot vector
        self.n_knots = self.n_ctrl + self.p + 1
        self.n_interior_knots = self.n_knots - 2*(self.p+1)
        if self.n_interior_knots < 0:
            raise ValueError("Too few control points for spline order")

        if U_interior is None:
            self.U_interior = np.linspace(0.15, 0.85, self.n_interior_knots+2)[1:-1]
        else:
            self.U_interior = U_interior
        self.U = np.concatenate(([0]*(self.p+1), self.U_interior, [1]*(self.p+1)))

        # control points
        if C_interior is None:
            C_interior = np.ones((self.n_ctrl-2, self.dim))
        self.C_interior = C_interior
        self.C = np.vstack([np.array([self.phi0, self.beta0]),
                            self.C_interior,
                            np.array([self.phif, self.betaf])])

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
        return ca.Function("N_func", [u_sym, U_sym], [Nvec_sym], ["u","U"], ["Nvec"])

    # -------------------------------
    # Build symbolic spline S(u) = N(u,U)*C
    # -------------------------------
    def build_bspline_symbolic(self, return_derivative=True):
        C_sym = ca.MX.sym("C", self.n_ctrl, self.dim)
        u_sym = ca.MX.sym("u")
        U_sym = ca.MX.sym("U", self.n_ctrl + self.p + 1)

        N_func = self.Nvec_symbolic()
        S_sym = ca.mtimes(N_func(u_sym, U_sym), C_sym)
        dS_sym = ca.jacobian(S_sym, u_sym) if return_derivative else None

        return ca.Function("spline_func",
                           [C_sym, u_sym, U_sym],
                           [S_sym, dS_sym],
                           ["C","u","U"],
                           ["S","dS"])

    def azimuth(self, s):
        res = self.spline_func(C=self.C, u=s, U=self.U)
        return res["S"][0]   # φ is first column of spline output

    def elevation(self, s):
        res = self.spline_func(C=self.C, u=s, U=self.U)
        return res["S"][1]   # β is second column of spline output


# =========================================================
# Demo plotting
# =========================================================
def azel_to_vec(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


if __name__ == "__main__":

    fitted = ribfit(
    file_path_full = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv",
    file_path_cycle = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv",
    cyc_idx=0,
    p=3,
    n_ctrl=8,
    c_penalty=1.0,
    v_penalty=0.0,
    eps_knot=1e-3
    )

    # u-grid
    N = 100
    u = np.linspace(0.0, 1.0, N)

    # initial radii
    r_start0, r_end0 = 300.0, 150.0

    # r(u) profiles
    def r_profiles(r0, r1):
        r_lin = r0 + (r1 - r0) * u
        r_quad = r0 + (r1 - r0) * (u**2)
        r_half = np.where(u <= 0.5, r0 + (r1 - r0) * (u / 0.5), r1)
        return r_lin, r_quad, r_half

    # Create pattern instance (update parameters directly here)

    # # Placeholder for C_interior, U_interior and u_vals
    # C_interior = None
    # U_interior = None
    # u_vals = None

    mode = "spherical"  # "spherical" or "cartesian"

    pat = ReelInBspline(
        p=fitted.p,
        n_ctrl=fitted.n_ctrl,
        r0=300.0,
        rf=150.0,
        crs0=fitted.ri_crs0,
        crsf=fitted.ri_crsf,
        phi0=fitted.ri_p0_sph[0],
        phif=fitted.ri_pf_sph[0],
        beta0=fitted.ri_p0_sph[1],
        betaf=fitted.ri_pf_sph[1],
        C_interior=fitted.C_sph[1:-1] if mode=="spherical" else fitted.C_cart[1:-1],
        u_vals=u,
        U_interior=fitted.U_sph[fitted.p+1:-(fitted.p+1)] if mode=="spherical" else fitted.U_cart
    )

    # Create kinematics object
    class Phase:
        def __init__(self):
            self.s = ca.MX.sym("s")
            self.s_dot = ca.MX.sym("s_dot")
            self.s_ddot = ca.MX.sym("s_ddot")
            self.vr = ca.MX.sym("vr")
            self.t = ca.MX.sym("t")
            self.kite_model = SystemModel()

    phase = Phase()
    kinematics = ParametrizedKinematics(pat, phase)

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    axr = fig.add_subplot(222)
    axt = fig.add_subplot(224)
    plt.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.35)

    # r(u) profiles
    r_lin, r_quad, r_half = r_profiles(r_start0, r_end0)

    # Evaluate 3D
    x1, y1, z1 = pat.eval_xyz(u, r_lin)
    x2, y2, z2 = pat.eval_xyz(u, r_quad)
    x3, y3, z3 = pat.eval_xyz(u, r_half)

    # Draw 3D
    ax3d.plot(x1, y1, z1, lw=2, label="r linear")
    ax3d.plot(x2, y2, z2, lw=2, ls="--", label="r quadratic")
    ax3d.plot(x3, y3, z3, lw=2, ls="-.", label="r half→const")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("φ(u), β(u) with r(u) profiles")
    ax3d.legend()
    all_pts = np.vstack([np.c_[x1, y1, z1], np.c_[x2, y2, z2], np.c_[x3, y3, z3]])
    ax3d.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax3d.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax3d.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())

    # r(u) subplot
    axr.plot(u, r_lin, lw=2, label="linear")
    axr.plot(u, r_quad, lw=2, ls="--", label="quadratic")
    axr.plot(u, r_half, lw=2, ls="-.", label="half→const")
    axr.set_xlabel("u")
    axr.set_ylabel("r (m)")
    axr.set_title("r(u) profiles")
    axr.set_xlim(0, 1)
    axr.set_ylim(
        min(r_lin.min(), r_quad.min(), r_half.min()),
        max(r_lin.max(), r_quad.max(), r_half.max()),
    )
    axr.legend()

    # Angles subplot
    phi_u, beta_u = pat.eval_angles(u)
    axt.plot(u, np.rad2deg(phi_u), lw=2, label="φ(u)")
    axt.plot(u, np.rad2deg(beta_u), lw=2, label="β(u)")
    axt.set_xlabel("u")
    axt.set_title("Angles (deg)")
    lo = min(np.rad2deg(phi_u).min(), np.rad2deg(beta_u).min())
    hi = max(np.rad2deg(phi_u).max(), np.rad2deg(beta_u).max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    axt.set_ylim(lo - pad, hi + pad)
    axt.legend()

    plt.show()

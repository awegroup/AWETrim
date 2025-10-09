# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from picawe.kinematics.my_Kinematics import ParametrizedKinematics
from picawe.system.system_model import SystemModel


# =========================================================
# Base class: angles-only pattern (radians) + numeric eval
# =========================================================
class ParametrizedPatternsAngles:
    """
    Subclasses implement:
      - azimuth(s):  φ(u) in radians (CasADi expression)
      - elevation(s): β(u) in radians (CasADi expression)

    Use eval_xyz(u_vec, r_vec) to get numeric arrays for plotting.
    """

    def azimuth(self, s):  # radians, CasADi MX/SX
        raise NotImplementedError

    def elevation(self, s):  # radians, CasADi MX/SX
        raise NotImplementedError

    def eval_angles(self, u_vec):
        """
        Evaluate φ(u), β(u) numerically on u_vec (1D np array).
        Returns (phi, beta) as NumPy arrays, both radians.
        """
        u_vec = np.asarray(u_vec).reshape(-1)
        N = u_vec.size
        s = ca.MX.sym("s")
        r = ca.MX.sym("r")
        phi_s, beta_s = self.azimuth(r, s), self.elevation(r, s)  # scalar MX
        f_ab = ca.Function("f_ab", [s], [phi_s, beta_s]).map(N)  # vectorized
        phi_row, beta_row = f_ab(ca.DM(u_vec).T)  # (1xN) DM
        return np.array(phi_row).ravel(), np.array(beta_row).ravel()

    def eval_xyz(self, u_vec, r_vec):
        """
        Numeric evaluation for plotting.
        u_vec, r_vec: 1D numpy arrays of same length.
        """
        u_vec = np.asarray(u_vec).reshape(-1)
        r_vec = np.asarray(r_vec).reshape(-1)
        assert u_vec.shape == r_vec.shape, "u_vec and r_vec must have same length"
        phi, beta = self.eval_angles(u_vec)  # NumPy arrays (rad)
        x = r_vec * np.cos(beta) * np.cos(phi)
        y = r_vec * np.cos(beta) * np.sin(phi)
        z = r_vec * np.sin(beta)
        return x, y, z


# =========================================================
# Direction-matched Bézier angles (all radians)
# =========================================================
class ReelInBezier(ParametrizedPatternsAngles):
    """
    φ(u), β(u) are quintic Bézier in u∈[0,1].
    Endpoint angular slopes match the *tangential* component of desired
    Cartesian directions at u=0 and u=1 (does not need r′).
    Set tangent_only=False + ru0, ru1 to do the exact endpoint match.
    """

    def __init__(
        self,
        # radii at ends (meters) — only to place the tangent plane
        r0,
        r1=None,
        # angle endpoint values (radians)
        phi0=0,
        phi5=np.deg2rad(-30.0),
        beta0=np.pi / 6,
        beta5=np.pi / 6,
        # interior controls (radians)
        phi2=np.deg2rad(128.0),
        phi3=np.deg2rad(116.0),
        beta2=np.deg2rad(40.0),
        beta3=np.deg2rad(100.0),
        # desired unit tangents at u=0,1 in Cartesian
        vhat0=(0.0, 0.0, 0.0),
        vhat1=(0.0, 0.0, 0.0),
        # matching mode
        tangent_only=True,
        ru0=0.0,
        ru1=0.0,  # only used if tangent_only=False
    ):
        self.r0 = ca.MX(r0)
        self.r1 = ca.MX(r1 if (r1 is not None) else r0)

        self.phi0 = ca.MX(phi0)
        self.phi5 = ca.MX(phi5)
        self.beta0 = ca.MX(beta0)
        self.beta5 = ca.MX(beta5)
        self.phi2 = ca.MX(phi2)
        self.phi3 = ca.MX(phi3)
        self.beta2 = ca.MX(beta2)
        self.beta3 = ca.MX(beta3)

        self.vhat0 = ca.MX(vhat0)
        self.vhat1 = ca.MX(vhat1)
        self.tangent_only = bool(tangent_only)
        self.ru0 = ca.MX(ru0)
        self.ru1 = ca.MX(ru1)

    # ---------- public API ----------
    def azimuth(self, r, s):  # radians
        phi_u, _ = self._angles_at_u(s)
        return phi_u

    def elevation(self, r, s):  # radians
        _, beta_u = self._angles_at_u(s)
        return beta_u

    # ---------- internals ----------
    @staticmethod
    def _bern5_mat(s):
        s = ca.reshape(s, -1, 1)
        one_u = 1 - s
        cols = [
            one_u**5,
            5 * s * one_u**4,
            10 * s**2 * one_u**3,
            10 * s**3 * one_u**2,
            5 * s**4 * one_u,
            s**5,
        ]
        return ca.hcat(cols)  # (N,6)

    @staticmethod
    def _endpoint_slopes_tangent_only(phi_rad, beta_rad, r, vhat, eps=1e-9):
        cb, sb = ca.cos(beta_rad), ca.sin(beta_rad)
        caa, saa = ca.cos(phi_rad), ca.sin(phi_rad)
        Jr = ca.vcat([cb * caa, cb * saa, sb])  # ∂p/∂r
        Jb = ca.vcat([-r * sb * caa, -r * sb * saa, r * cb])  # ∂p/∂beta
        Jp = ca.vcat([-r * cb * saa, r * cb * caa, 0.0])  # ∂p/∂phi
        vhat = vhat / (ca.norm_2(vhat) + 1e-16)
        er = Jr / (ca.norm_2(Jr) + 1e-16)
        v_tan = vhat - ca.dot(vhat, er) * er  # tangent component
        A = ca.hcat([Jb, Jp])  # 3x2
        ATA = A.T @ A + eps * ca.DM_eye(2)
        ATv = A.T @ v_tan
        x = ca.solve(ATA, ATv)  # (2,1)
        beta_u, phi_u = x[0], x[1]  # radians per unit-u
        return phi_u, beta_u

    @staticmethod
    def _endpoint_slopes_with_ru(phi_rad, beta_rad, r, rprime_u, vhat):
        cb, sb = ca.cos(beta_rad), ca.sin(beta_rad)
        caa, saa = ca.cos(phi_rad), ca.sin(phi_rad)
        Jr = ca.vcat([cb * caa, cb * saa, sb])
        Jb = ca.vcat([-r * sb * caa, -r * sb * saa, r * cb])
        Jp = ca.vcat([-r * cb * saa, r * cb * caa, 0.0])
        vhat = vhat / (ca.norm_2(vhat) + 1e-16)
        A = ca.hcat([Jb, Jp, -vhat])  # 3x3
        b = -Jr * rprime_u
        x = ca.solve(A, b)  # (3,1)
        beta_u, phi_u = x[0], x[1]
        return phi_u, beta_u

    def _endpoint_slopes(self):
        if self.tangent_only:
            phi_u0, beta_u0 = self._endpoint_slopes_tangent_only(
                self.phi0, self.beta0, self.r0, self.vhat0
            )
            phi_u1, beta_u1 = self._endpoint_slopes_tangent_only(
                self.phi5, self.beta5, self.r1, self.vhat1
            )
        else:
            phi_u0, beta_u0 = self._endpoint_slopes_with_ru(
                self.phi0, self.beta0, self.r0, self.ru0, self.vhat0
            )
            phi_u1, beta_u1 = self._endpoint_slopes_with_ru(
                self.phi5, self.beta5, self.r1, self.ru1, self.vhat1
            )
        return phi_u0, beta_u0, phi_u1, beta_u1

    def _angles_at_u(self, s):
        phi_u0, beta_u0, phi_u1, beta_u1 = self._endpoint_slopes()
        phi_ctrl = ca.vcat(
            [
                self.phi0,
                self.phi0 + phi_u0 / 5.0,
                self.phi2,
                self.phi3,
                self.phi5 - phi_u1 / 5.0,
                self.phi5,
            ]
        )
        beta_ctrl = ca.vcat(
            [
                self.beta0,
                self.beta0 + beta_u0 / 5.0,
                self.beta2,
                self.beta3,
                self.beta5 - beta_u1 / 5.0,
                self.beta5,
            ]
        )
        B = self._bern5_mat(s)  # (N,6)
        phi_u = B @ phi_ctrl  # (N,1)
        beta_u = B @ beta_ctrl  # (N,1)
        return phi_u, beta_u


# =========================================================
# Demo script
# =========================================================
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def azel_to_vec(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


if __name__ == "__main__":
    # u-grid
    N = 400
    u = np.linspace(0.0, 1.0, N)

    # initial radii
    r_start0, r_end0 = 300.0, 150.0

    # r(u) profiles
    def r_profiles(r0, r1):
        r_lin = r0 + (r1 - r0) * u
        r_quad = r0 + (r1 - r0) * (u**2)
        r_half = np.where(u <= 0.5, r0 + (r1 - r0) * (u / 0.5), r1)
        return r_lin, r_quad, r_half

    # Initial angle controls (deg for sliders; we convert to rad)
    init = dict(
        phi0=0.0,
        phi2=140.0,
        phi3=80.0,
        phi5=-30.0,
        beta0=35.0,
        beta2=80.0,
        beta3=100.0,
        beta5=35.0,
        v0_az=-90.0,
        v0_el=0.0,  # start tangent
        v1_az=90.0,
        v1_el=0.0,  # end tangent
    )

    # Figure & axes
    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    axr = fig.add_subplot(222)
    axt = fig.add_subplot(224)
    plt.subplots_adjust(bottom=0.34, wspace=0.25, hspace=0.35)

    # Lines
    (line_lin,) = ax3d.plot([], [], [], lw=2, label="r linear")
    (line_quad,) = ax3d.plot([], [], [], lw=2, ls="--", label="r quadratic")
    (line_half,) = ax3d.plot([], [], [], lw=2, ls="-.", label="r half→const")
    ax3d.legend()
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("Interactive φ(u), β(u) with three r(u)")

    axr.set_xlabel("u")
    axr.set_ylabel("r (m)")
    axr.set_title("r(u) profiles")
    (rp_lin,) = axr.plot([], [], lw=2, label="linear")
    (rp_quad,) = axr.plot([], [], lw=2, ls="--", label="quadratic")
    (rp_half,) = axr.plot([], [], lw=2, ls="-.", label="half→const")
    axr.legend()

    axt.set_xlabel("u")
    axt.set_title("Angles (deg)")
    (ang_phi,) = axt.plot([], [], lw=2, label="φ(u)")
    (ang_beta,) = axt.plot([], [], lw=2, label="β(u)")
    axt.legend()

    # Sliders layout
    axcolor = "lightgoldenrodyellow"
    w, h = 0.22, 0.03
    col1_x, col2_x, col3_x = 0.08, 0.38, 0.68
    yy = [0.28, 0.24, 0.20, 0.16]  # four rows

    # Angle sliders (deg)
    s_phi0 = Slider(
        plt.axes([col1_x, yy[0], w, h], facecolor=axcolor),
        "φ0 (°)",
        -180,
        180,
        valinit=init["phi0"],
    )
    s_phi2 = Slider(
        plt.axes([col1_x, yy[1], w, h], facecolor=axcolor),
        "φ2 (°)",
        -180,
        180,
        valinit=init["phi2"],
    )
    s_phi3 = Slider(
        plt.axes([col1_x, yy[2], w, h], facecolor=axcolor),
        "φ3 (°)",
        -180,
        180,
        valinit=init["phi3"],
    )
    s_phi5 = Slider(
        plt.axes([col1_x, yy[3], w, h], facecolor=axcolor),
        "φ5 (°)",
        -180,
        180,
        valinit=init["phi5"],
    )

    s_bet0 = Slider(
        plt.axes([col2_x, yy[0], w, h], facecolor=axcolor),
        "β0 (°)",
        -90,
        90,
        valinit=init["beta0"],
    )
    s_bet2 = Slider(
        plt.axes([col2_x, yy[1], w, h], facecolor=axcolor),
        "β2 (°)",
        -180,
        180,
        valinit=init["beta2"],
    )
    s_bet3 = Slider(
        plt.axes([col2_x, yy[2], w, h], facecolor=axcolor),
        "β3 (°)",
        -180,
        180,
        valinit=init["beta3"],
    )
    s_bet5 = Slider(
        plt.axes([col2_x, yy[3], w, h], facecolor=axcolor),
        "β5 (°)",
        -90,
        90,
        valinit=init["beta5"],
    )

    # Radii sliders
    s_r0 = Slider(
        plt.axes([col3_x, yy[0], w, h], facecolor=axcolor),
        "r0",
        50,
        800,
        valinit=r_start0,
    )
    s_r1 = Slider(
        plt.axes([col3_x, yy[1], w, h], facecolor=axcolor),
        "r1",
        50,
        800,
        valinit=r_end0,
    )

    # Direction sliders (deg)
    s_v0az = Slider(
        plt.axes([col3_x, yy[2], w, h], facecolor=axcolor),
        "v0 az (°)",
        -180,
        180,
        valinit=init["v0_az"],
    )
    s_v0el = Slider(
        plt.axes([col3_x, yy[3], w, h], facecolor=axcolor),
        "v0 el (°)",
        -90,
        90,
        valinit=init["v0_el"],
    )

    s_v1az = Slider(
        plt.axes([col2_x, 0.12, w, h], facecolor=axcolor),
        "v1 az (°)",
        -180,
        180,
        valinit=init["v1_az"],
    )
    s_v1el = Slider(
        plt.axes([col1_x, 0.12, w, h], facecolor=axcolor),
        "v1 el (°)",
        -90,
        90,
        valinit=init["v1_el"],
    )

    # Update function
    def update(_=None):
        # Read sliders; convert deg->rad for angles
        r0 = s_r0.val
        r1 = s_r1.val
        phi0 = np.deg2rad(s_phi0.val)
        phi2 = np.deg2rad(s_phi2.val)
        phi3 = np.deg2rad(s_phi3.val)
        phi5 = np.deg2rad(s_phi5.val)
        beta0 = np.deg2rad(s_bet0.val)
        beta2 = np.deg2rad(s_bet2.val)
        beta3 = np.deg2rad(s_bet3.val)
        beta5 = np.deg2rad(s_bet5.val)

        vhat0 = azel_to_vec(s_v0az.val, s_v0el.val)
        vhat1 = azel_to_vec(s_v1az.val, s_v1el.val)

        # Build pattern
        pat = ReelInBezier(
            r0=r0,
            r1=r1,
            phi0=phi0,
            phi2=phi2,
            phi3=phi3,
            phi5=phi5,
            beta0=beta0,
            beta2=beta2,
            beta3=beta3,
            beta5=beta5,
            vhat0=vhat0,
            vhat1=vhat1,
            # tangent_only=True,
        )

        # r(u) profiles
        r_lin, r_quad, r_half = r_profiles(r0, r1)

        # Evaluate 3D
        x1, y1, z1 = pat.eval_xyz(u, r_lin)
        x2, y2, z2 = pat.eval_xyz(u, r_quad)
        x3, y3, z3 = pat.eval_xyz(u, r_half)

        # Draw 3D
        line_lin.set_data(x1, y1)
        line_lin.set_3d_properties(z1)
        line_quad.set_data(x2, y2)
        line_quad.set_3d_properties(z2)
        line_half.set_data(x3, y3)
        line_half.set_3d_properties(z3)

        # Autoscale 3D
        all_pts = np.vstack([np.c_[x1, y1, z1], np.c_[x2, y2, z2], np.c_[x3, y3, z3]])
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        ax3d.set_xlim(mins[0], maxs[0])
        ax3d.set_ylim(mins[1], maxs[1])
        ax3d.set_zlim(mins[2], maxs[2])

        # r(u) subplot
        rp_lin.set_data(u, r_lin)
        rp_quad.set_data(u, r_quad)
        rp_half.set_data(u, r_half)
        axr.set_xlim(0, 1)
        axr.set_ylim(
            min(r_half.min(), r_quad.min(), r_lin.min()),
            max(r_half.max(), r_quad.max(), r_lin.max()),
        )

        # Angles subplot (deg)
        phi_u, beta_u = pat.eval_angles(u)
        ang_phi.set_data(u, np.rad2deg(phi_u))
        ang_beta.set_data(u, np.rad2deg(beta_u))
        axt.set_xlim(0, 1)
        lo = min(np.rad2deg(beta_u).min(), np.rad2deg(phi_u).min())
        hi = max(np.rad2deg(beta_u).max(), np.rad2deg(phi_u).max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        axt.set_ylim(lo - pad, hi + pad)

        fig.canvas.draw_idle()

    # Wire sliders
    for s in [
        s_phi0,
        s_phi2,
        s_phi3,
        s_phi5,
        s_bet0,
        s_bet2,
        s_bet3,
        s_bet5,
        s_r0,
        s_r1,
        s_v0az,
        s_v0el,
        s_v1az,
        s_v1el,
    ]:
        s.on_changed(update)

    update()
    plt.show()

    # Create empty class
    class Phase:
        def __init__(self):
            self.s = ca.MX.sym("s")
            self.s_dot = ca.MX.sym("s_dot")
            self.s_ddot = ca.MX.sym("s_ddot")
            self.vr = ca.MX.sym("vr")
            self.t = ca.MX.sym("t")
            self.kite_model = SystemModel()
            # self.kite_model.distance_radial = ca.MX.sym("distance_radial")
            # self.kite_model.speed_radial = ca.MX.sym("speed_radial")

    phase = Phase()

    # Build pattern
    pat = ReelInBezier(
        r0=300,
        r1=230,
        # phi0=phi0,
        # phi2=phi2,
        # phi3=phi3,
        # phi5=phi5,
        # beta0=beta0,
        # beta2=beta2,
        # beta3=beta3,
        # beta5=beta5,
        # vhat0=vhat0,
        # vhat1=vhat1,
        # tangent_only=True,
    )
    kinematics = ParametrizedKinematics(pat, phase)

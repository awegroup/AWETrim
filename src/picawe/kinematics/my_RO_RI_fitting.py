from picawe.kinematics.Bspline_build import Bspline_build
from picawe.kinematics.my_RO_RI_data_processing import RO_RI_data_processing
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

class RO_RI_fitting(RO_RI_data_processing, Bspline_build):
    def __init__(self, file_path_full=None, file_path_cycle=None, cyc_idx=0, p=3, n_ctrl=6,
                 c_penalty=0.0, v_penalty=0.0, eps_knot=1e-3):
        super().__init__(file_path_full, file_path_cycle, cyc_idx)

        self.p = p
        self.n_ctrl = n_ctrl
        self.c_penalty = c_penalty
        self.v_penalty = v_penalty
        self.eps_knot = eps_knot

        self.find_start_RO_RI_idx()

        self._compute_u_vals()

        self.fit_RO_RI("spherical")
        self.fit_RO_RI("cartesian")

    def _compute_u_vals(self):
        self.S_sph = np.array(self.RO_RI_sph).T
        self.S_cart = np.array(self.RO_RI_cart).T

        print(len(self.S_cart))

        dist = np.cumsum(np.linalg.norm(np.diff(self.S_cart, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        self.u_vals = dist / dist[-1]

    def compute_course_average_sph(self, dS_sph, S_sph, start=True):
        k = 10
        if start:
            course = -np.arctan2(dS_sph[:k,0] * np.cos(S_sph[:k,1]), dS_sph[:k,1]) + 2*np.pi
        else:
            course = -np.arctan2(dS_sph[-k:,0] * np.cos(S_sph[-k:,1]), dS_sph[-k:,1]) + 2*np.pi
        return np.mean(np.mod(course, 2*np.pi))

    def compute_velocity_average_cart(self, dS_cart, start=True):
        k = 10
        if start:
            velocity = dS_cart[:k,:]
        else:
            velocity = dS_cart[-k:,:]
        return np.mean(velocity, axis=0)

    def fit_RO_RI(self, mode="spherical"):
        # Implement the fitting procedure for the RI to RO transition

        if mode == "spherical":
            self.p0 = self.RO_RI_p0_sph
            self.pf = self.RO_RI_pf_sph
            self.S_data = self.S_sph# shape (N, 2)
            self.dim = 2
        elif mode == "cartesian":
            self.p0 = self.RO_RI_p0_cart
            self.pf = self.RO_RI_pf_cart
            self.S_data = self.S_cart# shape (N, 3)
            self.dim = 3

        # inital guess:

        n_knots = self.n_ctrl + self.p + 1
        n_interior_knots = n_knots - 2 * (self.p +1)

        U_interior_0 = np.linspace(0.1, 0.3, n_interior_knots)
        U0 = np.concatenate([[0]*(self.p+1), U_interior_0, [1]*(self.p+1)])

        C_interior_0 = np.zeros((self.n_ctrl-2, self.dim))
        C0 = np.vstack([self.p0, C_interior_0, self.pf])

        du0 = np.ones(self.n_ctrl - self.p - 1) * 0.2

        lb_C = np.full_like(C0.ravel()[self.dim:-self.dim], -np.inf)
        ub_C = np.full_like(C0.ravel()[self.dim:-self.dim], np.inf)
        lb_du = np.full_like(du0, self.eps_knot)
        ub_du = np.full_like(du0, 1 - self.eps_knot)
        lb = np.concatenate([lb_C, lb_du])
        ub = np.concatenate([ub_C, ub_du])

        builder = Bspline_build()
        builder.n_ctrl = self.n_ctrl
        builder.p = self.p
        builder.dim = self.dim
        builder.u_vals = self.u_vals
        builder.U = U0
        builder.C = C0

        spline_func = builder.build_bspline_symbolic()

        S, dS = builder.eval_spline(spline_func, C0, U0)

        x0 = np.concatenate([C_interior_0.ravel(), du0])

        def residuals(params):
            C_inner = params[:(self.n_ctrl-2)* self.dim].reshape(self.n_ctrl - 2, self.dim)
            C = np.vstack([self.p0, C_inner, self.pf])
            builder.C = C

            du = params[(self.n_ctrl - 2) * self.dim :]
            U_interior = np.cumsum(du)
            U_interior /= U_interior[-1] + 0.1
            U = np.concatenate(([0] * (self.p + 1), U_interior, [1] * (self.p + 1)))
            builder.U = U
            builder.u_vals = self.u_vals

            S_fit, dS_fit = builder.eval_spline(spline_func, C, U)

            res_data = (S_fit - self.S_data).ravel()

            # if mode == "spherical":
            #     avg_start = self.compute_course_average_sph(dS_fit, S_fit, start=True)
            #     avg_end = self.compute_course_average_sph(dS_fit, S_fit, start=False)
            #     res_extra = np.array(
            #         [self.ri_crs0 - avg_start, self.ri_crsf - avg_end]
            #     ) * self.c_penalty
            # else:
            #     vel_start = self.compute_velocity_average_cart(dS_fit, start=True)
            #     vel_end = self.compute_velocity_average_cart(dS_fit, start=False)
            #     res_extra = np.ravel(
            #         np.array([vel_start - self.ri_v0, vel_end - self.ri_vf])
            #     ) * self.v_penalty

            # For now I ignore res_extra because I need to recompute the crs and velocity at the start and end of RO_RI segment and I am too lazy

            return res_data # np.concatenate([res_data, res_extra])
        
        res = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            verbose=2,
        )

        C_opt = res.x[: (self.n_ctrl - 2) * self.dim].reshape(self.n_ctrl - 2, self.dim)
        C_opt = np.vstack([self.p0, C_opt, self.pf])
        du_opt = res.x[(self.n_ctrl - 2) * self.dim :]
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt /= U_interior_opt[-1] + 0.1
        U_opt = np.concatenate(([0] * (self.p + 1), U_interior_opt, [1] * (self.p + 1)))        

        if mode == "spherical":
            self.C_sph, self.U_sph = C_opt, U_opt
            return self.C_sph, self.u_vals, self.U_sph
        else:
            self.C_cart, self.U_cart = C_opt, U_opt
            return self.C_cart, self.u_vals, self.U_cart

    def plot_fitted_path(self, mode="cartesian"):
        # Evaluate numerically

        builder = Bspline_build()
        builder.n_ctrl = self.n_ctrl
        builder.p = self.p
        builder.u_vals = self.u_vals

        if mode == "cartesian":
            builder.C = self.C_cart
            self.p0 = self.RO_RI_p0_cart
            self.pf = self.RO_RI_pf_cart
            builder.U = self.U_cart
            builder.dim = 3
        elif mode == "spherical":
            builder.C = self.C_sph
            self.p0 = self.RO_RI_p0_sph
            self.pf = self.RO_RI_pf_sph
            builder.U = self.U_sph
            builder.dim = 2

        spline_func = builder.build_bspline_symbolic()

        S_fit, _ = builder.eval_spline(spline_func)

        if mode == "cartesian":
            # 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(self.x_cyc, self.y_cyc, self.z_cyc, label="Trajectory", alpha=0.6)
            ax.plot(S_fit[:,0], S_fit[:,1], S_fit[:,2], "r--", label="B-spline fit")

            # Control points
            ax.scatter(self.C_cart[:,0], self.C_cart[:,1], self.C_cart[:,2], color="black", s=30, label="Control points")
            ax.scatter(*self.p0, color="green", s=30, label="ri start")
            ax.scatter(*self.pf, color="red", s=30, label="ri end")

            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.legend(); ax.set_box_aspect([1,1,1])
            plt.show()
        
        if mode == "spherical":
            # Plot azimuth and elevation
            fig, (ax_az, ax_el) = plt.subplots(1, 2, figsize=(10,4), sharex=True)

            ax_az.plot(np.linspace(0, len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0), len(self.az_cyc)), self.az_cyc, label="Azimuth (data)", color="C0")
            ax_az.plot(self.u_vals + (self.RO_RI_idx0/len(self.az_cyc))*(len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0)), S_fit[:,0], "--", label="Azimuth (spline)", color="C1")
            ax_el.plot(np.linspace(0, len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0), len(self.az_cyc)), self.el_cyc, label="Elevation (data)", color="C0")
            ax_el.plot(self.u_vals + (self.RO_RI_idx0/len(self.az_cyc))*(len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0)), S_fit[:,1], "--", label="Elevation (spline)", color="C1")

            # Control points
            u_cp = np.linspace(self.u_vals[0] + (self.RO_RI_idx0/len(self.az_cyc))*(len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0)), 
                               self.u_vals[-1]+ (self.RO_RI_idx0/len(self.az_cyc))*(len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0)), 
                                len(self.C_sph))
            ax_az.scatter(u_cp, self.C_sph[:,0], color="black", s=30, label="Control points")
            ax_el.scatter(u_cp, self.C_sph[:,1], color="black", s=30, label="Control points")

            # RO_RI start/end
            ax_az.scatter(self.u_vals[0] + (self.RO_RI_idx0/len(self.az_cyc))*(len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0)), self.p0[0], color="green", s=30, label="ri start")
            ax_az.scatter(self.u_vals[-1] + (self.RO_RI_idx0/len(self.az_cyc))*(len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0)), self.pf[0], color="red", s=30, label="ri end")
            ax_el.scatter(self.u_vals[0] + (self.RO_RI_idx0/len(self.az_cyc))*(len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0)), self.p0[1], color="green", s=30, label="ri start")
            ax_el.scatter(self.u_vals[-1] + (self.RO_RI_idx0/len(self.az_cyc))*(len(self.az_cyc)/(self.RO_RI_idxf-self.RO_RI_idx0)), self.pf[1], color="red", s=30, label="ri end")

            ax_az.set_xlabel("u"); ax_az.set_ylabel("Azimuth [rad]"); ax_az.grid(True, alpha=0.3); ax_az.legend()
            ax_el.set_xlabel("u"); ax_el.set_ylabel("Elevation [rad]"); ax_el.grid(True, alpha=0.3); ax_el.legend()
            fig.tight_layout()
            plt.show()

if __name__ == "__main__":
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"

    obj = RO_RI_fitting(file_path_full=full_path, file_path_cycle=cycle_path, cyc_idx=0, p=3, n_ctrl=6,
                 c_penalty=0.0, v_penalty=0.0, eps_knot=1e-3)
    
    obj.plot_fitted_path(mode="cartesian")
    obj.plot_fitted_path(mode="spherical")
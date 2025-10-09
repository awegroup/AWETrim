import numpy as np
from scipy.optimize import least_squares
from picawe.kinematics.my_RI_data_processing import RI_data_processing as ribdata
from picawe.kinematics.my_parametrized_patterns import Bspline as Bspline_build
import casadi as ca

class RI_fitting(ribdata, Bspline_build):
    """
    B-spline fitting class using CasADi, fully leveraging the existing build class.
    """

    def __init__(self, file_path_full, file_path_cycle, cyc_idx, p, n_ctrl,
                 c_penalty=1.0, v_penalty=1.0, eps_knot=1e-3):

        super().__init__(file_path_full, file_path_cycle, cyc_idx)

        self.p = p
        self.n_ctrl = n_ctrl
        self.c_penalty = c_penalty
        self.v_penalty = v_penalty
        self.eps_knot = eps_knot

        # parameter along curve
        self._compute_u_vals()

        # Fit splines using CasADi
        self.C_sph, self.u_vals, self.U_sph = self.fit("spherical")
        self.C_cart, self.u_vals, self.U_cart = self.fit("cartesian")

    # -------------------------------
    # Average course and velocity for boundary conditions
    # -------------------------------
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

    def _compute_u_vals(self):
        self.S_sph = np.vstack([self.az_ri, self.el_ri]).T
        self.S_cart = np.vstack([self.x_ri, self.y_ri, self.z_ri]).T

        dist = np.cumsum(np.linalg.norm(np.diff(self.S_cart, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        self.u_vals = dist / dist[-1]

    # ----------------------------------------
    def fit(self, mode="spherical"):
        """
        CasADi least-squares fitting using the updated build class.
        """
        # Dimension & data
        dim = 2 if mode == "spherical" else 3
        S_data = self.S_sph if mode == "spherical" else self.S_cart
        ri_p0 = self.ri_p0_sph if mode == "spherical" else self.ri_p0_cart
        ri_pf = self.ri_pf_sph if mode == "spherical" else self.ri_pf_cart

        # -------- builder instance --------
        builder = Bspline_build()
        builder.n_ctrl = self.n_ctrl
        builder.p = self.p
        builder.dim = dim
        builder.u_vals = self.u_vals

        spline_func = builder.build_bspline_symbolic()

        #initial_values
        U_inner_0 = np.linspace(0.15, 0.85, self.n_ctrl - self.p - 1 + 2)[1:-1]
        U0 = np.concatenate(([0] * (self.p + 1), U_inner_0, [1] * (self.p + 1)))

        C0_inner = np.zeros((self.n_ctrl - 2, dim))
        C0 = np.vstack([ri_p0, C0_inner, ri_pf])

        du0 = np.ones(self.n_ctrl - self.p - 1) * 0.2

        lb_C = np.full_like(C0.ravel()[dim:-dim], -np.inf)
        ub_C = np.full_like(C0.ravel()[dim:-dim], np.inf)
        lb_du = np.full_like(du0, self.eps_knot)
        ub_du = np.full_like(du0, 1 - self.eps_knot)
        lb = np.concatenate([lb_C, lb_du])
        ub = np.concatenate([ub_C, ub_du])

        builder.U = U0
        builder.C = C0

        S, dS = builder.eval_spline(spline_func, C0, U0)

        x0 = np.concatenate([C0_inner.ravel(), du0])

        def residuals(params):
            C_inner = params[: (self.n_ctrl - 2) * dim].reshape(self.n_ctrl - 2, dim)
            C = np.vstack([ri_p0, C_inner, ri_pf])
            builder.C = C

            du = params[(self.n_ctrl - 2) * dim :]
            U_interior = np.cumsum(du)
            U_interior /= U_interior[-1] + 0.1
            U = np.concatenate(([0] * (self.p + 1), U_interior, [1] * (self.p + 1)))
            builder.U = U

            S_fit, dS_fit = builder.eval_spline(spline_func, C, U)

            res_data = (S_fit - S_data).ravel()

            if mode == "spherical":
                avg_start = self.compute_course_average_sph(dS_fit, S_fit, start=True)
                avg_end = self.compute_course_average_sph(dS_fit, S_fit, start=False)
                res_extra = np.array(
                    [self.ri_crs0 - avg_start, self.ri_crsf - avg_end]
                ) * self.c_penalty
            else:
                vel_start = self.compute_velocity_average_cart(dS_fit, start=True)
                vel_end = self.compute_velocity_average_cart(dS_fit, start=False)
                res_extra = np.ravel(
                    np.array([vel_start - self.ri_v0, vel_end - self.ri_vf])
                ) * self.v_penalty

            return np.concatenate([res_data, res_extra])
        
        res = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            verbose=2,
        )

        C_opt = res.x[: (self.n_ctrl - 2) * dim].reshape(self.n_ctrl - 2, dim)
        C_opt = np.vstack([ri_p0, C_opt, ri_pf])
        du_opt = res.x[(self.n_ctrl - 2) * dim :]
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt /= U_interior_opt[-1] + 0.1
        U_opt = np.concatenate(([0] * (self.p + 1), U_interior_opt, [1] * (self.p + 1)))        

        if mode == "spherical":
            self.C_sph, self.U_sph = C_opt, U_opt
            return self.C_sph, self.u_vals, self.U_sph
        else:
            self.C_cart, self.U_cart = C_opt, U_opt
            return self.C_cart, self.u_vals, self.U_cart

if __name__ == "__main__":
    # Test fitting class
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"
    cyc_idx = 0
    p = 3
    n_ctrl = 8
    c_penalty = 1
    v_penalty = 0
    eps_knot = 0.001

    fitter = RI_fitting(full_path, cycle_path, cyc_idx,
                                   p, n_ctrl, c_penalty, v_penalty, eps_knot)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

class ReelInBspline_fitting():
    def __init__(self, p, n_ctrl, course_penalty=1.0, vel_penalty=0.0, eps_knot=1e-3):
        self.p = p
        self.n_ctrl = n_ctrl
        self.course_penalty = course_penalty
        self.vel_penalty = vel_penalty
        self.eps_knot = eps_knot

    def u_vals(self):
        S_cart_for_param = np.vstack([self.x_RI, self.y_RI, self.z_RI]).T
        dist = np.cumsum(np.linalg.norm(np.diff(S_cart_for_param, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        self.u_vals = dist / dist[-1]

    def U_and_C_0(self):
        self.number_of_knots = self.n_ctrl + self.p + 1
        self.n_interior_knots = (self.number_of_knots - 2*(self.p+1))

        if self.n_interior_knots <= 0:
            raise ValueError("Too few control points for spline order")

        U_interior_0 = np.linspace(0.15, 0.85, self.n_interior_knots + 2)[1:-1]
        self.U_0 = np.concatenate(([0]*(self.p+1), U_interior_0, [1]*(self.p+1)))

        self.ri_start_sph = np.array([self.az_RI[0], self.el_RI[0]])
        self.ri_end_sph   = np.array([self.az_RI[-1], self.el_RI[-1]])

        self.C_inner_0 = np.zeros((self.n_ctrl-2, 2))
        self.C0 = np.vstack([self.ri_start_sph, self.C_inner_0, self.ri_end_sph])

    # -------------------------------
    # B-spline fitting (Spherical)
    # -------------------------------
    def fit_spherical_spline(self):
        """
        Fit a B-spline to azimuth/elevation (spherical) over the RI segment.
        - Fits 2D control points (az, el)
        - Clamped at endpoints (first/last control pts fixed to RI start/end)
        - Optimizes interior control points and interior knot locations (via du increments)
        """

        # -------------------
        # Prepare data (azimuth, elevation)
        # -------------------
        self.S_sph = np.vstack([self.az_RI, self.el_RI]).T


        
        _, Nmat_sph, _ = self.evaluate_bspline(self.C0, self.p, self.U_sph, self.u_vals, return_basis=True)

        rhs = self.S_sph - (Nmat_sph[:, [0, -1]] @ np.vstack([self.ri_start_sph, self.ri_end_sph]))
        self.C_inner_0, _, _, _ = np.linalg.lstsq(Nmat_sph[:, 1:-1], rhs, rcond=None)

        self.C0 = np.vstack([self.ri_start_sph, self.C_inner_0, self.ri_end_sph])
        self.C_0 = self.C0.ravel()

        # Interior knots as evenly spaced increments
        du0 = np.ones(self.n_interior_knots) * 0.3

        x0 = np.concatenate([self.C_0[2:-2], du0])

        # -------------------
        # Bounds
        # -------------------
        lb_C = np.full_like(self.C_0[2:-2], -3 * np.pi)
        ub_C = np.full_like(self.C_0[2:-2],  3 * np.pi)

        lb_du = np.full_like(du0, self.eps_knot)
        ub_du = np.full_like(du0, 1 - self.eps_knot)

        lb = np.concatenate([lb_C, lb_du])
        ub = np.concatenate([ub_C, ub_du])

        # -------------------
        # Residual function
        # -------------------
        def residuals(params):
            # Reconstruct control points
            C = params[:(self.n_ctrl-2)*2].reshape(self.n_ctrl-2,2)
            C = np.vstack([self.ri_start_sph, C, self.ri_end_sph])

            # Reconstruct knot vector
            du = params[(self.n_ctrl-2)*2:]
            U_interior = np.cumsum(du)
            U_interior = U_interior / (U_interior[-1]+0.1)
            U = np.concatenate(([0]*(self.p+1), U_interior, [1]*(self.p+1)))

            # Evaluate spline
            S_fit_sph, Nmat, _ = self.evaluate_bspline(C, self.p, U, self.u_vals, return_basis=True)

            # Data residual
            res_data = np.array((S_fit_sph - self.S_sph).ravel())

            # Course penalty placeholder
            average_course_spline_start = self.compute_course_average_sph(C, U, self.u_vals, True)
            average_course_spline_end = self.compute_course_average_sph(C, U, self.u_vals, False)
            res_course = np.array([self.course_RI[0] - average_course_spline_start,
                           self.course_RI[-1] - average_course_spline_end]) * self.course_penalty

            return np.concatenate([res_data, res_course])

        # -------------------
        # Solve least squares
        # -------------------
        res = least_squares(residuals, x0, bounds=(lb, ub),
                            ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2)

        # -------------------
        # Extract optimized control points and knots
        # -------------------
        C_opt_sph = res.x[:(self.n_ctrl-2)*2].reshape(self.n_ctrl-2,2)
        C_opt_sph = np.vstack([self.ri_start_sph, C_opt_sph, self.ri_end_sph])

        du_opt = res.x[(self.n_ctrl-2)*2:]
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt = U_interior_opt / (U_interior_opt[-1]+0.1)
        U_opt_sph = np.concatenate(([0]*(self.p+1), U_interior_opt, [1]*(self.p+1)))

        # Save
        self.C_sph = C_opt_sph
        self.U_sph = U_opt_sph

        return self.C_sph, self.u_vals, self.U_sph

    # -------------------------------
    # B-spline fitting (Cartesian)
    # -------------------------------
    def fit_cartesian_spline(self):
        """
        Fit a B-spline to the Reel-In segment, optimizing control points
        and interior knots (via du increments), while keeping start/end
        points fixed and optionally penalizing start/end velocities.
        """
        # -------------------
        # Prepare data
        # -------------------
        self.S_cart = np.vstack([self.x_RI, self.y_RI, self.z_RI]).T
        dist = np.cumsum(np.linalg.norm(np.diff(self.S_cart, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        self.u_vals = dist / dist[-1]

        # -------------------
        # Number of interior knots (clamped knots)
        # -------------------
        number_of_knots = self.n_ctrl + self.p + 1
        n_interior_knots = (number_of_knots - 2*(self.p+1))

        if n_interior_knots <= 0:
            raise ValueError("Too few control points for spline order")

        # -------------------
        # Initial guess
        # -------------------
        # LSQ ignoring velocities to get initial control points
        U_interior_0 = np.linspace(0.15, 0.85, n_interior_knots + 2)[1:-1]
        U0 = np.concatenate(([0]*(self.p+1), U_interior_0, [1]*(self.p+1)))

        C_inner_0 = np.zeros((self.n_ctrl-2, 3))
        C0 = np.vstack([self.ri_start_point, C_inner_0, self.ri_end_point])
        _, Nmat, _ = self.evaluate_bspline(C0, self.p, U0, self.u_vals, return_basis=True)

        # mask = (self.u_vals > 0.02) & (self.u_vals < 0.98) # Avoid endpoints

        # C_inner_0, _, _, _ = np.linalg.lstsq(
        #     Nmat[mask, 1:-1],      # exclude clamped basis fns at ends
        #     self.S_cart[mask],     # exclude endpoints of trajectory
        #     rcond=None
        # )

        rhs = self.S_cart - (Nmat[:, [0, -1]] @ np.vstack([self.ri_start_point,
                                                   self.ri_end_point]))

        C_inner_0, _, _, _ = np.linalg.lstsq(Nmat[:, 1:-1], rhs, rcond=None)


        C0 = np.vstack([self.ri_start_point, C_inner_0, self.ri_end_point])
        C_0 = C0.ravel()

        # Interior knots as evenly spaced increments
        du0 = np.ones(n_interior_knots) * 0.3

        x0 = np.concatenate([C_0[3:-3], du0])
        # x0 = C_0[3:-3]  # Fix knots for now

        # -------------------
        # Bounds
        # -------------------
        lb_C = np.full_like(C_0[3:-3], -1e4)
        ub_C = np.full_like(C_0[3:-3], 1e4)

        lb_du = np.full_like(du0, self.eps_knot)
        ub_du = np.full_like(du0, 1 - self.eps_knot)
        lb = np.concatenate([lb_C, lb_du])
        ub = np.concatenate([ub_C, ub_du])

        # -------------------
        # Residual function
        # -------------------
        def residuals(params):
            # Reconstruct control points
            C = params[:(self.n_ctrl-2)*3].reshape(self.n_ctrl-2,3)
            C = np.vstack([self.ri_start_point, C, self.ri_end_point])

            # # Reconstruct knot vector
            du = params[(self.n_ctrl-2)*3:]
            U_interior = np.cumsum(du)
            U_interior = U_interior / (U_interior[-1]+0.1)  # Normalize to [0,1]
            U = np.concatenate(([0]*(self.p+1), U_interior, [1]*(self.p+1)))

            # Evaluate spline and derivative matrices
            S_fit_cart, Nmat, _ = self.evaluate_bspline(C, self.p, U, self.u_vals, return_basis=True)
            _, _, dNmat0, _ = self.evaluate_bspline(C, self.p, U, np.array([0.0]), return_basis=True, return_derivative=True)
            _, _, dNmat1, _ = self.evaluate_bspline(C, self.p, U, np.array([1.0]), return_basis=True, return_derivative=True)

            # Data residual
            res_data = (S_fit_cart - self.S_cart).ravel()

            # Velocity residual (start/end)
            S0_vel = dNmat0[0,:] @ C
            S1_vel = dNmat1[0,:] @ C
            res_vel = self.vel_penalty * np.concatenate([S0_vel - self.ri_start_velocity,
                                                S1_vel - self.ri_end_velocity])
            return np.concatenate([res_data, res_vel])

        # -------------------
        # Solve least squares
        # -------------------
        res = least_squares(residuals, x0, bounds=(lb, ub), ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2)

        # -------------------
        # Extract optimized control points and knots
        # -------------------
        C_opt = res.x[:(self.n_ctrl-2)*3].reshape(self.n_ctrl-2,3)
        C_opt = np.vstack([self.ri_start_point, C_opt, self.ri_end_point])

        du_opt = res.x[(self.n_ctrl-2)*3:]
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt = U_interior_opt / (U_interior_opt[-1]+0.1)
        U_opt = np.concatenate(([0]*(self.p+1), U_interior_opt, [1]*(self.p+1)))

        # Save
        self.C_cart = C_opt
        self.U_cart = U_opt

        return self.C_cart, self.u_vals, self.U_cart

    # -------------------------------
    # Spline evaluation
    # -------------------------------
    def eval_cartesian_spline(self, u):
        result = self.evaluate_bspline(self.C_cart, self.p, self.U_cart, u)
        return result

    def eval_spherical_spline(self, u):
        xyz = self.eval_cartesian_spline(u)
        if xyz.ndim == 1:
            return self.cart2sph(*xyz)
        else:
            return np.array([self.cart2sph(*pt) for pt in xyz])



if __name__ == "__main__":
# --- File paths ---
    pass

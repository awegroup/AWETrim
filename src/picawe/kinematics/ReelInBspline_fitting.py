import numpy as np
from scipy.optimize import least_squares
from .ReelInBspline_data_processing import ReelInBspline_data_processing as ribdata
from .ReelInBspline_build import ReelInBspline_build as ribbuild

class ReelInBspline_fitting(ribdata, ribbuild):

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

    """
    B-spline fitting class for Reel-In trajectories.
    Inherits all instance variables from ReelInBspline_data_processing.
    """

    def __init__(self, file_path_full, file_path_cycle, cyc_idx, p, n_ctrl,
                 c_penalty=1.0, v_penalty=0.0, eps_knot=1e-3):
        # Initialize parent class to load data and compute variables
        super().__init__(file_path_full, file_path_cycle, cyc_idx)

        # B-spline parameters
        self.p = p
        self.n_ctrl = n_ctrl
        self.c_penalty = c_penalty
        self.v_penalty = v_penalty
        self.eps_knot = eps_knot

    def u_vals(self):
        # Spherical and Cartesian trajectory matrices
        self.S_sph = np.vstack([self.az_ri, self.el_ri]).T
        self.S_cart = np.vstack([self.x_ri, self.y_ri, self.z_ri]).T

        dist = np.cumsum(np.linalg.norm(np.diff(self.S_cart, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        self.u_vals = dist / dist[-1]

    def compute_course_average_sph(self, C, U, u, start=True):
        """Compute average course over first/last k points of spline."""
        k = 10
        S_sph, dS_sph = self.evaluate_bspline(C, p=3, U=U, u=u, return_derivative=True)

        if start:
            course = -np.arctan2(dS_sph[:k, 0] * np.cos(S_sph[:k, 1]), dS_sph[:k, 1]) + 2 * np.pi
        else:
            course = -np.arctan2(dS_sph[-k:, 0] * np.cos(S_sph[-k:, 1]), dS_sph[-k:, 1]) + 2 * np.pi

        course = np.mod(course, 2 * np.pi)
        self.course_avg = np.mean(course)
        return self.course_avg

    def compute_velocity_average_cart(self, C, U, u, start=True):
        """Compute average velocity over first/last k points of spline."""
        k = 10
        S_cart, dS_cart = self.evaluate_bspline(C, p=3, U=U, u=u, return_derivative=True)

        if start:
            velocity = dS_cart[:k, :]
        else:
            velocity = dS_cart[-k:, :]

        self.velocity_avg = np.mean(velocity, axis=0)
        return self.velocity_avg

    def fit_spline(self, mode="spherical"):
        """
        Fit a B-spline to either spherical (az/el) or Cartesian coordinates.
        """
        # Ensure u_vals exist
        if not hasattr(self, 'u_vals'):
            self.u_vals()

        # -------------------------------
        # Setup initial solution
        # -------------------------------
        self.number_of_knots = self.n_ctrl + self.p + 1
        self.n_interior_knots = self.number_of_knots - 2 * (self.p + 1)
        if self.n_interior_knots <= 0:
            raise ValueError("Too few control points for spline order")

        U_interior_0 = np.linspace(0.15, 0.85, self.n_interior_knots + 2)[1:-1]
        self.U0 = np.concatenate(([0] * (self.p + 1), U_interior_0, [1] * (self.p + 1)))

        if mode == "spherical":
            self.S_data = self.S_sph
            ri_p0 = self.ri_p0_sph
            ri_pf = self.ri_pf_sph
            dim = 2
        else:
            self.S_data = self.S_cart
            ri_p0 = self.ri_p0_cart
            ri_pf = self.ri_pf_cart
            dim = 3

        # Initial control points
        C_inner_0 = np.zeros((self.n_ctrl - 2, dim))
        C0 = np.vstack([ri_p0, C_inner_0, ri_pf])

        # Initial least-squares refinement
        _, Nmat, _, _ = self.evaluate_bspline(C0, self.p, self.U0, self.u_vals, return_basis=True)
        rhs = self.S_data - (Nmat[:, [0, -1]] @ np.vstack([ri_p0, ri_pf]))
        C_inner_0, _, _, _ = np.linalg.lstsq(Nmat[:, 1:-1], rhs, rcond=None)
        C0 = np.vstack([ri_p0, C_inner_0, ri_pf]).ravel()

        # Knot increments
        du0 = np.ones(self.n_interior_knots) * 0.3
        self.x0 = np.concatenate([C0[dim:-dim], du0])

        # Bounds
        lb_C = np.full_like(C0[dim:-dim], -3 * np.pi)
        ub_C = np.full_like(C0[dim:-dim], 3 * np.pi)
        lb_du = np.full_like(du0, self.eps_knot)
        ub_du = np.full_like(du0, 1 - self.eps_knot)
        self.lb = np.concatenate([lb_C, lb_du])
        self.ub = np.concatenate([ub_C, ub_du])

        # -------------------------------
        # Residual function
        # -------------------------------
        def residuals(params):
            C = params[:(self.n_ctrl - 2) * dim].reshape(self.n_ctrl - 2, dim)
            C = np.vstack([ri_p0, C, ri_pf])

            du = params[(self.n_ctrl - 2) * dim:]
            U_interior = np.cumsum(du)
            U_interior /= (U_interior[-1] + 0.1)
            U = np.concatenate(([0] * (self.p + 1), U_interior, [1] * (self.p + 1)))

            S_fit, Nmat, _, _ = self.evaluate_bspline(C, self.p, U, self.u_vals, return_basis=True)
            res_data = (S_fit - self.S_data).ravel()

            if mode == "spherical":
                # course penalties
                avg_start = self.compute_course_average_sph(C, U, self.u_vals, True)
                avg_end = self.compute_course_average_sph(C, U, self.u_vals, False)
                res_extra = np.array([self.ri_crs0 - avg_start, self.ri_crsf - avg_end]) * self.c_penalty
            else:
                # velocity penalties
                vel_start = self.compute_velocity_average_cart(C, U, self.u_vals, True)
                vel_end = self.compute_velocity_average_cart(C, U, self.u_vals, False)
                res_extra = np.ravel(np.array([vel_start - self.ri_v0, vel_end - self.ri_vf])) * self.v_penalty

            return np.concatenate([res_data, res_extra])

        # -------------------------------
        # Solve least squares
        # -------------------------------
        res = least_squares(residuals, self.x0, bounds=(self.lb, self.ub),
                            ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2)

        # -------------------------------
        # Save optimized results
        # -------------------------------
        C_opt = res.x[:(self.n_ctrl - 2) * dim].reshape(self.n_ctrl - 2, dim)
        C_opt = np.vstack([ri_p0, C_opt, ri_pf])

        du_opt = res.x[(self.n_ctrl - 2) * dim:]
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt /= (U_interior_opt[-1] + 0.1)
        U_opt = np.concatenate(([0] * (self.p + 1), U_interior_opt, [1] * (self.p + 1)))

        if mode == "spherical":
            self.C_sph, self.U_sph = C_opt, U_opt
            return self.C_sph, self.u_vals, self.U_sph
        else:
            self.C_cart, self.U_cart = C_opt, U_opt
            return self.C_cart, self.u_vals, self.U_cart
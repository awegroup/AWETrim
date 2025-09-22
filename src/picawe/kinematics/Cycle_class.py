import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# -------------------------------
# Helper Functions
# -------------------------------
def convert_time_to_seconds(time_array):
    """Convert HH:MM:SS.sss to seconds (float)."""
    seconds_array = []
    for time_str in time_array:
        h, m, s = map(float, str(time_str).split(":"))
        seconds_array.append(h * 3600 + m * 60 + s)
    return np.array(seconds_array)

def sph2cart(az, el, r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arcsin(z / r)
    return az, el, r

def evaluate_bspline(C, p, U, u, return_basis=False, return_derivative=False):
    """
    Evaluate B-spline at u (scalar or array).
    If return_basis=True, returns basis matrix Nmat.
    If return_derivative=True, also returns derivative basis matrix dNmat.
    """
    u = np.atleast_1d(u)
    n_ctrl = C.shape[0]
    Nmat = np.zeros((len(u), n_ctrl))
    dNmat = np.zeros((len(u), n_ctrl))

    def N(i, k, u_val):
        if k == 0:
            # Special case: include right endpoint for last knot span
            if (U[i] <= u_val < U[i+1]) or (
                u_val == U[-1] and U[i] <= u_val <= U[i+1]
            ):
                return 1.0
            else:
                return 0.0

        left = 0.0
        right = 0.0

        if U[i+k] > U[i]:
            left = (u_val - U[i])/(U[i+k]-U[i]) * N(i, k-1, u_val)
        if U[i+k+1] > U[i+1]:
            right = (U[i+k+1]-u_val)/(U[i+k+1]-U[i+1]) * N(i+1, k-1, u_val)
        return left + right

    def dN(i, k, u_val):
        if k == 0:
            return 0.0
        left = 0.0
        right = 0.0
        if U[i+k] > U[i]:
            left = k/(U[i+k]-U[i]) * N(i, k-1, u_val)
        if U[i+k+1] > U[i+1]:
            right = k/(U[i+k+1]-U[i+1]) * N(i+1, k-1, u_val)
        return left - right

    for ui, u_val in enumerate(u):
        for i in range(n_ctrl):
            Nmat[ui, i] = N(i, p, u_val)
            if return_derivative:
                dNmat[ui, i] = dN(i, p, u_val)

    S = Nmat @ C
    dS = dNmat @ C if return_derivative else None

    if return_basis and return_derivative:
        return S, Nmat, dNmat, dS
    elif return_basis:
        return S, Nmat, dNmat
    elif return_derivative:
        return S, dS
    else:
        return S

# -------------------------------
# Cycle Class
# -------------------------------
class Cycle:
    def __init__(self, file_path_full, file_path_cycle, cycle_idx=0):
        """Initialize Cycle object from CSV files and cycle index."""
        # Load CSVs
        self.full_df = pd.read_csv(file_path_full)
        self.cycle_df = pd.read_csv(file_path_cycle)
        self.cycle_idx = cycle_idx

        # Preprocess times
        self.full_df['time_s'] = np.round(convert_time_to_seconds(self.full_df['time_of_day'].to_numpy()), 1)
        self.cycle_df['start_time_s'] = np.round(convert_time_to_seconds(self.cycle_df['start_time_cycle_LT'].to_numpy()), 1)

        # Extract variables
        self.time_full = self.full_df['time_s'].to_numpy()
        self.time_cycle = self.cycle_df['start_time_s'].to_numpy()
        self.az = self.full_df['kite_azimuth'].to_numpy()
        self.el = self.full_df['kite_elevation'].to_numpy()
        self.r = self.full_df['kite_distance'].to_numpy()
        self.phase = self.full_df['flight_phase'].to_numpy()
        self.course = self.full_df['kite_course'].to_numpy()

        # Compute cycle boundaries
        self._compute_cycle_indices()
        self._extract_cycle_data()

        # B-spline variables
        self.C_cart, self.p, self.U_cart = None, None, None
        self.C_sph, self.U_sph = None, None
        self.u_vals = None

        # Reel-In points and velocities
        self.ri_start_point, self.ri_end_point = None, None
        self.ri_start_velocity, self.ri_end_velocity = None, None
        self.az_RI, self.el_RI, self.r_RI = None, None, None
        self.x_RI, self.y_RI, self.z_RI = None, None, None
        self.ri_start_course, self.ri_end_course = None, None
        self.RI_start_idx, self.RI_end_idx = None, None

    # -------------------------------
    # Internal methods
    # -------------------------------
    def _compute_cycle_indices(self):
        """Find start/end indices of the selected cycle."""
        self.start_indices = np.array([
            i for i, t in enumerate(self.time_full) 
            for tc in self.time_cycle if t == tc
        ])
        if self.cycle_idx >= len(self.start_indices)-1:
            raise IndexError("cycle_idx out of range")
        self.cycle_start_idx = self.start_indices[self.cycle_idx]
        self.cycle_end_idx   = self.start_indices[self.cycle_idx + 1] - 1 if self.cycle_idx + 1 < len(self.start_indices) else len(self.time_full)-1

    def _extract_cycle_data(self):
        """Extract only the selected cycle data (spherical and cartesian)."""
        self.az_cyc = self.az[self.cycle_start_idx:self.cycle_end_idx+1]
        self.el_cyc = self.el[self.cycle_start_idx:self.cycle_end_idx+1]
        self.r_cyc  = self.r[self.cycle_start_idx:self.cycle_end_idx+1]
        self.phase_cyc = self.phase[self.cycle_start_idx:self.cycle_end_idx+1]
        self.course_cyc = self.course[self.cycle_start_idx:self.cycle_end_idx+1]

        self.x_cyc, self.y_cyc, self.z_cyc = sph2cart(self.az_cyc, self.el_cyc, self.r_cyc)
        self.dx_cyc, self.dy_cyc, self.dz_cyc = np.gradient(self.x_cyc), np.gradient(self.y_cyc), np.gradient(self.z_cyc)
        self.num_points = len(self.x_cyc)

    def investigate_course_computations(self):
        """Compare different course computations over the Reel-In segment in one plot."""

        # -----------------------------
        # Cartesian course from dx, dy
        # -----------------------------
        course_cart = np.arctan2(
            self.dy_cyc[self.RI_start_idx:self.RI_end_idx],
            self.dx_cyc[self.RI_start_idx:self.RI_end_idx]
        )
        course_cart = np.mod(course_cart, 2 * np.pi)

        # -----------------------------
        # Spherical course from az/el rates
        # -----------------------------
        az_dot = np.gradient(self.az_RI)
        el_dot = np.gradient(self.el_RI)

        course_sph = -np.arctan2(az_dot * np.cos(self.el_RI), el_dot) + 2*np.pi
        course_sph = np.mod(course_sph, 2 * np.pi)

        # -----------------------------
        # Reference course (CSV)
        # -----------------------------
        course_ref = np.mod(self.course_RI, 2 * np.pi)

        # -----------------------------
        # Plotting all in one plot
        # -----------------------------
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(course_cart, label="Cartesian course", color="C0")
        ax.plot(course_sph, label="Spherical course", color="C1")
        ax.plot(course_ref, label="Reference (CSV)", color="C2")

        ax.set_ylabel("Course [rad]")
        ax.set_xlabel("Sample index")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("Comparison of Course Computations over Reel-In Segment")

        plt.show()

    def compute_course_average_sph(self, C, U, u, start):
        """Compute average course over first or last k points of spline."""
        k = 2  # Number of points to average over

        if not start:
            S_sph, dS_sph = evaluate_bspline(C, p=3, U=U, u=u, return_derivative=True)
            course = -np.arctan2(dS_sph[-k:,0]*np.cos(S_sph[-k:,1]), dS_sph[-k:,1]) + 2*np.pi
            course = np.mod(course, 2 * np.pi)
            self.course_avg = np.mean(course)
            return self.course_avg
        if start:
            S_sph, dS_sph = evaluate_bspline(C, p=3, U=U, u=u, return_derivative=True)
            course = -np.arctan2(dS_sph[:k,0]*np.cos(S_sph[:k,1]), dS_sph[:k,1]) + 2*np.pi
            course = np.mod(course, 2 * np.pi)
            self.course_avg = np.mean(course)
            return self.course_avg

    # -------------------------------
    # Reel-In/Out boundaries
    # -------------------------------
    def get_RI_RO_boundaries(self):
        """Compute start/end points and velocities of Reel-In."""
        # Find start of Reel-In
        RI_start_idx = None
        for i, tag in enumerate(self.phase_cyc):
            if tag.lower() in ["pp-ri", "pp-rori", "pp-riro"]:
                RI_start_idx = i
                break
        if RI_start_idx is None:
            raise ValueError("Reel-In start not found in this cycle")

        RI_end_idx = len(self.phase_cyc) - 1  # Last point of cycle

        self.RI_start_idx, self.RI_end_idx = RI_start_idx, RI_end_idx

        # Store points and velocities
        self.ri_start_point_sph = np.array([self.az_cyc[RI_start_idx], self.el_cyc[RI_start_idx]])
        self.ri_end_point_sph = np.array([self.az_cyc[RI_end_idx], self.el_cyc[RI_end_idx]])
        self.ri_start_point = np.array([self.x_cyc[RI_start_idx], self.y_cyc[RI_start_idx], self.z_cyc[RI_start_idx]])
        self.ri_end_point   = np.array([self.x_cyc[RI_end_idx], self.y_cyc[RI_end_idx], self.z_cyc[RI_end_idx]])
        self.ri_start_velocity = np.array([self.dx_cyc[RI_start_idx], self.dy_cyc[RI_start_idx], self.dz_cyc[RI_start_idx]])
        self.ri_end_velocity   = np.array([self.dx_cyc[RI_end_idx], self.dy_cyc[RI_end_idx], self.dz_cyc[RI_end_idx]])
        self.ri_start_course = self.course_cyc[RI_start_idx]
        self.ri_end_course   = self.course_cyc[RI_end_idx]

        # Spherical RI segment
        self.az_RI = self.az_cyc[RI_start_idx:RI_end_idx+1]
        self.el_RI = self.el_cyc[RI_start_idx:RI_end_idx+1]
        self.r_RI  = self.r_cyc[RI_start_idx:RI_end_idx+1]
        self.course_RI = self.course_cyc[RI_start_idx:RI_end_idx+1]

        # Cartesian RI segment
        self.x_RI, self.y_RI, self.z_RI = self.x_cyc[RI_start_idx:RI_end_idx+1], self.y_cyc[RI_start_idx:RI_end_idx+1], self.z_cyc[RI_start_idx:RI_end_idx+1]

        return (self.ri_start_point, self.ri_end_point,
                self.ri_start_velocity, self.ri_end_velocity,
                self.az_RI, self.el_RI, self.r_RI,
                self.RI_start_idx, self.RI_end_idx)

    # -------------------------------
    # B-spline fitting (Spherical)
    # -------------------------------
    def fit_spherical_spline(self, p=3, n_ctrl=8, course_penalty=1.0, eps_knot=1e-3):
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

        S_cart_for_param = np.vstack([self.x_RI, self.y_RI, self.z_RI]).T
        dist = np.cumsum(np.linalg.norm(np.diff(S_cart_for_param, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        self.u_vals = dist / dist[-1]

        # -------------------
        # Number of interior knots (clamped knots)
        # -------------------
        number_of_knots = n_ctrl + p + 1
        n_interior_knots = number_of_knots - 2*(p+1)

        if n_interior_knots <= 0:
            raise ValueError("Too few control points for spline order")

        # -------------------
        # Initial guess
        # -------------------
        U_interior_0 = np.linspace(0.15, 0.85, n_interior_knots + 2)[1:-1]
        U0 = np.concatenate(([0]*(p+1), U_interior_0, [1]*(p+1)))

        ri_start_sph = np.array([self.az_RI[0], self.el_RI[0]])
        ri_end_sph   = np.array([self.az_RI[-1], self.el_RI[-1]])

        C_inner_0 = np.zeros((n_ctrl-2, 2))
        C0 = np.vstack([ri_start_sph, C_inner_0, ri_end_sph])
        _, Nmat_sph, _ = evaluate_bspline(C0, p, U0, self.u_vals, return_basis=True)

        rhs = self.S_sph - (Nmat_sph[:, [0, -1]] @ np.vstack([ri_start_sph, ri_end_sph]))
        C_inner_0, _, _, _ = np.linalg.lstsq(Nmat_sph[:, 1:-1], rhs, rcond=None)

        C0 = np.vstack([ri_start_sph, C_inner_0, ri_end_sph])
        C_0 = C0.ravel()

        # Interior knots as evenly spaced increments
        du0 = np.ones(n_interior_knots) * 0.3

        x0 = np.concatenate([C_0[2:-2], du0])

        # -------------------
        # Bounds
        # -------------------
        lb_C = np.full_like(C_0[2:-2], -3 * np.pi)
        ub_C = np.full_like(C_0[2:-2],  3 * np.pi)

        lb_du = np.full_like(du0, eps_knot)
        ub_du = np.full_like(du0, 1-eps_knot)

        lb = np.concatenate([lb_C, lb_du])
        ub = np.concatenate([ub_C, ub_du])

        # -------------------
        # Residual function
        # -------------------
        def residuals(params):
            # Reconstruct control points
            C = params[:(n_ctrl-2)*2].reshape(n_ctrl-2,2)
            C = np.vstack([ri_start_sph, C, ri_end_sph])

            # Reconstruct knot vector
            du = params[(n_ctrl-2)*2:]
            U_interior = np.cumsum(du)
            U_interior = U_interior / (U_interior[-1]+0.1)
            U = np.concatenate(([0]*(p+1), U_interior, [1]*(p+1)))

            # Evaluate spline
            S_fit_sph, Nmat, _ = evaluate_bspline(C, p, U, self.u_vals, return_basis=True)

            # Data residual
            res_data = np.array((S_fit_sph - self.S_sph).ravel())

            # Course penalty placeholder
            average_course_spline_start = self.compute_course_average_sph(C, U, self.u_vals, True)
            average_course_spline_end = self.compute_course_average_sph(C, U, self.u_vals, False)
            res_course = np.array([self.course_RI[0] - average_course_spline_start,
                           self.course_RI[-1] - average_course_spline_end]) * course_penalty

            return np.concatenate([res_data, res_course])

        # -------------------
        # Solve least squares
        # -------------------
        res = least_squares(residuals, x0, bounds=(lb, ub),
                            ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2)

        # -------------------
        # Extract optimized control points and knots
        # -------------------
        C_opt_sph = res.x[:(n_ctrl-2)*2].reshape(n_ctrl-2,2)
        C_opt_sph = np.vstack([ri_start_sph, C_opt_sph, ri_end_sph])

        du_opt = res.x[(n_ctrl-2)*2:]
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt = U_interior_opt / (U_interior_opt[-1]+0.1)
        U_opt_sph = np.concatenate(([0]*(p+1), U_interior_opt, [1]*(p+1)))

        # Save
        self.C_sph = C_opt_sph
        self.U_sph = U_opt_sph
        self.p = p

        return self.C_sph, self.u_vals, self.U_sph

    # -------------------------------
    # B-spline fitting (Cartesian)
    # -------------------------------
    def fit_cartesian_spline(self, p=3, n_ctrl=8, vel_penalty=0.0, eps_knot=1e-3):
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
        number_of_knots = n_ctrl + p + 1
        n_interior_knots = (number_of_knots - 2*(p+1))

        if n_interior_knots <= 0:
            raise ValueError("Too few control points for spline order")

        # -------------------
        # Initial guess
        # -------------------
        # LSQ ignoring velocities to get initial control points
        U_interior_0 = np.linspace(0.15, 0.85, n_interior_knots + 2)[1:-1]
        U0 = np.concatenate(([0]*(p+1), U_interior_0, [1]*(p+1)))

        C_inner_0 = np.zeros((n_ctrl-2, 3))
        C0 = np.vstack([self.ri_start_point, C_inner_0, self.ri_end_point])
        _, Nmat, _ = evaluate_bspline(C0, p, U0, self.u_vals, return_basis=True)

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

        lb_du = np.full_like(du0, eps_knot)
        ub_du = np.full_like(du0, 1-eps_knot)
        lb = np.concatenate([lb_C, lb_du])
        ub = np.concatenate([ub_C, ub_du])

        # -------------------
        # Residual function
        # -------------------
        def residuals(params):
            # Reconstruct control points
            C = params[:(n_ctrl-2)*3].reshape(n_ctrl-2,3)
            C = np.vstack([self.ri_start_point, C, self.ri_end_point])

            # # Reconstruct knot vector
            du = params[(n_ctrl-2)*3:]
            U_interior = np.cumsum(du)
            U_interior = U_interior / (U_interior[-1]+0.1)  # Normalize to [0,1]
            U = np.concatenate(([0]*(p+1), U_interior, [1]*(p+1)))

            # Evaluate spline and derivative matrices
            S_fit_cart, Nmat, _ = evaluate_bspline(C, p, U, self.u_vals, return_basis=True)
            _, _, dNmat0, _ = evaluate_bspline(C, p, U, np.array([0.0]), return_basis=True, return_derivative=True)
            _, _, dNmat1, _ = evaluate_bspline(C, p, U, np.array([1.0]), return_basis=True, return_derivative=True)

            # Data residual
            res_data = (S_fit_cart - self.S_cart).ravel()

            # Velocity residual (start/end)
            S0_vel = dNmat0[0,:] @ C
            S1_vel = dNmat1[0,:] @ C
            res_vel = vel_penalty * np.concatenate([S0_vel - self.ri_start_velocity,
                                                S1_vel - self.ri_end_velocity])
            return np.concatenate([res_data, res_vel])

        # -------------------
        # Solve least squares
        # -------------------
        res = least_squares(residuals, x0, bounds=(lb, ub), ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2)

        # -------------------
        # Extract optimized control points and knots
        # -------------------
        C_opt = res.x[:(n_ctrl-2)*3].reshape(n_ctrl-2,3)
        C_opt = np.vstack([self.ri_start_point, C_opt, self.ri_end_point])

        du_opt = res.x[(n_ctrl-2)*3:]
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt = U_interior_opt / (U_interior_opt[-1]+0.1)
        U_opt = np.concatenate(([0]*(p+1), U_interior_opt, [1]*(p+1)))

        # Save
        self.C_cart = C_opt
        self.U_cart = U_opt
        self.p = p

        return self.C_cart, self.u_vals, self.U_cart

    # -------------------------------
    # Spline evaluation
    # -------------------------------
    def eval_cartesian_spline(self, u):
        result = evaluate_bspline(self.C_cart, self.p, self.U_cart, u)
        return result

    def eval_spherical_spline(self, u):
        xyz = self.eval_cartesian_spline(u)
        if xyz.ndim == 1:
            return cart2sph(*xyz)
        else:
            return np.array([cart2sph(*pt) for pt in xyz])

    # -------------------------------
    # Plotting
    # -------------------------------
    def plot_spline_fit_cart(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.x_cyc, self.y_cyc, self.z_cyc, label="Trajectory", alpha=0.6)
        S_fit_cart = np.vstack([self.eval_cartesian_spline(u) for u in self.u_vals])
        ax.plot(S_fit_cart[:,0], S_fit_cart[:,1], S_fit_cart[:,2], "r--", label="B-spline fit")
        
        # Plot control points
        if self.C_cart is not None:
            ax.scatter(self.C_cart[:,0], self.C_cart[:,1], self.C_cart[:,2],
                    color="black", s=30, label="Control points")
        
        # Plot RI start/end
        if self.ri_start_point is not None and self.ri_end_point is not None:
            ax.scatter(*self.ri_start_point, color="green", s=30, label="RI Start")
            ax.scatter(*self.ri_end_point, color="red", s=30, label="RI End")
        
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(); ax.set_box_aspect([1,1,1])
        plt.show()

    def plot_spline_fit_sph(self):
        if self.C_sph is None or self.U_sph is None or self.u_vals is None:
            raise ValueError("Run fit_spherical_spline() before plotting spherical fit.")
        if self.az_RI is None or self.el_RI is None:
            raise ValueError("Run get_RI_RO_boundaries() before plotting spherical fit.")

        # Evaluate spherical spline
        S_fit_sph = np.vstack([
            evaluate_bspline(self.C_sph, self.p, self.U_sph, u) for u in self.u_vals
        ])

        # Create subplots (single fig only!)
        fig, (ax_az, ax_el) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

        # Azimuth plot
        ax_az.plot(self.u_vals, self.az_RI, label="Azimuth (data)", color="C0")
        ax_az.plot(self.u_vals, S_fit_sph[:, 0], "--", label="Azimuth (spline)", color="C1")
        
        # Add control points
        if self.C_sph is not None:
            ax_az.scatter(
                np.linspace(self.u_vals[0], self.u_vals[-1], len(self.C_sph)),
                self.C_sph[:, 0], color="black", s=30, label="Control points"
            )

        # Add RI endpoints
        if self.ri_start_point is not None and self.ri_end_point is not None:
            ax_az.scatter(self.u_vals[0], self.ri_start_point_sph[0], color="green", s=30, label="RI Start")
            ax_az.scatter(self.u_vals[-1], self.ri_end_point_sph[0], color="red", s=30, label="RI End")

        ax_az.set_xlabel("u")
        ax_az.set_ylabel("Azimuth [rad]")
        ax_az.grid(True, alpha=0.3)
        ax_az.legend()

        # Elevation plot
        ax_el.plot(self.u_vals, self.el_RI, label="Elevation (data)", color="C0")
        ax_el.plot(self.u_vals, S_fit_sph[:, 1], "--", label="Elevation (spline)", color="C1")
        
        # Add control points
        if self.C_sph is not None:
            ax_el.scatter(
                np.linspace(self.u_vals[0], self.u_vals[-1], len(self.C_sph)),
                self.C_sph[:, 1], color="black", s=30, label="Control points"
            )

        # Add RI endpoints
        if self.ri_start_point is not None and self.ri_end_point is not None:
            ax_el.scatter(self.u_vals[0], self.ri_start_point_sph[1], color="green", s=30, label="RI Start")
            ax_el.scatter(self.u_vals[-1], self.ri_end_point_sph[1], color="red", s=30, label="RI End")

        ax_el.set_xlabel("u")
        ax_el.set_ylabel("Elevation [rad]")
        ax_el.grid(True, alpha=0.3)
        ax_el.legend()

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
# --- File paths ---
    full_df = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
    cycle_df = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"

    # Create a Cycle object for the first cycle (cycle_idx=0)
    cycle = Cycle(full_df, cycle_df, cycle_idx=1)

    # Compute Reel-In boundaries
    ri_start, ri_end, ri_v0, ri_vf, az_RI, el_RI, r_RI, ri_start_idx, ri_end_idx = cycle.get_RI_RO_boundaries()

    # Fit a B-spline to the full cycle (or later to RI only)
    C_cart, u_vals, U = cycle.fit_cartesian_spline()
    C_sph, u_vals_sph, U_sph = cycle.fit_spherical_spline()

    # Plot trajectory and spline
    cycle.plot_spline_fit_cart()

    cycle.plot_spline_fit_sph()

    # Investigate course computations
    # cycle.investigate_course_computations()

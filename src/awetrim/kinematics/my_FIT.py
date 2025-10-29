import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import least_squares

from awetrim.kinematics.parametrized_patterns import CasadiSpline, CST_Lissajous
from awetrim.kinematics.my_DP import (
    DataProcessing,
)  # Your refactored DataProcessing class


class Fitting(DataProcessing):
    """
    Unified fitting class for kite path segments.
    Supports:
      - Spline fitting for RI, RI_RO, RO_RI
      - Lissajous fitting for RO Lissajous pattern
    """

    def __init__(
        self,
        file_path_full,
        file_path_cycle,
        file_path_waypoints,
        cyc_idx=0,
        n_ctrl_pts=30,
        run_plots=True,
    ):
        # Initialize DataProcessing parent class
        super().__init__(file_path_full, file_path_cycle, file_path_waypoints, cyc_idx, run_plots_DP=run_plots)
        self.n_ctrl = n_ctrl_pts
    
        self._setup_spline_segment()
        self._setup_lissajous_segment()

        self.FitSpline()
        self.FitLissajous()

        self.save_data_RI_spline()
        self.save_data_L_shape()

        if run_plots:
            self.plot_spline_cart()
            self.plot_fit_L_shape()

    # -------------------------------------------------------------------------
    # ------------------------ Lissajous Segment Setup ------------------------
    # -------------------------------------------------------------------------
    def _setup_lissajous_segment(self):
        """Prepare data for Lissajous fitting from RO Lissajous."""
        self.L_shape_r0 = self.L_shape_r[0]
        self.L_shape_u_vals = np.linspace(0, 2 * np.pi, len(self.L_shape_az))

    def FitLissajous(self):
        """Run least-squares Lissajous fitting."""
        print("Starting Lissajous fitting...")
        fixed_params = {
            "omega": 1,
            "r0": self.L_shape_r0,
            "kappa": 0.0,
            "kbeta": 0.0,
            "width_phi": 0.5,
            "width_beta": 0.5,
            "left_first": True,
            "normalize_bumps": False,
            "repeat_phi": True,
            "repeat_beta": True,
            "k_vr": 2716,
        }
        n_coeffs = 5
        params_init = {
            "az_amp0": 0.34,
            "beta_amp0": 0.08,
            "beta0": 0.48,
            "beta_coeffs": list(np.random.uniform(-1, 1, n_coeffs)),
            "az_coeffs": list(np.random.uniform(-1, 1, n_coeffs)),
        }
        x0 = np.concatenate(
            [
                [params_init["az_amp0"]],
                [params_init["beta_amp0"]],
                [params_init["beta0"]],
                params_init["beta_coeffs"],
                params_init["az_coeffs"],
            ]
        )
        lower_bounds = [0, 0, 0] + [-2] * n_coeffs + [-2] * n_coeffs
        upper_bounds = [2, 1, 1] + [2] * n_coeffs + [2] * n_coeffs

        def unpack_params(x):
            return {
                "az_amp0": x[0],
                "beta_amp0": x[1],
                "beta0": x[2],
                "beta_coeffs": x[3 : 3 + n_coeffs].tolist(),
                "az_coeffs": x[3 + n_coeffs :].tolist(),
                **fixed_params,
            }

        def residual(x):
            params = unpack_params(x)
            obj = CST_Lissajous(**params)
            az_model = obj.azimuth(params["r0"], self.L_shape_u_vals)
            el_model = obj.elevation(params["r0"], self.L_shape_u_vals)
            return np.concatenate(
                (self.L_shape_az - az_model, self.L_shape_el - el_model)
            ).ravel()

        res = least_squares(
            residual,
            x0,
            bounds=(lower_bounds, upper_bounds),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            verbose=0,
        )

        self.best_params = unpack_params(res.x)
        obj = CST_Lissajous(**self.best_params)
        self.L_shape_az_fit = obj.azimuth(self.best_params["r0"], self.L_shape_u_vals)
        self.L_shape_el_fit = obj.elevation(self.best_params["r0"], self.L_shape_u_vals)
        print("✅ Lissajous fitting completed.")

    # -------------------------------------------------------------------------
    # ------------------------- Spline Segment Setup --------------------------
    # -------------------------------------------------------------------------
    def _setup_spline_segment(self):
        """Prepare data for spline fitting based on segment."""

        self.data_az, self.data_el, self.u_vals, self.r0, self.r1, self.data_r = (
            self.RI_Spline_az,
            self.RI_Spline_el,
            self.RI_Spline_u_vals,
            self.RI_Spline_r0,
            self.RI_Spline_r1,
            self.RI_Spline_r,
        )

        # Initial control points
        self.indices0 = np.linspace(0, len(self.data_az) - 1, self.n_ctrl, dtype=int)[
            1:-1
        ]
        self.initial_params_az = self.data_az[self.indices0]
        self.initial_params_el = self.data_el[self.indices0]
        self.init_params = np.concatenate(
            [
                self.initial_params_az,
                self.initial_params_el,
                self.indices0,
                self.indices0,
            ]
        )

        # Parameter bounds
        bounds_az = (
            np.full(self.n_ctrl - 2, -np.pi / 2),
            np.full(self.n_ctrl - 2, np.pi / 2),
        )
        bounds_el = (
            np.full(self.n_ctrl - 2, -0.01),
            np.full(self.n_ctrl - 2, np.pi / 2 + 0.01),
        )
        bounds_idx = (
            np.full(self.n_ctrl - 2, 2),
            np.full(self.n_ctrl - 2, len(self.data_az) - 3),
        )
        self.bounds = (
            np.concatenate([bounds_az[0], bounds_el[0], bounds_idx[0], bounds_idx[0]]),
            np.concatenate([bounds_az[1], bounds_el[1], bounds_idx[1], bounds_idx[1]]),
        )

    def residuals(self, params):
        """Residuals function for spline fitting."""
        n = self.n_ctrl - 2
        params_az = params[:n]
        params_el = params[n : 2 * n]
        indices_az = np.concatenate(
            ([0], params[2 * n : 3 * n].astype(int), [len(self.data_az) - 1])
        )
        indices_el = np.concatenate(
            ([0], params[3 * n :].astype(int), [len(self.data_az) - 1])
        )

        if not np.all(np.diff(self.u_vals[indices_az]) >= 0) or not np.all(
            np.diff(self.u_vals[indices_el]) >= 0
        ):
            return np.full(2 * len(self.data_az), 1e7)

        try:
            self.spline = CasadiSpline(
                C_az=np.concatenate(([self.data_az[0]], params_az, [self.data_az[-1]])),
                C_el=np.concatenate(([self.data_el[0]], params_el, [self.data_el[-1]])),
                s_norm_az=self.u_vals[indices_az],
                s_norm_el=self.u_vals[indices_el],
            )
        except Exception as e:
            print("Error creating spline:", e)
            return np.full(2 * len(self.data_az), 1e7)

        az_fit = np.array(self.spline.azimuth(1.0, self.u_vals).full()).ravel()
        el_fit = np.array(self.spline.elevation(1.0, self.u_vals).full()).ravel()
        return np.concatenate([az_fit - self.data_az, el_fit - self.data_el])

    def FitSpline(self):
        """Run least-squares spline fitting."""
        print("Starting spline fitting...")
        result = least_squares(
            self.residuals,
            self.init_params,
            bounds=self.bounds,
            verbose=0,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        n = self.n_ctrl - 2
        self.fitted_params_az = result.x[:n]
        self.fitted_params_el = result.x[n : 2 * n]
        self.fitted_indices_az = np.concatenate(
            ([0], result.x[2 * n : 3 * n].astype(int), [len(self.data_az) - 1])
        )
        self.fitted_indices_el = np.concatenate(
            ([0], result.x[3 * n :].astype(int), [len(self.data_az) - 1])
        )

        self.final_spline = CasadiSpline(
            C_az=np.concatenate(
                ([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]])
            ),
            C_el=np.concatenate(
                ([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]])
            ),
            s_norm_az=self.u_vals[self.fitted_indices_az],
            s_norm_el=self.u_vals[self.fitted_indices_el],
        )

        self.spline_az_fit = np.array(
            self.final_spline.azimuth(1.0, self.u_vals).full()
        ).ravel()
        self.spline_el_fit = np.array(
            self.final_spline.elevation(1.0, self.u_vals).full()
        ).ravel()

        print(f"✅ Spline fitting completed.")

    # -------------------------------------------------------------------------
    # ------------------------- Unified Plotting -----------------------------
    # -------------------------------------------------------------------------
    def plot_fit(self, title_prefix="", ax=None, show_control_points=True):

        # --- spline plotting ---
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        else:
            fig = ax[0].get_figure()
            axes = ax

        C_az_full = np.concatenate(
            ([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]])
        )
        C_el_full = np.concatenate(
            ([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]])
        )

        axes[0].plot(self.u_vals, self.data_az, "b-", label="Data", linewidth=2)
        axes[0].plot(self.u_vals, self.spline_az_fit, "r--", label="Fitted", linewidth=2)
        if show_control_points:
            axes[0].scatter(
                self.u_vals[self.fitted_indices_az],
                C_az_full,
                c="red",
                s=25,
                edgecolors="black",
                label="Control Points",
                zorder=5,
            )
        axes[0].set_title(f"{title_prefix} Azimuth Fit")
        axes[0].set_xlabel("u parameter")
        axes[0].set_ylabel("Azimuth (rad)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.u_vals, self.data_el, "b-", label="Data", linewidth=2)
        axes[1].plot(self.u_vals, self.spline_el_fit, "r--", label="Fitted", linewidth=2)
        if show_control_points:
            axes[1].scatter(
                self.u_vals[self.fitted_indices_el],
                C_el_full,
                c="red",
                s=25,
                edgecolors="black",
                label="Control Points",
                zorder=5,
            )
        axes[1].set_title(f"{title_prefix} Elevation Fit")
        axes[1].set_xlabel("u parameter")
        axes[1].set_ylabel("Elevation (rad)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    # -------------------------------------------------------------------------
    # ------------------------- Unified Save ---------------------------------
    # -------------------------------------------------------------------------
    def save_data_RI_spline(self):
        """Save fitted spline results to pickle."""
        filename = f"fit_results_RI_Spline.pkl"
        fitted_data = {
            "n_ctrl": self.n_ctrl,
            "s_norm_az": self.u_vals[self.fitted_indices_az],
            "s_norm_el": self.u_vals[self.fitted_indices_el],
            "r0": self.r0,
            "r1": self.r1,
            "data_az": self.data_az,
            "data_el": self.data_el,
            "C_az": np.concatenate(
                ([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]])
            ),
            "C_el": np.concatenate(
                ([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]])
            ),
        }
        with open(filename, "wb") as f:
            pickle.dump(fitted_data, f)
        print(f"💾 Saved RI_Spline results to {filename}")
    
    def save_data_L_shape(self):
        """Save fitted spline or L_shape results to pickle."""
        self.segment = "L_shape"
        filename = f"fit_results_{self.segment}.pkl"

        fitted_data = {
            "segment_name": self.segment,
            "best_params": self.best_params,
            "u_vals": self.L_shape_u_vals,
            "r0": self.L_shape_r0,
            "duration": self.L_shape_duration,
            "data_az": self.L_shape_az,
            "data_el": self.L_shape_el,
            "az_fit": self.L_shape_az_fit,
            "el_fit": self.L_shape_el_fit,
        }
        with open(filename, "wb") as f:
            pickle.dump(fitted_data, f)
        print(f"💾 Saved {self.segment} results to {filename}")

    # -----------------------------------------------------------------------------
    # Function to plot the 3 splines on one 3D cycle trajectory for validation
    # -----------------------------------------------------------------------------

    def plot_spline_cart(self):

        x_cyc = self.x_cyc
        y_cyc = self.y_cyc
        z_cyc = self.z_cyc

        x, y, z = self._sph2cart(self.spline_az_fit, self.spline_el_fit, self.data_r)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot full cycle trajectory - solid line, light blue
        ax.plot(
            x_cyc,
            y_cyc,
            z_cyc,
            color="lightblue",
            linewidth=2,
            label="Full Cycle Trajectory",
            linestyle="-",
        )

        # Plot spline as dashed line
        ax.plot(x, y, z, color="r", label="Spline", linewidth=2, linestyle="--")

        # Plot start point
        ax.scatter(
            x[0],
            y[0],
            z[0],
            color="g",
            s=80,
            marker="o",
            edgecolors="black",
            label="Spline Start",
        )

        # Plot end point
        ax.scatter(
            x[-1],
            y[-1],
            z[-1],
            color="b",
            s=80,
            marker="s",
            edgecolors="black",
            label="Spline End",
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Trajectory with Fitted Spline Segments")
        ax.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0)
        plt.tight_layout()
        plt.show()

    def plot_spline_sph(self):
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        axes[0].plot(
            self.u_vals, self.spline_az_fit, "r-", label="Fitted Azimuth", linewidth=2
        )
        axes[0].plot(
            self.u_vals, self.data_az, "b--", label="Data Azimuth", linewidth=2
        )
        axes[0].set_title("Azimuth vs u parameter")
        axes[0].set_xlabel("u parameter")
        axes[0].set_ylabel("Azimuth (rad)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            self.u_vals, self.spline_el_fit, "g-", label="Fitted Elevation", linewidth=2
        )
        axes[1].plot(
            self.u_vals, self.data_el, "b--", label="Data Elevation", linewidth=2
        )
        axes[1].set_title("Elevation vs u parameter")
        axes[1].set_xlabel("u parameter")
        axes[1].set_ylabel("Elevation (rad)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_fit_L_shape(self, title_prefix=""):
        fig = plt.figure()
        plt.plot(self.L_shape_az_fit, self.L_shape_el_fit, "r-", label="Fitted Lissajous")
        plt.plot(self.L_shape_az, self.L_shape_el, "b--", label="Data")
        plt.xlabel("Azimuth (rad)")
        plt.ylabel("Elevation (rad)")
        plt.title(f"{title_prefix} Lissajous Fit")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        return fig, None


# =============================================================================
# MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    # File paths
    base_path = "./processed_data/fitting"
    waypoint_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"

    # base_path = "./processed_data/experimental"
    # waypoint_path = f"{base_path}/2024-11-05_12-58-54_ProtoLogger_waypoints.csv"
    # full_path = f"{base_path}/2024-11-05_12-58-54_ProtoLogger.csv"
    # cycle_path = f"{base_path}/2024-11-05_12-58-54_full_log.txt"

    figs = []
    axes_list = []

    fit = Fitting(full_path, cycle_path, waypoint_path, cyc_idx=0, n_ctrl_pts=25)

    # fit._setup_lissajous_segment()
    # fit._setup_spline_segment()

    # fit.FitSpline()
    # fit.save_data_RI_spline()
    # fit.plot_spline_cart()

    # fit.FitLissajous()
    # fit.save_data_L_shape()
    # fit.plot_fit_L_shape()

    # print(fit.u_vals[-1])

    # fit.plot_spline_sph()

    # if fit.az_fit[-1] == fit.data_az[-1]:
    #     print("Max s check passed: final azimuth matches data azimuth.")
    # else:
    #     print("Max s check failed: final azimuth does not match data azimuth.")
    #     print(fit.az_fit[-1], fit.data_az[-1])

    # if fit.el_fit[-1] == fit.data_el[-1]:
    #     print("Max s check passed: final elevation matches data elevation.")
    # else:
    #     print("Max s check failed: final elevation does not match data elevation.")
    #     print(fit.el_fit[-1], fit.data_el[-1])

    # s = np.linspace(0, 1, len(fit.data_az))
    # az = []
    # el = []
    # for i in s:
    #     a = np.array(fit.final_spline.azimuth(1.0, i).full()).ravel()[0]
    #     e = np.array(fit.final_spline.elevation(1.0, i).full()).ravel()[0]
    #     az.append(a)
    #     el.append(e)
    # print("Done evaluating final spline at high resolution.")

    # plt.figure()
    # plt.plot(s, az, "r-", label="Fitted Azimuth")
    # plt.plot(fit.u_vals, fit.data_az, "b--", label="Data Azimuth")
    # plt.show()

    # plt.figure()
    # plt.plot(s, el, "r-", label="Fitted Elevation")
    # plt.plot(fit.u_vals, fit.data_el, "b--", label="Data Elevation")
    # plt.show()

    # # ---------- Load precomputed fit data ----------
    # segment_name = "RI_Spline"

    # filename = f"fit_results_{segment_name}.pkl"
    # with open(filename, "rb") as f:
    #     fit_data = pickle.load(f)

    # r0 = fit_data["r0"]
    # r1 = fit_data["r1"]
    # C_az = fit_data["C_az"]
    # C_el = fit_data["C_el"]
    # s_norm_az = fit_data["s_norm_az"]
    # s_norm_el = fit_data["s_norm_el"]

    # # r0=None, r1=None, C_az=None, C_el=None, s_norm_az=None, s_norm_el=None

    # obj = CasadiSpline(
    #     r0=r0,
    #     r1=r1,
    #     C_az=C_az,
    #     C_el=C_el,
    #     s_norm_az=s_norm_az,
    #     s_norm_el=s_norm_el,
    # )

    # s = np.linspace(0, 1, len(fit.data_az))
    # az = []
    # el = []
    # for i in s:
    #     az_spline = az.append((obj.azimuth(1, i).full().ravel()[0]))
    #     el_spline = el.append((obj.elevation(1, i).full().ravel()[0]))

    # # print(az)
    # # # print(el)

    # plt.figure()
    # plt.plot(s, az, "r-", label="Fitted Azimuth")
    # # plt.plot(fit.u_vals, fit.data_az, 'b--', label='Data Azimuth')
    # plt.title("Azimuth vs u parameter from loaded spline")
    # plt.xlabel("u parameter")
    # plt.ylabel("Azimuth (rad)")
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # plt.figure()
    # plt.plot(s, el, "g-", label="Fitted Elevation")
    # # plt.plot(fit.u_vals, fit.data_el, 'b--', label='Data Elevation')
    # plt.title("Elevation vs u parameter from loaded spline")
    # plt.xlabel("u parameter")
    # plt.ylabel("Elevation (rad)")
    # plt.grid(True, alpha=0.3)
    # plt.show()

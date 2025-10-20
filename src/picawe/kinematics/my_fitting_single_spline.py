import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import least_squares

from picawe.kinematics.my_parametrized_patterns import CasadiSpline, CST_Lissajous
from picawe.kinematics.my_data_processing_single_spline import (
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
    ):
        # Initialize DataProcessing parent class
        super().__init__(file_path_full, file_path_cycle, file_path_waypoints, cyc_idx)
        self.n_ctrl = n_ctrl_pts

    # -------------------------------------------------------------------------
    # ------------------------- Spline Segment Setup --------------------------
    # -------------------------------------------------------------------------
    def _setup_spline_segment(self):
        """Prepare data for spline fitting based on segment."""

        self.data_az, self.data_el, self.u_vals, self.r0, self.r1, self.data_r = (self.Single_Spline_az, 
                                                                                  self.Single_Spline_el, 
                                                                                  self.Single_Spline_u_vals, 
                                                                                  self.Single_Spline_r0, 
                                                                                  self.Single_Spline_r1, 
                                                                                  self.Single_Spline_r)
 
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

        if not np.all(np.diff(self.u_vals[indices_az]) >= 0) or not np.all(np.diff(self.u_vals[indices_el]) >= 0):
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
        result = least_squares(
            self.residuals,
            self.init_params,
            bounds=self.bounds,
            verbose=2,
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
            C_az=np.concatenate(([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]])),
            C_el=np.concatenate(([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]])),
            s_norm_az=self.u_vals[self.fitted_indices_az],
            s_norm_el=self.u_vals[self.fitted_indices_el],
        )

        self.az_fit = np.array(self.final_spline.azimuth(1.0, self.u_vals).full()).ravel()
        self.el_fit = np.array(self.final_spline.elevation(1.0, self.u_vals).full()).ravel()

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
        axes[0].plot(self.u_vals, self.az_fit, "r--", label="Fitted", linewidth=2)
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
        axes[1].plot(self.u_vals, self.el_fit, "r--", label="Fitted", linewidth=2)
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
    def save_data(self):
        """Save fitted spline or Lissajous results to pickle."""
        filename = f"fit_results_Single_Spline.pkl"
        fitted_data = {
                "n_ctrl": self.n_ctrl,
                "s_norm_az": self.fitted_indices_az,
                "s_norm_el": self.fitted_indices_el,
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
        print(f"💾 Saved Single_Spline results to {filename}")

# -----------------------------------------------------------------------------
# Function to plot the 3 splines on one 3D cycle trajectory for validation
# -----------------------------------------------------------------------------

def plot_spline(object):

    x_cyc = object.x_cyc
    y_cyc = object.y_cyc
    z_cyc = object.z_cyc

    az = object.az_fit
    el = object.el_fit
    r = object.data_r

    x, y, z = object._sph2cart(object.az_fit, object.el_fit, object.data_r)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot full cycle trajectory - solid line, light blue
    ax.plot(x_cyc, y_cyc, z_cyc, color='lightblue', linewidth=2, 
            label='Full Cycle Trajectory', linestyle='-')

    
    # Plot spline as dashed line
    ax.plot(x, y, z, color="r", 
            label="Spline", linewidth=2, linestyle='--')
    
    # Plot start point
    ax.scatter(x[0], y[0], z[0], 
                color="g", s=80, marker='o', edgecolors='black', 
                label='Spline Start')
    
    # Plot end point
    ax.scatter(x[-1], y[-1], z[-1], 
                color="b", s=80, marker='s', edgecolors='black',
                label='Spline End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory with Fitted Spline Segments')
    ax.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    # File paths
    base_path = "./processed_data/fitting"
    waypoint_path = f"{base_path}/2025-09-25_11-48-58_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"

    figs = []
    axes_list = []

    fit = Fitting(
            full_path, cycle_path, waypoint_path, cyc_idx=0, n_ctrl_pts=30
        )

    # Set up spline segment and perform fitting
    fit._setup_spline_segment()
    fit.FitSpline()
    fit.save_data()

    # Plot all spline segments on one 3D trajectory for validation
    plot_spline(fit)


    s = np.linspace(0, 1, 10000)
    for i in s:
        print(i)
        az = fit.final_spline.azimuth(1.0, i)
        el = fit.final_spline.elevation(1.0, i)
    print("Done evaluating final spline at high resolution.")
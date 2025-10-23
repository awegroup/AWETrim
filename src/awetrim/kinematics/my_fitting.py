import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import least_squares

from awetrim.kinematics.parametrized_patterns import CasadiSpline, CST_Lissajous
from awetrim.kinematics.my_data_processing import (
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
        segment="RI",
        n_ctrl_pts=17,
    ):
        # Initialize DataProcessing parent class
        super().__init__(file_path_full, file_path_cycle, file_path_waypoints, cyc_idx)
        self.segment = segment.upper()
        self.n_ctrl = n_ctrl_pts

        # Select segment data
        if self.segment in ["RI", "RI_RO", "RO_RI"]:
            self._setup_spline_segment()
            self.FitSpline()
        elif self.segment == "LISSAJOUS":
            self._setup_lissajous_segment()
            self.FitLissajous()
        else:
            raise ValueError(f"Unknown segment '{segment}' for fitting.")

    # -------------------------------------------------------------------------
    # ------------------------- Spline Segment Setup --------------------------
    # -------------------------------------------------------------------------
    def _setup_spline_segment(self):
        """Prepare data for spline fitting based on segment."""

        self.data_az, self.data_el, self.u_vals, self.r0, self.r1, self.data_r = (
                getattr(self, f"{self.segment}_az"),
                getattr(self, f"{self.segment}_el"),
                getattr(self, f"{self.segment}_u_vals"),
                getattr(self, f"{self.segment}_r0"),
                getattr(self, f"{self.segment}_r1"),
                getattr(self, f"{self.segment}_r")
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

        if not np.all(np.diff(self.u_vals[indices_az]) >= 0) or not np.all(np.diff(self.u_vals[indices_el]) >= 0):
            return np.full(2 * len(self.data_az), 1e7)

        try:
            spline = CasadiSpline(
                C_az=np.concatenate(([self.data_az[0]], params_az, [self.data_az[-1]])),
                C_el=np.concatenate(([self.data_el[0]], params_el, [self.data_el[-1]])),
                s_norm_az=self.u_vals[indices_az],
                s_norm_el=self.u_vals[indices_el],
            )
        except Exception as e:
            print("Error creating spline:", e)
            return np.full(2 * len(self.data_az), 1e7)
    
        az_fit = np.array(spline.azimuth(1.0, self.u_vals).full()).ravel()
        el_fit = np.array(spline.elevation(1.0, self.u_vals).full()).ravel()
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

        spline = CasadiSpline(
            C_az=np.concatenate(([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]])),
            C_el=np.concatenate(([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]])),
            s_norm_az=self.u_vals[self.fitted_indices_az],
            s_norm_el=self.u_vals[self.fitted_indices_el],
        )

        self.az_fit = np.array(spline.azimuth(1.0, self.u_vals).full()).ravel()
        self.el_fit = np.array(spline.elevation(1.0, self.u_vals).full()).ravel()

        print(f"✅ {self.segment} spline fitting completed.")

    # -------------------------------------------------------------------------
    # ------------------------ Lissajous Segment Setup ------------------------
    # -------------------------------------------------------------------------
    def _setup_lissajous_segment(self):
        """Prepare data for Lissajous fitting from RO Lissajous."""
        self.data_az = self.Lissajous_az
        self.data_el = self.Lissajous_el
        self.data_r = self.Lissajous_r
        self.Lissajous_r0 = self.Lissajous_r0
        self.u_vals = np.linspace(0, 2 * np.pi, len(self.data_az))

    def FitLissajous(self):
        """Run least-squares Lissajous fitting."""
        fixed_params = {
            "omega": 1,
            "r0": self.Lissajous_r0,
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
            az_model = obj.azimuth(params["r0"], self.u_vals)
            el_model = obj.elevation(params["r0"], self.u_vals)
            return np.concatenate(
                (self.data_az - az_model, self.data_el - el_model)
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
        self.az_fit = obj.azimuth(self.best_params["r0"], self.u_vals)
        self.el_fit = obj.elevation(self.best_params["r0"], self.u_vals)
        print("✅ Lissajous fitting completed.")

    # -------------------------------------------------------------------------
    # ------------------------- Unified Plotting -----------------------------
    # -------------------------------------------------------------------------
    def plot_fit(self, title_prefix="", ax=None, show_control_points=True):
        """Unified plotting for spline or Lissajous."""
        if self.segment == "LISSAJOUS":
            fig = plt.figure()
            plt.plot(self.az_fit, self.el_fit, "r-", label="Fitted Lissajous")
            plt.plot(self.data_az, self.data_el, "b--", label="Data")
            plt.xlabel("Azimuth (rad)")
            plt.ylabel("Elevation (rad)")
            plt.title(f"{title_prefix} Lissajous Fit")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            return fig, None

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

        spline = CasadiSpline(
            C_az=C_az_full,
            C_el=C_el_full,
            s_norm_az=self.u_vals[self.fitted_indices_az],
            s_norm_el=self.u_vals[self.fitted_indices_el],
        )

        fitted_az = np.array(spline.azimuth(1.0, self.u_vals).full()).ravel()
        fitted_el = np.array(spline.elevation(1.0, self.u_vals).full()).ravel()

        axes[0].plot(self.u_vals, self.data_az, "b-", label="Data", linewidth=2)
        axes[0].plot(self.u_vals, fitted_az, "r--", label="Fitted", linewidth=2)
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
        axes[1].plot(self.u_vals, fitted_el, "r--", label="Fitted", linewidth=2)
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
        filename = f"fit_results_{self.segment}.pkl"
        if self.segment == "LISSAJOUS":
            fitted_data = {
                "segment_name": self.segment,
                "best_params": self.best_params,
                "u_vals": self.u_vals,
                "r0": self.Lissajous_r0,
                "duration": self.Lissajous_Duration,
                "data_az": self.data_az,
                "data_el": self.data_el,
                "az_fit": self.az_fit,
                "el_fit": self.el_fit,
            }
        else:
            fitted_data = {
                "segment_name": self.segment,
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
        print(f"💾 Saved {self.segment} results to {filename}")

# -----------------------------------------------------------------------------
# Function to plot the 3 splines on one 3D cycle trajectory for validation
# -----------------------------------------------------------------------------

def plot_all_splines(objects):

    az = [] 
    el = [] 
    r = []

    x_list = []
    y_list = []
    z_list = []

    x_cyc = objects[0].x_cyc
    y_cyc = objects[0].y_cyc
    z_cyc = objects[0].z_cyc

    for fit in objects:
        az.append(fit.az_fit)
        el.append(fit.el_fit)
        r.append(fit.data_r)

        x, y, z = fit._sph2cart(fit.az_fit, fit.el_fit, fit.data_r)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot full cycle trajectory - solid line, light blue
    ax.plot(x_cyc, y_cyc, z_cyc, color='lightblue', linewidth=2, 
            label='Full Cycle Trajectory', linestyle='-')
    
    # Define colors and labels for each segment
    colors = ['lightgreen', 'orange', 'hotpink']
    labels = ['RI Segment', 'RIRO Segment', 'RORI Segment']
    
    # Plot spline segments with dashed lines and start/end points
    for i in range(len(objects)):
        # Plot spline as dashed line
        ax.plot(x_list[i], y_list[i], z_list[i], color=colors[i], 
                label=labels[i], linewidth=2, linestyle='--')
        
        # Plot start point
        ax.scatter(x_list[i][0], y_list[i][0], z_list[i][0], 
                  color=colors[i], s=80, marker='o', edgecolors='black', 
                  label=f'{labels[i]} Start')
        
        # Plot end point
        ax.scatter(x_list[i][-1], y_list[i][-1], z_list[i][-1], 
                  color=colors[i], s=80, marker='s', edgecolors='black',
                  label=f'{labels[i]} End')
    
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
    waypoint_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"

    segments = ["RI", "RI_RO", "RO_RI", "LISSAJOUS"]

    figs = []
    axes_list = []

    objects = []

    for seg in segments:
        print(f"\n🔹 Fitting {seg} segment...")
        fit = Fitting(
            full_path, cycle_path, waypoint_path, cyc_idx=0, segment=seg, n_ctrl_pts=17
        )
        if seg != "LISSAJOUS":
            objects.append(fit)
        
        fit.save_data()
        fig, axes = fit.plot_fit(title_prefix=seg)
        figs.append(fig)
        axes_list.append(axes)
    plt.show()

    # Plot all spline segments on one 3D trajectory for validation
    plot_all_splines(objects)
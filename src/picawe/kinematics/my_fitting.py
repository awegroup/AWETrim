import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import casadi as ca
import pickle
from scipy.optimize import least_squares

from picawe.kinematics.my_parametrized_patterns import CasadiSpline as build
from picawe.kinematics.my_RI_data_processing import RI_data_processing
from picawe.kinematics.my_RI_RO_data_processing import RI_RO_data_processing
from picawe.kinematics.my_RO_RI_data_processing import RO_RI_data_processing


class Fitting(build):
    """Performs spline fitting on azimuth and elevation data."""

    def __init__(self, az_data, el_data, u_vals, n_ctrl_pts=8):
        # Parameters
        self.n_ctrl = n_ctrl_pts
        self.data_az = az_data
        self.data_el = el_data
        self.u_vals = u_vals

        # Choose evenly spaced indices (excluding endpoints)
        self.indices0 = np.linspace(0, len(az_data) - 1, n_ctrl_pts, dtype=int)[1:-1]

        # Initialize control points
        self.initial_params_az = az_data[self.indices0]
        self.initial_params_el = el_data[self.indices0]

        # Concatenate all initial parameters into a flat array
        self.init_params = np.concatenate([
            self.initial_params_az,
            self.initial_params_el,
            self.indices0,
            self.indices0
        ])

        # Parameter bounds
        bounds_az = (np.full(n_ctrl_pts - 2, -np.pi / 2), np.full(n_ctrl_pts - 2, np.pi / 2))
        bounds_el = (np.full(n_ctrl_pts - 2, -0.01), np.full(n_ctrl_pts - 2, np.pi / 2 + 0.01))
        bounds_idx = (np.full(n_ctrl_pts - 2, 2), np.full(n_ctrl_pts - 2, len(az_data) - 3))

        self.bounds = (
            np.concatenate([bounds_az[0], bounds_el[0], bounds_idx[0], bounds_idx[0]]),
            np.concatenate([bounds_az[1], bounds_el[1], bounds_idx[1], bounds_idx[1]])
        )

        # Perform the fit
        self.Fit()

    # -------------------------------------------------------------------------
    def residuals(self, params):
        """Compute residuals between data and fitted spline."""
        n = self.n_ctrl - 2
        params_az = params[:n]
        params_el = params[n:2 * n]
        indices_az = np.concatenate(([0], params[2 * n:3 * n].astype(int), [len(self.data_az)-1]))
        indices_el = np.concatenate(([0], params[3 * n:].astype(int), [len(self.data_az)-1]))

        if not np.all(np.diff(self.u_vals[indices_az]) > 0):
            print("⚠️ Non-increasing u_vals[indices_az]:", self.u_vals[indices_az])
            return np.full(2 * len(self.data_az), 1e6)


        spline = build(
            C_az=np.concatenate(([self.data_az[0]], params_az, [self.data_az[len(self.data_az)-1]])),
            C_el=np.concatenate(([self.data_el[0]], params_el, [self.data_el[len(self.data_az)-1]])),
            s_norm_az=self.u_vals[indices_az],
            s_norm_el=self.u_vals[indices_el]
        )

        az = spline.azimuth(1.0, self.u_vals)
        el = spline.elevation(1.0, self.u_vals)

        return np.concatenate([az - self.data_az, el - self.data_el])

    # -------------------------------------------------------------------------
    def Fit(self):
        """Run least-squares fitting."""
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
        self.fitted_params_el = result.x[n:2 * n]
        self.fitted_indices_az = np.concatenate(([0], result.x[2 * n:3 * n].astype(int), [len(self.data_az)-1]))
        print("Fitted azimuth indices:", self.fitted_indices_az)
        self.fitted_indices_el = np.concatenate(([0], result.x[3 * n:].astype(int), [len(self.data_az)-1]))
        print("Fitted elevation indices:", self.fitted_indices_el)
        print("✅ Fitting completed.")

    # -------------------------------------------------------------------------
    def plot_fit(self, title_prefix="", ax=None, show_control_points=True):
        """Plot the original and fitted spline for azimuth and elevation."""
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        else:
            fig = ax[0].get_figure()
            axes = ax

        # Build fitted spline
        C_az_full = np.concatenate(([self.data_az[0]], self.fitted_params_az, [self.data_az[len(self.data_az)-1]]))
        C_el_full = np.concatenate(([self.data_el[0]], self.fitted_params_el, [self.data_el[len(self.data_az)-1]]))

        spline = build(
            C_az=C_az_full,
            C_el=C_el_full,
            s_norm_az=self.u_vals[self.fitted_indices_az],
            s_norm_el=self.u_vals[self.fitted_indices_el]
        )

        fitted_az = spline.azimuth(1.0, self.u_vals)
        fitted_el = spline.elevation(1.0, self.u_vals)

        # --- Azimuth Plot ---
        axes[0].plot(self.u_vals, self.data_az, 'b-', label='Data', linewidth=2)
        axes[0].plot(self.u_vals, fitted_az, 'r--', label='Fitted', linewidth=2)
        if show_control_points:
            axes[0].scatter(
                self.u_vals[self.fitted_indices_az], C_az_full,
                c='red', s=25, edgecolors='black', label='Control Points', zorder=5
            )
        axes[0].set_title(f'{title_prefix} Azimuth Fit')
        axes[0].set_xlabel('u parameter')
        axes[0].set_ylabel('Azimuth (rad)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Elevation Plot ---
        axes[1].plot(self.u_vals, self.data_el, 'b-', label='Data', linewidth=2)
        axes[1].plot(self.u_vals, fitted_el, 'r--', label='Fitted', linewidth=2)
        if show_control_points:
            axes[1].scatter(
                self.u_vals[self.fitted_indices_el], C_el_full,
                c='red', s=25, edgecolors='black', label='Control Points', zorder=5
            )
        axes[1].set_title(f'{title_prefix} Elevation Fit')
        axes[1].set_xlabel('u parameter')
        axes[1].set_ylabel('Elevation (rad)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    # -------------------------------------------------------------------------
    def save_data(self, segment_name=""):
        """Save fitted spline results to a pickle file."""
        fitted_data = {
            'segment_name': segment_name,
            'n_ctrl_pts': self.n_ctrl,
            's_norm_az': self.fitted_indices_az,
            's_norm_el': self.fitted_indices_el,
            'original_data_az': self.data_az,
            'original_data_el': self.data_el,
            'C_az': np.concatenate(([self.data_az[0]], self.fitted_params_az, [self.data_az[len(self.data_az)-1]])),
            'C_el': np.concatenate(([self.data_el[0]], self.fitted_params_el, [self.data_el[len(self.data_az)-1]])),
        }

        filename = f"fit_results_{segment_name}.pkl" if segment_name else "fit_results.pkl"
        with open(filename, "wb") as f:
            pickle.dump(fitted_data, f)

        print(f"💾 Saved results to {filename}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    # File paths
    base_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2"
    waypoint_path = f"{base_path}/waypoints/2025-09-25_11-48-58_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycles/cycle_data_sheet_lines.csv"

    # --- Reel In ---
    print("🔹 Fitting RI segment...")
    RI = RI_data_processing(full_path, cycle_path, waypoint_path, cyc_idx=0)
    fit_RI = Fitting(RI.RI_az, RI.RI_el, RI.u_vals, n_ctrl_pts=15)
    fit_RI.save_data("RI")

    # --- Reel In to Reel Out ---
    print("\n🔹 Fitting RI_RO segment...")
    RIRO = RI_RO_data_processing(full_path, cycle_path, waypoint_path, cyc_idx=0)
    fit_RIRO = Fitting(RIRO.RI_RO_az, RIRO.RI_RO_el, RIRO.u_vals)
    fit_RIRO.save_data("RI_RO")

    # --- Reel Out to Reel In ---
    print("\n🔹 Fitting RO_RI segment...")
    RORI = RO_RI_data_processing(full_path, cycle_path, waypoint_path, cyc_idx=0)
    fit_RORI = Fitting(RORI.RO_RI_az, RORI.RO_RI_el, RORI.u_vals)
    fit_RORI.save_data("RO_RI")

    # --- Plot all segments ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fit_RI.plot_fit(title_prefix="RI", ax=axes[:, 0])
    fit_RIRO.plot_fit(title_prefix="RI_RO", ax=axes[:, 1])
    fit_RORI.plot_fit(title_prefix="RO_RI", ax=axes[:, 2])

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import casadi as ca
import pickle
from scipy.optimize import least_squares

from awetrim.kinematics.parametrized_patterns import CasadiSpline as build
from awetrim.kinematics.Other_code_by_theophile_dubois.my_RI_data_processing import RI_data_processing
from awetrim.kinematics.Other_code_by_theophile_dubois.my_RI_RO_data_processing import RI_RO_data_processing
from awetrim.kinematics.Other_code_by_theophile_dubois.my_RO_RI_data_processing import RO_RI_data_processing
from awetrim.kinematics.Other_code_by_theophile_dubois.my_Lisajous_fitting import Lisajous_fitting

from awetrim.kinematics.my_data_processing import DataProcessing  # Your refactored Data_processing class


class Fitting(DataProcessing):
    """Spline fitting on azimuth and elevation data, inherits DataProcessing."""

    def __init__(self, file_path_full, file_path_cycle, file_path_waypoints, cyc_idx=0,
                 segment="RI", n_ctrl_pts=8):
        # Initialize parent class to load and preprocess data
        super().__init__(file_path_full, file_path_cycle, file_path_waypoints, cyc_idx)

        self.segment = segment
        self.n_ctrl = n_ctrl_pts

        # Determine which segment to fit
        if segment.upper() == "RI":
            self.data_az, self.data_el, self.u_vals = self.RI_az, self.RI_el, self.RI_u_vals
            self.r0, self.r1 = self.RI_r0, self.RI_r1
        elif segment.upper() == "RI_RO":
            self.data_az, self.data_el, self.u_vals = self.RI_RO_az, self.RI_RO_el, self.RI_RO_u_vals
            self.r0, self.r1 = self.RI_RO_r0, self.RI_RO_r1
        elif segment.upper() == "RO_RI":
            self.data_az, self.data_el, self.u_vals = self.RO_RI_az, self.RO_RI_el, self.RO_RI_u_vals
            self.r0, self.r1 = self.RO_RI_r0, self.RO_RI_r1
        else:
            raise ValueError(f"Unknown segment '{segment}' for fitting.")

        # Initial control points
        self.indices0 = np.linspace(0, len(self.data_az) - 1, self.n_ctrl, dtype=int)[1:-1]
        self.initial_params_az = self.data_az[self.indices0]
        self.initial_params_el = self.data_el[self.indices0]
        self.init_params = np.concatenate([
            self.initial_params_az,
            self.initial_params_el,
            self.indices0,
            self.indices0
        ])

        # Parameter bounds
        bounds_az = (np.full(self.n_ctrl-2, -np.pi/2), np.full(self.n_ctrl-2, np.pi/2))
        bounds_el = (np.full(self.n_ctrl-2, -0.01), np.full(self.n_ctrl-2, np.pi/2+0.01))
        bounds_idx = (np.full(self.n_ctrl-2, 2), np.full(self.n_ctrl-2, len(self.data_az)-3))
        self.bounds = (
            np.concatenate([bounds_az[0], bounds_el[0], bounds_idx[0], bounds_idx[0]]),
            np.concatenate([bounds_az[1], bounds_el[1], bounds_idx[1], bounds_idx[1]])
        )

        # Fit the spline
        self.Fit()

    # -------------------------------------------------------------------------
    def residuals(self, params):
        n = self.n_ctrl - 2
        params_az = params[:n]
        params_el = params[n:2*n]
        indices_az = np.concatenate(([0], params[2*n:3*n].astype(int), [len(self.data_az)-1]))
        indices_el = np.concatenate(([0], params[3*n:].astype(int), [len(self.data_az)-1]))

        # Check monotonicity of u_vals
        if not np.all(np.diff(self.u_vals[indices_az]) > 0):
            return np.full(2*len(self.data_az), 1e6)

        spline = build(
            C_az=np.concatenate(([self.data_az[0]], params_az, [self.data_az[-1]])),
            C_el=np.concatenate(([self.data_el[0]], params_el, [self.data_el[-1]])),
            s_norm_az=self.u_vals[indices_az],
            s_norm_el=self.u_vals[indices_el]
        )

        az_fit = np.array(spline.azimuth(1.0, self.u_vals).full()).ravel()
        el_fit = np.array(spline.elevation(1.0, self.u_vals).full()).ravel()
        return np.concatenate([az_fit - self.data_az, el_fit - self.data_el])

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
            gtol=1e-10
        )
        n = self.n_ctrl - 2
        self.fitted_params_az = result.x[:n]
        self.fitted_params_el = result.x[n:2*n]
        self.fitted_indices_az = np.concatenate(([0], result.x[2*n:3*n].astype(int), [len(self.data_az)-1]))
        self.fitted_indices_el = np.concatenate(([0], result.x[3*n:].astype(int), [len(self.data_az)-1]))
        print(f"✅ {self.segment} fitting completed.")

    # -------------------------------------------------------------------------
    def plot_fit(self, title_prefix="", ax=None, show_control_points=True):
        """Plot the original and fitted spline for azimuth and elevation."""
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        else:
            fig = ax[0].get_figure()
            axes = ax

        # Build fitted spline
        C_az_full = np.concatenate(([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]]))
        C_el_full = np.concatenate(([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]]))

        spline = build(
            C_az=C_az_full,
            C_el=C_el_full,
            s_norm_az=self.u_vals[self.fitted_indices_az],
            s_norm_el=self.u_vals[self.fitted_indices_el]
        )

        fitted_az = np.array(spline.azimuth(1.0, self.u_vals).full()).ravel()
        fitted_el = np.array(spline.elevation(1.0, self.u_vals).full()).ravel()

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
    def save_data(self):
        """Save fitted spline results to a pickle file."""
        filename = f"fit_results_{self.segment}.pkl"
        fitted_data = {
            'segment_name': self.segment,
            'n_ctrl': self.n_ctrl,
            's_norm_az': self.fitted_indices_az,
            's_norm_el': self.fitted_indices_el,
            'r0': self.r0,
            'r1': self.r1,
            'data_az': self.data_az,
            'data_el': self.data_el,
            'C_az': np.concatenate(([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]])),
            'C_el': np.concatenate(([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]])),
        }
        with open(filename, "wb") as f:
            pickle.dump(fitted_data, f)
        print(f"💾 Saved {self.segment} results to {filename}")



# =============================================================================
# MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    # File paths
    base_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2"
    waypoint_path = f"{base_path}/waypoints/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/ProtoLogger_csv/2025-10-23_09-43-50_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycles/cycle_data_sheet_lines.csv"

    segments = ["RI", "RI_RO", "RO_RI"]

    figs = []
    axes_list = []

    for seg in segments:
        print(f"\n🔹 Fitting {seg} segment...")
        fit = Fitting(full_path, cycle_path, waypoint_path, cyc_idx=0, segment=seg, n_ctrl_pts=15)
        fit.save_data()
        fig, axes = fit.plot_fit(title_prefix=seg)
        figs.append(fig)
        axes_list.append(axes)

    # Optional: Show all figures together
    plt.show()

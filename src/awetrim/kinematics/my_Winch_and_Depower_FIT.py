import json
import numpy as np
import pickle
import casadi as ca
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from awetrim.kinematics.my_Winch_and_Depower_DP import Winch_and_Depower_data_processing
from awetrim.system.winch import Winch
from awetrim.kinematics.winch_force_curve import (
    WinchControllerCharacteristics,
    WinchControllerParameters,
)


class WinchCurveFitter:
    def __init__(self, json_path, base_path, cycle_idx=0, n_knots=25, run_plots=True, run_plots_DP=True):
        self.cycle_idx = cycle_idx
        self.json_path = json_path
        self.base_path = base_path
        self.gravity = 9.81  # m/s^2
        self.n_knots = n_knots
        self.run_plots = run_plots
        self.run_plots_DP = run_plots_DP

        # Load all data
        self._load_data()
        self.run()

        if self.run_plots:
            # Example plots
            for i in range(len(self.processed.RI_Spline_phase_settings)):
                self.plot_example(self.SS_curve_data_stored, self.SS_final_params, phase_index=i)

            # RO phase example
            self.plot_example(self.RO_curve_data_stored, self.RO_final_params, phase_index=0)

    # ----------------------------
    # Data loading and preparation
    # ----------------------------
    def _load_data(self):
        """Load logger CSVs and JSON trajectory."""
        with open(self.json_path) as f:
            data = json.load(f)

        json_trajectory = data["trajectory"]

        # File paths
        base_path = "./processed_data/fitting"
        waypoint_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
        full_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger.csv"
        cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"

        # base_path = "./processed_data/experimental"
        # waypoint_path = f"{base_path}/2024-11-05_12-58-54_ProtoLogger_waypoints.csv"
        # full_path = f"{base_path}/2024-11-05_12-58-54_ProtoLogger.csv"
        # cycle_path = f"{base_path}/2024-11-05_12-58-54_full_log.txt"

        # custom data-processing class
        self.processed = Winch_and_Depower_data_processing(
            full_path, cycle_path, waypoint_path, json_trajectory, cyc_idx=self.cycle_idx, run_plots_DP=self.run_plots_DP
        )

    # ----------------------------
    # Residuals for least squares
    # ----------------------------
    def residuals(self, params, v_m_data, force_data, v_knots):
        """
        Compute residuals between spline tension and measured force data.
        """

        C = params  # tension values at knots

        spline = ca.interpolant("T_spline", "bspline", [v_knots], np.array(C))
        T_vals = np.array([float(spline(v)) for v in v_m_data])  # predicted tension values
        return T_vals - force_data

    # ----------------------------
    # Fit function
    # ----------------------------
    def fit_winch_curve(self, v_m_data, force_data):
        """
        Fit a CasADi spline to measured (v_m, force) data.
        n_knots: number of spline nodes used for fitting
        """
        # Define knot positions evenly spaced across the velocity range
        v_knots = np.linspace(np.min(v_m_data), np.max(v_m_data), self.n_knots)

        # Initial guess for tension values at each knot
        C_init = np.linspace(np.min(force_data), np.max(force_data), self.n_knots)

        initial_guess = C_init

        # Run least squares optimization
        result = least_squares(
            self.residuals,
            initial_guess,
            args=(v_m_data, force_data, v_knots),
            method="trf",
            loss="soft_l1",
            f_scale=0.5,
            verbose=0,
        )

        # Build final spline model with fitted parameters
        fitted_params = result.x
        C_fitted = fitted_params
        fitted_spline = ca.interpolant("T_spline", "bspline", [v_knots], np.array(C_fitted))

        print("✅ Winch spline fit completed. \n")
        # print(f"Fitted parameters: {np.round(fitted_params, 3)} \n")

        # Save for later use
        self.v_knots = v_knots
        self.C_fitted = C_fitted
        self.fitted_spline = fitted_spline
        return fitted_spline, self.v_knots, self.C_fitted

    # --------------------------------------------
    # Run fitting for all RI_Spline phases
    # --------------------------------------------
    def run(self):
        curve_data_stored_SS = []
        final_params_SS = []

        curve_data_stored_RO = []
        final_params_RO = []

        idx = 0

        for settings in self.processed.RI_Spline_phase_settings: 

            s = settings["s"]
            depower = settings["depower"]

            params = WinchControllerParameters(
                f_min=settings["f_low"],
                f_max=settings["f_high"],
                v_cmd=settings["reelout_speed"],
                force_slope_factor=settings["force_slope_factor"],
                force_knee=settings["force_knee"],
                p_gain_v=settings["kp_v"],
                p_gain_f=settings["kp_f"],
            )

            # True winch curve (from controller characteristics)
            KP_winch = WinchControllerCharacteristics()
            winch_curve_force_data = (
                np.array(
                    KP_winch.get_effective_controller_function(
                        v_cmd=params.v_cmd,
                        p_gain_v=params.p_gain_v,
                        p_gain_f=params.p_gain_f,
                        f_min=params.f_min,
                        f_max=params.f_max,
                        f_knee=params.force_knee,
                        force_slope_factor=params.force_slope_factor,
                    )
                )
                * self.gravity
            )
            winch_curve_velocity_data = KP_winch.v_m_list

            curve_data_stored_SS.append(
                {
                    "force": winch_curve_force_data,
                    "velocity": winch_curve_velocity_data,
                    "KP_settings": settings,
                }
            )
            # Fit parameters to reproduce the true curve
            print(f"Fitting RI_Spline phase {idx} ... ")
            fitted_spline, v_knots, C_fitted = self.fit_winch_curve(
                winch_curve_velocity_data, winch_curve_force_data
            )

            idx += 1

            final_params_SS.append({                                 
                                    "s": s,
                                    "v_knots": v_knots,
                                    "C_fitted": C_fitted,
                                    "depower": depower
                                    })

        # Save the list to a pickle file
        with open("fit_winch_results_RI_Spline_phase_settings.pkl", "wb") as f:
            pickle.dump(final_params_SS, f)

        self.SS_curve_data_stored = curve_data_stored_SS
        self.SS_final_params = final_params_SS

        # --------------------------------------------
        # Run fitting for RO phase
        # --------------------------------------------

        settings = self.processed.RO_phase_settings

        s = settings["s"]
        depower = settings["depower"]

        params = WinchControllerParameters(
            f_min=settings["f_low"],
            f_max=settings["f_high"],
            v_cmd=settings["reelout_speed"],
            force_slope_factor=settings["force_slope_factor"],
            force_knee=settings["force_knee"],
            p_gain_v=settings["kp_v"],
            p_gain_f=settings["kp_f"],
        )

        # True winch curve (from controller characteristics)
        KP_winch = WinchControllerCharacteristics()
        winch_curve_force_data = (
            np.array(
                KP_winch.get_effective_controller_function(
                    v_cmd=params.v_cmd,
                    p_gain_v=params.p_gain_v,
                    p_gain_f=params.p_gain_f,
                    f_min=params.f_min,
                    f_max=params.f_max,
                    f_knee=params.force_knee,
                    force_slope_factor=params.force_slope_factor,
                )
            )
            * self.gravity
        )
        winch_curve_velocity_data = KP_winch.v_m_list

        curve_data_stored_RO.append(
            {
                "force": winch_curve_force_data,
                "velocity": winch_curve_velocity_data,
                "KP_settings": settings,
            }
        )
        print(f"Fitting winch curve for RO phase ...")

        # Fit parameters to reproduce the true curve
        fitted_spline, v_knots, C_fitted = self.fit_winch_curve(
            winch_curve_velocity_data, winch_curve_force_data
        )

        final_params_RO.append({"s": s,
                                "v_knots": v_knots,
                                "C_fitted": C_fitted,
                                "depower": depower
                                })

        # Save the list to a pickle file
        with open("fit_winch_results_RO_phase_settings.pkl", "wb") as f:
            pickle.dump(final_params_RO, f)

        self.RO_curve_data_stored = curve_data_stored_RO
        self.RO_final_params = final_params_RO

        return self.SS_curve_data_stored, self.SS_final_params, self.RO_curve_data_stored, self.RO_final_params

    # ----------------------------
    # Plot example comparison
    # ----------------------------
    def plot_example(self, curve_data, params, phase_index=0):
        data = curve_data[phase_index]
        fitted = params[phase_index]

        v_data = data["velocity"]
        f_true = data["force"]

        T_fun_fit = ca.interpolant(
            "T_spline_fit",
            "bspline",
            [fitted["v_knots"]],
            np.array(fitted["C_fitted"]),
        )

        f_fit = np.array([float(T_fun_fit(v)) for v in v_data])

        plt.figure(figsize=(8, 5))
        plt.plot(v_data, f_true, label="True Winch Curve", linewidth=2)
        plt.plot(v_data, f_fit, "--", label="Fitted Winch Curve", linewidth=2)
        plt.xlabel("Velocity [m/s]")
        plt.ylabel("Force [N]")
        plt.title(f"Winch Curve Fit — Phase {phase_index}")
        plt.legend()
        plt.grid(True)
        plt.show()


# ----------------------------
# Run script
# ----------------------------
if __name__ == "__main__":
    json_path = "src/awetrim/kinematics/pp_ws6-9_GS3_KCU4.A_KiteV9.60.A.json"
    base_path = "./processed_data/fitting"

    fitter = WinchCurveFitter(json_path, base_path, cycle_idx=0)
    # fitter.run()

    # for i in range(len(fitter.processed.RI_Spline_phase_settings)):
    #     fitter.plot_example(fitter.SS_curve_data_stored, fitter.SS_final_params, phase_index=i)

    # # RO phase example
    # fitter.plot_example(fitter.RO_curve_data_stored, fitter.RO_final_params, phase_index=0)
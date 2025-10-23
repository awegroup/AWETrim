import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from awetrim.kinematics.my_Winch_and_Depower_data_processing import Winch_and_Depower_data_processing
from awetrim.system.winch import Winch
from awetrim.kinematics.winch_force_curve import (
    WinchControllerCharacteristics,
    WinchControllerParameters,
)

class WinchCurveFitter:
    def __init__(self, json_path, base_path):
        self.json_path = json_path
        self.base_path = base_path
        self.gravity = 9.81  # m/s^2

        # Load all data
        self._load_data()

    # ----------------------------
    # Data loading and preparation
    # ----------------------------
    def _load_data(self):
        with open(self.json_path) as f:
            data = json.load(f)

        json_trajectory = data["trajectory"]

        waypoint_path = f"{self.base_path}/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
        full_path = f"{self.base_path}/2025-10-23_09-43-50_ProtoLogger.csv"
        cycle_path = f"{self.base_path}/cycle_data_sheet_lines.csv"

        self.processed = Winch_and_Depower_data_processing(
            full_path, cycle_path, waypoint_path, json_trajectory
        )

    # ----------------------------
    # Residuals for least squares
    # ----------------------------
    def residuals(self, params, v_m_data, force_data):
        pattern_config = {
            "force_model": "quadratic",
            "max_tether_force": params[0],
            "min_tether_force": params[1],
            "softplus": True,
            "softplus_beta": params[2],
            "softminus": True,
            "softminus_beta": params[3],
            "slope": params[4],
            "offset": params[5],
        }

        winch = Winch(pattern_config=pattern_config)

        T_fun = winch.tension_curve
        T_vals = np.array([float(T_fun(v)) for v in v_m_data])

        return T_vals - force_data

    # ----------------------------
    # Fit function
    # ----------------------------
    def fit_winch_curve(self, v_m_data, force_data):
        initial_guess = [
            max(force_data),  # max_tether_force
            min(force_data),  # min_tether_force
            1e-4,  # softplus_beta
            1e-3,  # softminus_beta
            1400.0,  # slope
            -2.8,  # offset
        ]

        result = least_squares(self.residuals, initial_guess, args=(v_m_data, force_data))
        return result.x

    # --------------------------------------------
    # Run fitting for all Single_Spline phases
    # --------------------------------------------
    def run(self):
        curve_data_stored = []
        final_fitted_params = []

        for settings in self.processed.Single_Spline_phase_settings: 

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

            curve_data_stored.append(
                {
                    "force": winch_curve_force_data,
                    "velocity": winch_curve_velocity_data,
                    "KP_settings": settings,
                }
            )

            # Fit parameters to reproduce the true curve
            fitted_params = self.fit_winch_curve(
                winch_curve_velocity_data, winch_curve_force_data
            )

            final_fitted_params.append(
                {
                    "max_tether_force": fitted_params[0],
                    "min_tether_force": fitted_params[1],
                    "softplus_beta": fitted_params[2],
                    "softminus_beta": fitted_params[3],
                    "slope": fitted_params[4],
                    "offset": fitted_params[5],
                    "s": s,
                    "depower": depower,
                }
            )

        # Save the list to a pickle file
        with open("fit_winch_results_Single_Spline_phase_settings.pkl", "wb") as f:
            pickle.dump(final_fitted_params, f)

        self.SS_curve_data_stored = curve_data_stored
        self.SS_final_fitted_params = final_fitted_params
        
        # Save the RO winch data
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

        curve_data_stored.append(
            {
                "force": winch_curve_force_data,
                "velocity": winch_curve_velocity_data,
                "KP_settings": settings,
            }
        )

        # Fit parameters to reproduce the true curve
        fitted_params = self.fit_winch_curve(
            winch_curve_velocity_data, winch_curve_force_data
        )

        final_fitted_params.append(
            {
                "max_tether_force": fitted_params[0],
                "min_tether_force": fitted_params[1],
                "softplus_beta": fitted_params[2],
                "softminus_beta": fitted_params[3],
                "slope": fitted_params[4],
                "offset": fitted_params[5],
                "s": s,
                "depower": depower,
            }
        )

        # Save the list to a pickle file
        with open("fit_winch_results_RO_phase_settings.pkl", "wb") as f:
            pickle.dump(final_fitted_params, f)

        self.RO_curve_data_stored = curve_data_stored
        self.RO_final_fitted_params = final_fitted_params

        return self.SS_curve_data_stored, self.SS_final_fitted_params, self.RO_curve_data_stored, self.RO_final_fitted_params

    # ----------------------------
    # Plot example comparison
    # ----------------------------
    def plot_example(self, curve_data, fitted_params, phase_index=0):
        data = curve_data[phase_index]
        fitted = fitted_params[phase_index]

        v_data = data["velocity"]
        f_true = data["force"]

        pattern_config = {
            "force_model": "quadratic",
            "max_tether_force": fitted["max_tether_force"],
            "min_tether_force": fitted["min_tether_force"],
            "softplus": True,
            "softplus_beta": fitted["softplus_beta"],
            "softminus": True,
            "softminus_beta": fitted["softminus_beta"],
            "slope": fitted["slope"],
            "offset": fitted["offset"],
        }

        winch_fit = Winch(pattern_config=pattern_config)
        T_fun_fit = winch_fit.tension_curve
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

    fitter = WinchCurveFitter(json_path, base_path)
    fitter.run()

    for i in range(len(fitter.processed.Single_Spline_phase_settings)):
        fitter.plot_example(fitter.SS_curve_data_stored, fitter.SS_final_fitted_params, phase_index=i)

    # RO phase example
    fitter.plot_example(fitter.RO_curve_data_stored, fitter.RO_final_fitted_params, phase_index=0)
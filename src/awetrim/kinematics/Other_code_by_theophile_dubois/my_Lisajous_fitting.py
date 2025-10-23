from awetrim.kinematics.parametrized_patterns import CST_Lissajous
from awetrim.kinematics.Other_code_by_theophile_dubois.my_Lisajous_data_processing import Lisajous_data_processing
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

class Lisajous_fitting(Lisajous_data_processing, CST_Lissajous):
    def __init__(self, file_path_cycle=None, file_path_full=None, file_path_waypoint=None, cyc_idx=0):
        super().__init__(file_path_cycle=file_path_cycle, file_path_full=file_path_full, file_path_waypoints=file_path_waypoint, cyc_idx=cyc_idx)

        self.az_data = self.az_Lisajous
        self.el_data = self.el_Lisajous

    def LSQ(self):

        self.s = np.linspace(0, 2*np.pi, len(self.az_Lisajous))

        # fixed parameters
        fixed_params = {
            "omega": 1,
            "r0": 200,
            "kappa": 0.0,
            "kbeta": 0.0,
            "width_phi": 0.5,
            "width_beta": 0.5,
            "left_first": True,
            "normalize_bumps": False,
            "repeat_phi": False,
            "repeat_beta": False,
            "k_vr": 2716,
        }

        n_coeffs = 10  # number of Fourier coefficients for azimuth and elevation

        # parameters to optimize
        params_init = {
            "az_amp0": 0.34,
            "beta_amp0": 0.08,
            "beta0": 0.48,
            "beta_coeffs": list(np.random.uniform(-1, 1, n_coeffs)),
            "az_coeffs": list(np.random.uniform(-1, 1, n_coeffs)),
        }

        # flatten into 1D vector for least_squares
        x0 = np.concatenate([
            [params_init["az_amp0"]],
            [params_init["beta_amp0"]],
            [params_init["beta0"]],
            params_init["beta_coeffs"],
            params_init["az_coeffs"],
        ])

        # build bounds (example: restrict amplitudes to positive)
        lower_bounds = [0, 0, 0] + [-2]*n_coeffs + [-2]*n_coeffs
        upper_bounds = [2, 1, 1] + [ 2]*n_coeffs + [ 2]*n_coeffs

        def unpack_params(x):
            """Convert flat x back into dict for CST_Lissajous"""
            return {
                "az_amp0": x[0],
                "beta_amp0": x[1],
                "beta0": x[2],
                "beta_coeffs": x[3:11].tolist(),
                "az_coeffs": x[11:19].tolist(),
                **fixed_params,
            }

        def residual(x):
            params = unpack_params(x)
            obj = CST_Lissajous(**params)
            az_model = obj.azimuth(params["r0"], self.s)
            el_model = obj.elevation(params["r0"], self.s)

            res_az = self.az_data - az_model
            res_el = self.el_data - el_model
            return np.concatenate((res_az, res_el)).ravel()

        res = least_squares(
            residual,
            x0,
            bounds=(lower_bounds, upper_bounds),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            verbose=2
        )

        best_params = unpack_params(res.x)
        return res, best_params
    
    def plot_fitted_path(self, best_params):
        obj = CST_Lissajous(**best_params)

        r = np.full(len(self.az_Lisajous), 200) 

        az = obj.azimuth(r, self.s)
        el = obj.elevation(r, self.s)

        plt.figure()
        plt.plot(az, el)
        plt.plot(self.az_data, self.el_data, '--')
        plt.show()

if __name__ == "__main__":
    waypoint_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/waypoints/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-10-23_09-43-50_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"

    lsq_obj = Lisajous_fitting(file_path_cycle=cycle_path, file_path_full=full_path, file_path_waypoint=waypoint_path, cyc_idx=0)
    results, best_params = lsq_obj.LSQ()
    lsq_obj.plot_fitted_path(best_params)

    print("Best-fit parameters:", best_params)
    print("az_amp0:", best_params["az_amp0"])
    print("beta_amp0:", best_params["beta_amp0"])
    print("beta0:", best_params["beta0"])
    print("beta_coeffs:", best_params["beta_coeffs"])
    print("az_coeffs:", best_params["az_coeffs"])
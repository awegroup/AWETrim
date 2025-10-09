from picawe.kinematics.my_RI_fitting import RI_fitting as rifit
from picawe.kinematics.my_RI_RO_fitting import RI_RO_fitting as rirofit
from picawe.kinematics.my_RO_RI_fitting import RO_RI_fitting as rorifit
import pickle
import numpy as np

if __name__ == "__main__":
    ri_fitted = rifit(
        file_path_full="/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv",
        file_path_cycle="/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv",
        cyc_idx=0,
        p=3,
        n_ctrl=8,
        c_penalty=1.0,
        v_penalty=0.0,
        eps_knot=1e-3
        )

    # Store only what you actually need later
    results = {
        "C_sph": ri_fitted.C_sph,
        "crs0": ri_fitted.ri_crs0,
        "crsf": ri_fitted.ri_crsf,
        "phi0": ri_fitted.ri_p0_sph[0],
        "phif": ri_fitted.ri_pf_sph[0],
        "beta0": ri_fitted.ri_p0_sph[1],
        "betaf": ri_fitted.ri_pf_sph[1],
        "C_interior": ri_fitted.C_sph[1:-1],
        "u_vals": ri_fitted.u_vals,
        "U_interior": ri_fitted.U_sph[ri_fitted.p+1:-(ri_fitted.p+1)],
        "v0": ri_fitted.ri_v0,
    }

    # Save once to disk
    with open("fit_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Fitting complete. Results saved to fit_results.pkl \n")
    
    riro_fitted = rirofit(
        file_path_full="/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv",
        file_path_cycle="/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv",
        cyc_idx=0,
        p=3,
        n_ctrl=6,
        c_penalty=1.0,
        v_penalty=0.0,
        eps_knot=1e-3
        )
    
    rori_fitted = rorifit(
        file_path_full="/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv",
        file_path_cycle="/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv",
        cyc_idx=0,
        p=3,
        n_ctrl=6,
        c_penalty=1.0,
        v_penalty=0.0,
        eps_knot=1e-3
        )
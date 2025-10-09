from picawe.kinematics.my_RI_plotting import RI_plotting as plot

# -------------------------------
# Main script
# -------------------------------

if __name__ == "__main__":
    # --- File paths ---
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"

    cyc_idx = 0       # index of the cycle you want to process
    p = 3             # spline degree
    n_ctrl = 10        # number of control points
    c_penalty = 1     # course penalty (for spherical)
    v_penalty = 0     # velocity penalty (for cartesian)
    eps_knot = 0.01  # minimum knot spacing

    # Create the plotting object
    plotter = plot(full_path, cycle_path, cyc_idx,
                                    p, n_ctrl, c_penalty, v_penalty, eps_knot)
    
    plotter.plot_spline_fit_cart()
    plotter.plot_spline_fit_sph()
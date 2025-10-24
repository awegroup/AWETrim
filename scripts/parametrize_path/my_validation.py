import sys
import os

from my_reel_in import main as main_reel_in
from my_reelout_cst import main as main_reelout_cst
from my_cycle import main as main_cycle

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from src.awetrim.kinematics.compare import Compare
from src.awetrim.kinematics.my_FIT import Fitting
from src.awetrim.kinematics.my_Winch_and_Depower_FIT import WinchCurveFitter

source = ["JULIA", "EXPERIMENTAL"]

for data_source in source:

    if data_source == "EXPERIMENTAL":
        base_path = "./processed_data/experimental"
        waypoint_path = f"{base_path}/2024-11-05_12-58-54_ProtoLogger_waypoints.csv"
        full_path = f"{base_path}/2024-11-05_12-58-54_ProtoLogger.csv"
        cycle_path = f"{base_path}/2024-11-05_12-58-54_full_log.txt"
        csv_path = "./results/timeseries/cycle_timeseries.csv"
    elif data_source == "JULIA":
        base_path = "./processed_data/fitting"
        waypoint_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
        full_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger.csv"
        cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"
        csv_path = "./results/timeseries/cycle_timeseries.csv"

    depower_denominator = [0.28, 0.34] #, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20]

    for depower in depower_denominator:
        fit = Fitting(full_path, cycle_path, waypoint_path, cyc_idx=0, n_ctrl_pts=25)

        fit._setup_lissajous_segment()
        fit._setup_spline_segment()

        fit.FitSpline()
        fit.save_data_RI_spline()
        fit.plot_spline_cart()

        fit.FitLissajous()
        fit.save_data_L_shape()
        fit.plot_fit_L_shape()

        json_path = "src/awetrim/kinematics/pp_ws6-9_GS3_KCU4.A_KiteV9.60.A.json"
        base_path = "./processed_data/fitting"

        fitter = WinchCurveFitter(json_path, base_path)
        fitter.run()
        # Run the fitting scripts before this to generate the necessary data files
        main_reel_in(run_plots=False, depower_denom=depower)  # Run with default value
        main_reelout_cst(run_plots=False, depower_denom=depower)  # Run with default value
        main_cycle(run_plots=False)

        compare = Compare(full_path, cycle_path, waypoint_path, csv_path)
        compare._plot_all_data_overlayed()
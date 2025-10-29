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
    print("===================================================")
    print(f"PROCESSING DATA SOURCE: {data_source}")
    print("===================================================")

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

    depower_denominator = [0.36, 0.28, 0.20]

    cycle_idx = 0  # You can modify this index as needed

    run_plots_DP = True

    fit = Fitting(full_path, cycle_path, waypoint_path, cyc_idx=cycle_idx, n_ctrl_pts=25, run_plots=run_plots_DP)

    print("RO and RI_Spline fitting completed. Proceeding to Winch Curve Fitting...\n")

    json_path = "src/awetrim/kinematics/pp_ws6-9_GS3_KCU4.A_KiteV9.60.A.json"
    base_path = "./processed_data/fitting"
    fitter = WinchCurveFitter(json_path, base_path, cycle_idx=cycle_idx, n_knots=25, run_plots=False, run_plots_DP=False)

    print("===================================================================")
    print("Winch Curve Fitting completed. Proceeding to cycle simulation...")
    print("===================================================================")

    for depower in depower_denominator:

        print("\n")
        print("==================================================")
        print(f"Running for depower denominator: {depower}")
        print("===================================================\n")

        print("--------------------------------------------------")
        print("Starting Reel-In Simulation...")
        print("--------------------------------------------------")
        main_reel_in(run_plots=False, depower_denom=depower)  # Run with default value

        print("--------------------------------------------------")
        print("Starting Reel-Out Simulation...")
        print("--------------------------------------------------")
        main_reelout_cst(run_plots=False, depower_denom=depower)  # Run with default value

        print("--------------------------------------------------")
        print("Starting Full Cycle Data Concatenation...")
        print("--------------------------------------------------")
        main_cycle(run_plots=False)

        print("--------------------------------------------------")
        print("Starting Data Comparison...")
        print("--------------------------------------------------")
        compare = Compare(full_path, cycle_path, waypoint_path, csv_path, run_plots_DP=False)
        compare._plot_all_data_overlayed()
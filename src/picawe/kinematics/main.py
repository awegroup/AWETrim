from .ReelInBspline_build import ReelInBspline_build as ribbuild
from .ReelInBspline_data_processing import ReelInBspline_data_processing as ribdata
from .ReelInBspline_fitting import ReelInBspline_fitting as ribfit
from .ReelInBspline_plotting import ReelInBspline_plotting as ribplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# -------------------------------
# Main script
# -------------------------------

if __name__ == "__main__":
    # --- File paths ---
    full_df = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
    cycle_df = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"

    
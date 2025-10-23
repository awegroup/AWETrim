from awetrim.kinematics.my_data_processing_single_spline import DataProcessing
import matplotlib.pyplot as plt
import numpy as np

class Compare(DataProcessing):
    def __init__(self, full_path, cycle_path, waypoint_path):
        super().__init__(full_path, cycle_path, waypoint_path)

    def plot_CSV_data(self):

        # Example setup
        segments = ["RO", "Single_Spline"]
        variables = ["CL", "CD", "Mech_Power", "Vr", "Vtan", "tension_tether_ground"]

        # Plot loop
        for seg in segments:
            fig, axes = plt.subplots(len(variables), 1, figsize=(8, 12), sharex=True)
            fig.suptitle(f"{seg} Segment Results", fontsize=14, weight="bold")

            for ax, var in zip(axes, variables):
                y = getattr(self, f"{seg}_{var}")
                time = getattr(self, f"{seg}_time")
                ax.plot(time, y, label=f"{seg}_{var}")
                ax.set_ylabel(var)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="upper right", fontsize=8)

            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

    def plot_QS_data(self):
        # Placeholder for QS data plotting method
        pass

    def plot_overlayed_data(self):
        # Placeholder for overlayed data plotting method
        pass

if __name__ == "__main__":
    # File paths
    base_path = "./processed_data/fitting"
    waypoint_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"

    compare = Compare(full_path, cycle_path, waypoint_path)
    compare.plot_CSV_data()

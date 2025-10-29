from awetrim.kinematics.my_DP import DataProcessing
import importlib.util
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Compare(DataProcessing):
    def __init__(self, full_path, cycle_path, waypoint_path, csv_path, run_plots_DP=True):
        super().__init__(full_path, cycle_path, waypoint_path, cyc_idx=0, run_plots_DP=run_plots_DP)
        self.csv_path = Path(csv_path)
        self.df = None
        self.sim_type = "quasi_steady"
        # Load and process CSV automatically
        self._import_csv_data()
        self._set_attributes_from_csv()

    def _import_csv_data(self):
        """Read the CSV into a pandas DataFrame."""
        self.df = pd.read_csv(self.csv_path)
        print(
            f"✅ Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns"
        )

    def _set_attributes_from_csv(self):
        """Extract quasi-steady reel_in/out data and store as class attributes with consistent names."""
        if self.df is None:
            raise ValueError("CSV not loaded. Call _import_csv_data first.")

        # Filter only sim_type simulation
        df_qs = self.df[self.df["simulation"] == self.sim_type]

        segments_to_extract = ["reel_in", "reel_out"]

        # Mapping from CSV columns → DataProcessing variable names
        name_map = {
            "lift_coefficient": "CL",
            "drag_coefficient": "CD",
            "mechanical_power": "Mech_Power",
            "speed_radial": "Vr",
            "speed_tangential": "Vtan",
            "tension_tether_ground": "tension_tether_ground",
            "time": "time",
        }

        for segment in segments_to_extract:
            df_seg = df_qs[df_qs["segment"] == segment]

            if df_seg.empty:
                print(f"⚠️ No data found for segment: {segment}")
                continue

            for csv_col, var_name in name_map.items():
                values = df_seg[csv_col].values
                setattr(self, f"{segment}_{var_name}", values)

            print(
                f"✅ Stored attributes for segment '{segment}' with mapped variable names."
            )

    def _plot_all_data_overlayed(self):
        """Overlay results for both reel_in and reel_out."""
        overlay_pairs = [
            ("reel_in", "RI_Spline"),
            ("reel_out", "RO"),
        ]

        variables = [
            "Mech_Power",
            "Vr",
            "Vtan",
            "tension_tether_ground",
        ]

        for seg_qs, seg_ref in overlay_pairs:
            fig, axes = plt.subplots(len(variables), 1, figsize=(12, 12), sharex=True)
            fig.suptitle(
                f"Overlay: {seg_qs} {self.sim_type} vs {seg_ref} (reference)",
                fontsize=14,
                weight="bold",
            )

            t_qs = getattr(self, f"{seg_qs}_time")
            t_ref = getattr(self, f"{seg_ref}_time")

            t_diff = t_qs[0] - t_ref[0]
            print(f"Time offset for {seg_qs} vs {seg_ref}: {t_diff:.4f} s")

            if t_qs[0] < t_ref[0]:
                t_qs = t_qs - t_diff
                print(f"Adjusted t_qs start time to {t_qs[0]:.4f} s")
                print(f"t_ref start time remains {t_ref[0]:.4f} s")

            elif t_ref[0] < t_qs[0]:
                t_ref = t_ref + t_diff
                print(f"Adjusted t_ref start time to {t_ref[0]:.4f} s")
                print(f"t_qs start time remains {t_qs[0]:.4f} s")

            for ax, var in zip(axes, variables):
                # quasi steady data
                y_qs = getattr(self, f"{seg_qs}_{var}")

                # reference data from DataProcessing
                if var == "tension_tether_ground":
                    y_ref = (
                        getattr(self, f"{seg_ref}_{var}") * 9.81
                    )  # Convert from kgf to N
                else:
                    y_ref = getattr(self, f"{seg_ref}_{var}")

                ax.plot(t_qs, y_qs, label=f"{seg_qs} {self.sim_type}", linewidth=1.8)
                ax.plot(
                    t_ref,
                    y_ref,
                    label=f"{seg_ref} (reference)",
                    linestyle="--",
                    linewidth=1.3,
                )
                ax.set_ylabel(var)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="upper right", fontsize=8)

            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()


if __name__ == "__main__":
    # File paths
    base_path = "./processed_data/fitting"
    waypoint_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"
    csv_path = "./results/timeseries/cycle_timeseries.csv"

    compare = Compare(full_path, cycle_path, waypoint_path, csv_path)
    compare._plot_all_data_overlayed()

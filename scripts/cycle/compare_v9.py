import pandas as pd
import numpy as np
from scipy import signal

# Re-import necessary libraries after kernel reset
import pandas as pd
import numpy as np
import os
from scipy import signal
from awetrim.utils.utils import read_ekf_results


# Re-define the function after reset
def load_and_group_cycles(flight_data, delimiter=","):
    df = flight_data
    columns_to_extract = [
        "time",
        "kite_0_pitch",
        "kite_velocity_abs",
        "ground_tether_reelout_speed",
        "ground_tether_length",
        "ground_tether_force",
        "airspeed_angle_of_attack",
        "ground_mech_power",
        "kite_actual_depower",
        "kite_pos_east",
        "kite_pos_north",
        "kite_height",
        "kite_elevation",
        "kite_azimuth",
        "kite_distance",
        "airspeed_apparent_windspeed",
        "kite_estimated_va",
        "kite_measured_va",
        "kite_heading",
        "kite_course",
        "lift_coeff",
        "drag_coeff",
        "100m Wind Speed (m/s)",
    ]
    cycle_dfs = []

    def group_phases(index_vector):
        group_ids = np.full(len(index_vector), -1)
        current_group = 0
        in_group = False
        for i, val in enumerate(index_vector):
            if val == 1:
                if not in_group:
                    in_group = True
                    current_group += 1
                group_ids[i] = current_group
            else:
                in_group = False
        return group_ids

    def group_variable(var_vector, group_ids):
        grouped_var = []
        for gid in np.unique(group_ids):
            if gid > 0:
                grouped_var.append(var_vector[group_ids == gid])
        return grouped_var

    for col in df.columns:
        if "cycle" in col:
            print("cycle column found:", col)

    for cycle in range(0, int(df["cycle"].max()) + 1):

        cycle = df[df["cycle"] == cycle]

        cycle.reset_index(drop=True, inplace=True)
        cycle_dfs.append(cycle)

    return cycle_dfs


# File path used previously
results, flight_data, config_data = read_ekf_results(
    "2023", "11", "27", "v9", addition=""
)
cycle_dfs = load_and_group_cycles(flight_data, delimiter=" ")

print(f"Found {len(cycle_dfs)} cycles in the flight data.")


def find_min_RI_tether_length(df, threshold=2):
    avg_riro_depwr = np.mean(
        df.kite_actual_depower[df.flight_phase_index == 4].iloc[-5:-1]
    )
    dep_idx = (df.kite_actual_depower - avg_riro_depwr) > threshold
    dep_idx = dep_idx.iloc[::-1]
    start_pow_idx = len(dep_idx) - next(
        (i for i, x in enumerate(dep_idx) if x != 0), -1
    )
    min_tether_length_RI = df.ground_tether_length[start_pow_idx]
    return min_tether_length_RI, start_pow_idx


def find_RO_pattern_param(df_RO):
    df_RO = df_RO.reset_index(drop=True)

    def extract_complete_peaks(sig):
        peaks, _ = signal.find_peaks(sig, distance=10)
        valleys, _ = signal.find_peaks(-sig, distance=10)

        complete_peaks = []

        # Loop through valleys to find enclosed peaks
        for i in range(len(valleys) - 1):
            start = valleys[i]
            end = valleys[i + 1]
            enclosed_peaks = [p for p in peaks if start < p < end]
            if enclosed_peaks:
                tallest = max(enclosed_peaks, key=lambda p: sig[p])
                complete_peaks.append(tallest)

        # Optionally check for a final complete cycle after the last valley
        if len(valleys) >= 1:
            last_valley = valleys[-1]
            trailing_peaks = [p for p in peaks if p > last_valley]
            if trailing_peaks:
                tallest = max(trailing_peaks, key=lambda p: sig[p])
                complete_peaks.append(tallest)

        return complete_peaks

    peaks_idx, _ = signal.find_peaks(
        np.abs(df_RO.kite_azimuth), prominence=0.1, distance=10
    )
    max_az_trac = np.mean(np.abs(df_RO.kite_azimuth)[peaks_idx])

    peaks_idx_el = extract_complete_peaks(df_RO.kite_elevation)
    valleys_idx_el = extract_complete_peaks(-df_RO.kite_elevation)
    n_peaks = len(peaks_idx_el)
    if len(peaks_idx_el) == 0 or len(valleys_idx_el) == 0:
        peaks_idx_el, _ = signal.find_peaks(df_RO.kite_elevation, distance=10)
        valleys_idx_el, _ = signal.find_peaks(df_RO.kite_elevation, distance=10)
        raise (Exception)

    avg_el_peak = np.mean(df_RO.kite_elevation[peaks_idx_el])
    avg_el_valley = np.mean(df_RO.kite_elevation[valleys_idx_el])
    rel_el_angle = (
        0.5 * (avg_el_peak - avg_el_valley)
        if not np.isnan(avg_el_peak) and not np.isnan(avg_el_valley)
        else np.nan
    )
    avg_el_angle = 0.5 * (avg_el_peak + avg_el_valley)

    return max_az_trac, rel_el_angle, avg_el_angle, n_peaks


results = []
import matplotlib.pyplot as plt

valid_cycle_dfs = []
for df in cycle_dfs[2:10]:
    # df = df.reset_index(drop=True)
    # plt.plot(df.tether_reelout_speed, label="Kite Actual Depower")
    # plt.show()
    # try:
    max_az_trac, rel_el_angle, avg_el_angle, n_peaks = find_RO_pattern_param(
        df[df.kcu_actual_depower < 42]
    )
    min_tether_length_RO = df.tether_length.min()
    print("Min tether length RO:", min_tether_length_RO)
    print("Max tether length RO:", df.tether_length.max())
    avg_reel_speed_ro = np.mean(
        df[
            (df.kcu_actual_depower < 42) & (df.tether_reelout_speed > 0.7)
        ].tether_reelout_speed
    )
    avg_reel_speed_ri = np.mean(
        df[
            (df.kcu_actual_depower > 42) & (df.tether_reelout_speed < 0)
        ].tether_reelout_speed
    )
    deployed_tether_length = df.tether_length.max() - df.tether_length.min()
    # except Exception as e:
    #     print(f"Error processing cycle: {e}")
    #     # Skip this cycle if it fails
    #     print("Skipping this cycle due to error.")
    #     continue

    # plt.plot(df.kite_azimuth,df.kite_elevation, label="RO Azimuth vs Elevation")
    # plt.xlabel("Azimuth Angle (rad)")
    # plt.ylabel("Elevation Angle (rad)")
    # plt.show()
    ground_mech_power = df.ground_tether_force * df.tether_reelout_speed
    df["ground_mech_power"] = ground_mech_power

    energy_ro = np.sum(ground_mech_power[df.kcu_actual_depower < 42] * 0.1)
    energy_ri = np.sum(ground_mech_power[df.kcu_actual_depower > 42] * 0.1)
    print(
        f"Energy RO: {energy_ro:.2f} J, Energy RI: {energy_ri:.2f} J, Total: {energy_ro + energy_ri:.2f} J"
    )

    dt = np.diff(df.time, prepend=df.time[0])
    dt_ri = dt[df.kcu_actual_depower > 42]
    dt_ro = dt[df.kcu_actual_depower < 42]
    wind_speed_cols = [col for col in df.columns if col.endswith("m_Wind_Speed_m_s")]

    # Parse heights from column names (e.g., "200m_Wind_Speed_m_s" → 200.0)
    heights = [float(col.split("m_")[0]) for col in wind_speed_cols]

    # Compute average wind speed at each height
    mean_speeds = [np.mean(df[col]) for col in wind_speed_cols]

    # Convert to tabulated format
    tabulated_heights = list(heights)
    tabulated_speeds = list(mean_speeds)

    # Optional: sort by height
    tabulated_heights, tabulated_speeds = zip(
        *sorted(zip(tabulated_heights, tabulated_speeds))
    )
    tabulated_heights = list(tabulated_heights)
    tabulated_speeds = list(tabulated_speeds)
    results.append(
        {
            "RO_max_azimuth_rad": max_az_trac,
            "RO_rel_elevation_rad": rel_el_angle,
            "RO_avg_elevation_rad": avg_el_angle,
            "RO_min_tether_length_m": min_tether_length_RO,
            "RO_max_tether_length_m": df.tether_length.max(),
            "avg_reeling_speed_RO_mps": avg_reel_speed_ro,
            "avg_reeling_speed_RI_mps": avg_reel_speed_ri,
            "min_reeling_speed_RI_mps": np.min(
                df[df.kcu_actual_depower > 42].tether_reelout_speed
            ),
            "tabulated_heights": tabulated_heights,
            "tabulated_speeds": tabulated_speeds,
            "avg_mechanical_power": np.sum(df.ground_mech_power * 0.1)
            / (df.time.iloc[-1] - df.time.iloc[0]),
            "avg_mechanical_power_RO": np.mean(
                df[df.kcu_actual_depower < 42].ground_mech_power
            ),
            "avg_mechanical_power_RI": np.mean(
                df[df.kcu_actual_depower > 42].ground_mech_power
            ),
            "total_time_RO": np.sum(dt_ro),
            "total_time_RI": np.sum(dt_ri),
            "n_peaks_RO": n_peaks,
            "deployed_tether_length": deployed_tether_length,
        }
    )
    valid_cycle_dfs.append(df)
cycle_dfs = valid_cycle_dfs

df_results = pd.DataFrame(results)


import json
from awetrim import Cycle


# Function to simulate experimental cycles using stats extracted from flight logs
def simulate_cycles_from_stats(
    aero_input_path, sim_config, flight_stats, cycle_dfs=None
):
    with open(aero_input_path, "r") as file:
        aero_input = json.load(file)

    cycle_results = []
    for i, row in flight_stats.iterrows():
        az_max = row["RO_max_azimuth_rad"]
        rel_elevation = row["RO_rel_elevation_rad"]
        avg_elevation = row["RO_avg_elevation_rad"]
        r0 = row["RO_min_tether_length_m"]
        ry = r0 * np.sin(az_max)
        rz = r0 * np.sin(rel_elevation) * 2 / 0.8
        vr = row["avg_reeling_speed_RO_mps"]
        lt = row["deployed_tether_length"]
        pattern_config = {
            "pattern_type": "lissajous_angles",
            "parameters": {
                "omega": -1.0,
                "r0": r0,
                "az_amp0": az_max,
                "beta_amp0": rel_elevation * 2,
                "vr": vr,
                "beta0": avg_elevation,
                "kappa": 1,
            },
            "control": {
                "input_depower": 0.0,
            },
            "start_time": 0,
            "end_time": lt / vr,
            "n_points": 300,
            "quasi_steady": True,
        }
        # raise ValueError("Pattern config not implemented yet")
        # print("reeling speed RO:", row["avg_reeling_speed_RO_mps"])
        # print("min reeling speed RI:", row["min_reeling_speed_RI_mps"])
        CYCLE_SETTINGS = {
            "reelout": pattern_config,
            "reelin": {
                "control": {
                    "max_elevation": np.radians(100),
                    "min_elevation": np.radians(25),
                    "reeling_speed": row["avg_reeling_speed_RI_mps"],
                    "min_tether_force": sim_config["mass_wing"] * 9.81,
                    "length_tether_ro": pattern_config["parameters"]["r0"],
                    "ri_elevation": np.radians(40),  # Initial elevation for reeling in
                },
                "initial_state": {
                    "angle_course": 0,
                    "input_steering": 0,
                    "input_depower": 0,
                    "speed_tangential": 40,
                    "timeder_angle_course": 0,
                    "tension_tether_ground": 1e4,
                },
                "time_step": 0.1,
                "quasi_steady": True,
            },
        }
        # --- Energy calculations ---

        sim_config["tabulated_heights"] = row["tabulated_heights"]
        sim_config["tabulated_speeds"] = row["tabulated_speeds"]
        cycle_sim = Cycle(aero_input, sim_config)
        # dt_reelout = np.diff(phase_ro.return_variable("t"), prepend=0.0)
        # total_reelout_time = np.sum(dt_reelout)
        try:
            phase_ro, phase_ri = cycle_sim.run_cycle(CYCLE_SETTINGS)

            energy_ro = np.sum(
                phase_ro.return_variable("mechanical_power")
                * np.diff(
                    phase_ro.return_variable("t"),
                    prepend=phase_ro.return_variable("t")[0],
                )
            )
            energy_ri = np.sum(
                phase_ri.return_variable("mechanical_power")
                * np.diff(
                    phase_ri.return_variable("t"),
                    prepend=phase_ri.return_variable("t")[0],
                )
            )
            avg_power_ro = energy_ro / (
                phase_ro.return_variable("t")[-1] - phase_ro.return_variable("t")[0]
            )
            avg_power_ri = (
                (
                    energy_ri
                    / (
                        phase_ri.return_variable("t")[-1]
                        - phase_ri.return_variable("t")[0]
                    )
                )
                if phase_ri is not None
                else 0
            )
            avg_power = (energy_ro + energy_ri) / (
                (phase_ri.return_variable("t")[-1] - phase_ri.return_variable("t")[0])
                + (phase_ro.return_variable("t")[-1] - phase_ro.return_variable("t")[0])
            )
            results = {
                # Reelout (RO) phase
                "RO_max_azimuth_rad": phase_ro.return_variable("angle_azimuth")[-1],
                "RO_avg_elevation_rad": np.mean(
                    phase_ro.return_variable("angle_elevation")
                ),
                "RO_rel_elevation_rad": max(phase_ro.return_variable("angle_elevation"))
                - np.mean(phase_ro.return_variable("angle_elevation")),
                "RO_max_tether_length_m": phase_ro.return_variable("distance_radial")[
                    -1
                ],
                "avg_mechanical_power_RO": avg_power_ro,
                # Reelin (RI) phase
                "avg_mechanical_power_RI": avg_power_ri,
                # Both phases together
                "avg_mechanical_power": avg_power,
                "total_time_RO": (
                    phase_ro.return_variable("t")[-1]
                    if phase_ro is not None
                    else np.nan
                ),
                "total_time_RI": (
                    phase_ri.return_variable("t")[-1] - phase_ri.return_variable("t")[0]
                    if phase_ri is not None
                    else np.nan
                ),
            }
            if cycle_dfs is not None:
                time_ro = (
                    phase_ro.return_variable("t")
                    + phase_ri.return_variable("t")[-1]
                    - phase_ri.return_variable("t")[0]
                )
                time_ri = (
                    phase_ri.return_variable("t") - phase_ri.return_variable("t")[0]
                )
                plt.figure()
                plt.plot(
                    cycle_dfs[i]["time"] - cycle_dfs[i]["time"].iloc[0],
                    cycle_dfs[i]["ground_tether_force"],
                    label="Cycle DFS",
                )
                plt.plot(
                    time_ro,
                    phase_ro.return_variable("tension_tether_ground"),
                    label="Reelout Tension",
                )
                plt.plot(
                    time_ri,
                    phase_ri.return_variable("tension_tether_ground"),
                    label="Reelin Tension",
                )
                plt.xlabel("Wind Speed at 200m (m/s)")
                plt.ylabel("Mechanical Power (W)")
                plt.legend()

                plt.figure()
                plt.plot(
                    time_ri,
                    phase_ri.return_variable("speed_radial"),
                    label="Reelin Speed Radial",
                )
                plt.plot(
                    time_ro,
                    phase_ro.return_variable("speed_radial"),
                    label="Reelout Speed Radial",
                )
                plt.plot(
                    cycle_dfs[i]["time"] - cycle_dfs[i]["time"].iloc[0],
                    cycle_dfs[i]["tether_reelout_speed"],
                    label="Cycle DFS Reelout Speed",
                )
                plt.show()
            cycle_results.append(results)

        except Exception as e:
            print(f"Cycle {i+1} simulation failed: {e}")
            cycle_results.append(
                {
                    "RO_max_azimuth_rad": np.nan,
                    "RO_avg_elevation_rad": np.nan,
                    "RO_rel_elevation_rad": np.nan,
                    "RO_max_tether_length_m": np.nan,
                    "avg_mechanical_power_RO": np.nan,
                    "avg_mechanical_power_RI": np.nan,
                    "avg_mechanical_power": np.nan,
                    "total_time_RO": np.nan,
                    "total_time_RI": np.nan,
                }
            )

    return pd.DataFrame(cycle_results)


# Store the code in a callable function for later use
def run_flight_log_simulation(df_results, cycle_dfs=None):
    aero_input_path = "./data/LEI-V9-KITE/v9_aero_input.json"
    SIMULATION_CONFIG = {
        "dof": 3,
        "area_wing": 47,
        "mass_wing": 90,
        "mass_kcu": 0,  # Assuming no mass for KCU in this simulation
        "tether_diameter": 0.014,
        "quasi_steady": True,
        "wind_model": "tabulated",
        "steering_control": "roll",
    }
    return simulate_cycles_from_stats(aero_input_path, SIMULATION_CONFIG, df_results)


# Run the simulation with the extracted flight log statistics
cycle_results = run_flight_log_simulation(df_results, cycle_dfs)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["avg_mechanical_power"], label="Avg Mechanical Power")
plt.plot(
    df_results["avg_mechanical_power"],
    label="Flight Log Avg Mechanical Power",
    linestyle="--",
)
plt.xlabel("Cycle Index")
plt.ylabel("Mechanical Power (W)")
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["avg_mechanical_power_RO"], label="Avg Mechanical Power RO")
plt.plot(
    df_results["avg_mechanical_power_RO"],
    label="Flight Log Avg Mechanical Power RO",
    linestyle="--",
)
plt.plot(cycle_results["avg_mechanical_power_RI"], label="Avg Mechanical Power RI")
plt.plot(
    df_results["avg_mechanical_power_RI"],
    label="Flight Log Avg Mechanical Power RI",
    linestyle="--",
)
plt.xlabel("Cycle Index")
plt.ylabel("Mechanical Power (W)")
plt.legend()
# plt.show()

wind_speed_200m = []
for i, row in df_results.iterrows():
    idx_200_height = row["tabulated_heights"].index(200)
    wind_speed_200m.append(row["tabulated_speeds"][idx_200_height])


plt.figure(figsize=(12, 8))
plt.scatter(
    wind_speed_200m,
    cycle_results["avg_mechanical_power"],
    label="Avg Mechanical Power",
)
plt.scatter(
    wind_speed_200m,
    df_results["avg_mechanical_power"],
    label="Flight Log Avg Mechanical Power",
    linestyle="--",
)
plt.xlabel("Wind Speed at 200m (m/s)")
plt.legend()
plt.ylabel("Mechanical Power (W)")
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["RO_max_tether_length_m"], label="RO Max Tether Length (m)")
plt.plot(
    df_results["RO_max_tether_length_m"],
    label="Flight Log RO Max Tether Length (m)",
    linestyle="--",
)
plt.xlabel("Cycle Index")
plt.ylabel("Max Tether Length (m)")
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["total_time_RO"], label="RO Total Time (s)")
plt.plot(
    df_results["total_time_RO"], label="Flight Log RO Total Time (s)", linestyle="--"
)
plt.plot(cycle_results["total_time_RI"], label="RI Total Time (s)")
plt.plot(
    df_results["total_time_RI"], label="Flight Log RI Total Time (s)", linestyle="--"
)
plt.plot(
    cycle_results["total_time_RO"] + cycle_results["total_time_RI"],
    label="Total Time (s)",
)
plt.plot(
    df_results["total_time_RO"] + df_results["total_time_RI"],
    label="Flight Log Total Time (s)",
    linestyle="--",
)
plt.xlabel("Cycle Index")
plt.ylabel("Total Time (s)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["RO_rel_elevation_rad"], label="RO Relative Elevation (rad)")
plt.plot(
    df_results["RO_rel_elevation_rad"],
    label="Flight Log RO Relative Elevation (rad)",
    linestyle="--",
)
plt.xlabel("Cycle Index")
plt.ylabel("Relative Elevation (rad)")
plt.legend()
plt.show()
# import ace_tools as tools; tools.display_dataframe_to_user(name="Cycle Statistics Extracted", dataframe=df_results)

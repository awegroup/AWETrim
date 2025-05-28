import pandas as pd
import numpy as np
from scipy import signal

# Re-import necessary libraries after kernel reset
import pandas as pd
import numpy as np
import os
from scipy import signal

# Re-define the function after reset
def load_and_group_cycles(proto_logger_file, delimiter=","):
    df = pd.read_csv(proto_logger_file,delimiter= delimiter, low_memory=False)
    
    columns_to_extract = [
        'time', 'kite_0_pitch', 'kite_velocity_abs', 'ground_tether_reelout_speed', 'ground_tether_length',
        'ground_tether_force', 'airspeed_angle_of_attack', 'ground_mech_power', 'kite_actual_depower',
        'kite_pos_east', 'kite_pos_north', 'kite_height', 'kite_elevation', 'kite_azimuth', 'kite_distance',
        'airspeed_apparent_windspeed', 'kite_estimated_va', 'kite_measured_va', 'kite_heading', 'kite_course',
        'lift_coeff', 'drag_coeff', '100m Wind Speed (m/s)', 'flight_phase', 'flight_phase_index'
    ]
    df = df[df["flight_phase_index"].notna()]
    df = df[columns_to_extract]

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

    results = [[], [], [], []]
    flight_idx = range(1, 5)
    for i in flight_idx:
        idx = df.flight_phase_index == i
        group_ids = group_phases(idx)
        grouped_var = group_variable(df.time.to_numpy(), group_ids)
        for n in range(len(grouped_var)):
            results[i - 1].append(pd.DataFrame(grouped_var[n], columns=[df.columns[0]]))

        for var in df.columns[1:]:
            grouped_var = group_variable(df[var].to_numpy(), group_ids)
            for n in range(len(grouped_var)):
                results[i - 1][n][var] = pd.Series(grouped_var[n])

    cycle_dfs = []
    for n in range(len(results[0])):
        cycle = pd.concat([results[0][n], results[1][n], results[2][n], results[3][n]], axis=0, ignore_index=True)
        cycle.reset_index(drop=True, inplace=True)
        cycle_dfs.append(cycle)

    return cycle_dfs

# File path used previously
file_path = "data/flight_logs/2024-02-15_12-59-57_ProtoLogger_lidar.csv"
# file_path = "data/flight_logs/2024-06-05_11-33-16_ProtoLogger_lidar.csv"
cycle_dfs = load_and_group_cycles(file_path, delimiter=" ")



def find_min_RI_tether_length(df, threshold=2):
    avg_riro_depwr = np.mean(df.kite_actual_depower[df.flight_phase_index == 4].iloc[-5:-1])
    dep_idx = (df.kite_actual_depower -  avg_riro_depwr) > threshold
    dep_idx = dep_idx.iloc[::-1]
    start_pow_idx = len(dep_idx) - next((i for i, x in enumerate(dep_idx) if x != 0), -1)
    min_tether_length_RI = df.ground_tether_length[start_pow_idx]
    return min_tether_length_RI, start_pow_idx

def find_RO_pattern_param(df_RO):
    def extract_complete_peaks(sig):    
        peaks, _ = signal.find_peaks(sig, distance = 5)
        valleys, _ = signal.find_peaks(-sig, distance = 5)

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

    peaks_idx, _ = signal.find_peaks(np.abs(df_RO.kite_azimuth), prominence=0.1, distance=10)    
    max_az_trac = np.mean(np.abs(df_RO.kite_azimuth)[peaks_idx]) 

    peaks_idx_el = extract_complete_peaks(df_RO.kite_elevation)
    valleys_idx_el = extract_complete_peaks(-df_RO.kite_elevation)
    n_peaks = len(peaks_idx_el)
    if len(peaks_idx_el) == 0 or len(valleys_idx_el) == 0:
        peaks_idx_el, _ = signal.find_peaks(df_RO.kite_elevation,  distance=10)
        valleys_idx_el, _ = signal.find_peaks(df_RO.kite_elevation, distance=10)
        raise(Exception)
    
    avg_el_peak = np.mean(df_RO.kite_elevation[peaks_idx_el])
    avg_el_valley = np.mean(df_RO.kite_elevation[valleys_idx_el])
    rel_el_angle = 0.5*(avg_el_peak - avg_el_valley) if not np.isnan(avg_el_peak) and not np.isnan(avg_el_valley) else np.nan
    avg_el_angle = 0.5*(avg_el_peak + avg_el_valley)

    return max_az_trac, rel_el_angle, avg_el_angle, n_peaks

results = []
import matplotlib.pyplot as plt
valid_cycle_dfs = []
for df in cycle_dfs:
    # df = df.reset_index(drop=True)
    try:
        max_az_trac, rel_el_angle, avg_el_angle, n_peaks = find_RO_pattern_param(df[df.flight_phase_index == 1])
        min_tether_length_RO = df.ground_tether_length.min()
        avg_reel_speed_ro = np.mean(df[df.flight_phase_index == 1].ground_tether_reelout_speed)
        avg_reel_speed_ri = np.mean(df[df.flight_phase_index == 3].ground_tether_reelout_speed)
    except Exception as e:
        print(f"Error processing cycle: {e}")
        # Skip this cycle if it fails
        print("Skipping this cycle due to error.")
        continue

    # plt.plot(df.kite_azimuth,df.kite_elevation, label="RO Azimuth vs Elevation")
    # plt.xlabel("Azimuth Angle (rad)")
    # plt.ylabel("Elevation Angle (rad)")
    # plt.show()
    results.append({
        "RO_max_azimuth_rad": max_az_trac,
        "RO_rel_elevation_rad": rel_el_angle,
        "RO_avg_elevation_rad": avg_el_angle,
        "RO_min_tether_length_m": min_tether_length_RO,
        "RO_max_tether_length_m": df.ground_tether_length.max(),
        "avg_reeling_speed_RO_mps": avg_reel_speed_ro,
        "avg_reeling_speed_RI_mps": avg_reel_speed_ri,
        "min_reeling_speed_RI_mps": np.min(df[df.flight_phase_index == 3].ground_tether_reelout_speed),
        'wind_speed_100m_mps': np.mean(df['100m Wind Speed (m/s)']),
        "avg_mechanical_power": np.mean(df.ground_mech_power),
        "avg_mechanical_power_RO": np.mean(df[df.flight_phase_index == 1].ground_mech_power),
        "avg_mechanical_power_RI": np.mean(df[df.flight_phase_index.isin([2, 3, 4])].ground_mech_power),
        "total_time_RO": df[df.flight_phase_index == 1].time.max()- df[df.flight_phase_index == 1].time.min(),
        "total_time_RI": df[df.flight_phase_index.isin([2, 3, 4])].time.max()- df[df.flight_phase_index.isin([2, 3, 4])].time.min(),
        "n_peaks_RO": n_peaks,
    })
    valid_cycle_dfs.append(df)
cycle_dfs = valid_cycle_dfs

df_results = pd.DataFrame(results)



import json
from picawe import Cycle

# Function to simulate experimental cycles using stats extracted from flight logs
def simulate_cycles_from_stats(aero_input_path, sim_config, flight_stats):
    with open(aero_input_path, "r") as file:
        aero_input = json.load(file)


    cycle_results = []
    for i, row in flight_stats.iterrows():
        az_max = row["RO_max_azimuth_rad"]
        rel_elevation = row["RO_rel_elevation_rad"]
        avg_elevation = row["RO_avg_elevation_rad"]
        print(row["n_peaks_RO"], "peaks in RO cycle")
        r0 = row["RO_min_tether_length_m"]
        ry = r0 * np.sin(az_max)
        rz = r0 * np.sin(rel_elevation)*2/0.8
        vr = row["avg_reeling_speed_RO_mps"]
        
    
        pattern_config = {
            "pattern_type": "figure_eight",
            "parameters": {
                "omega": -1.0,
                "r0": r0,
                "ry": ry,
                "rz": rz,
                "ky": 0.7,
                "kz": 0.7,
                "vr": vr,
                "beta0": avg_elevation,
                "kappa": 1,
            },
            "control": {
                "input_depower": 0.0,
            },
            "start_path_angle": -np.pi / 2,
            "end_path_angle": np.pi / 2 + 2*np.pi,
            "n_points": 200,
        }
        print(pattern_config)
        # raise ValueError("Pattern config not implemented yet")
        # print("reeling speed RO:", row["avg_reeling_speed_RO_mps"])
        print("avg reeling speed RI:", row["avg_reeling_speed_RI_mps"])
        # print("min reeling speed RI:", row["min_reeling_speed_RI_mps"])
        CYCLE_SETTINGS = {
            "reelout": pattern_config,
            "reelin": {
                "control": {
                    "max_elevation": np.radians(85),
                    "min_elevation": np.radians(25),
                    "reeling_speed": row["avg_reeling_speed_RI_mps"],
                    "min_tether_force": sim_config["mass_wing"] * 9.81,
                    "length_tether_ro": r0,
                },
                "initial_state": {
                    "angle_course": 0,
                    "input_steering": 0,
                    "input_depower": 0,
                    "speed_tangential": 40,
                    "timeder_angle_course": 0,
                    "tension_tether_ground": 1e4,
                },
                "time_step": 0.1
            }
        }
# --- Energy calculations ---
        
        speed_friction = row["wind_speed_100m_mps"] *0.4 / np.log(100 / sim_config["z0"])
        # print(f"Wind speed at 100m: {row['wind_speed_100m_mps']} m/s")
        print(f"Wind speed at 100m: {speed_friction} m/s")
        sim_config["speed_friction"] = speed_friction
        cycle_sim = Cycle(aero_input, sim_config)
        # dt_reelout = np.diff(phase_ro.return_variable("t"), prepend=0.0)
        # total_reelout_time = np.sum(dt_reelout)
        try:
            phase_ro, phase_ri = cycle_sim.run_cycle(CYCLE_SETTINGS)

            results = {
                # Reelout (RO) phase
                "RO_max_azimuth_rad": phase_ro.return_variable("angle_azimuth")[-1],
                "RO_avg_elevation_rad": np.mean(phase_ro.return_variable("angle_elevation")),
                "RO_rel_elevation_rad": max(phase_ro.return_variable("angle_elevation")) - np.mean(phase_ro.return_variable("angle_elevation")),
                "RO_max_tether_length_m": phase_ro.return_variable("distance_radial")[-1],
                "avg_mechanical_power_RO": np.mean(phase_ro.return_variable("mechanical_power")),
                # Reelin (RI) phase
                "avg_mechanical_power_RI": np.mean(phase_ri.return_variable("mechanical_power")) if phase_ri is not None else np.nan,
                # Both phases together
                "avg_mechanical_power": np.mean(
                    np.concatenate([
                        phase_ro.return_variable("mechanical_power"),
                        phase_ri.return_variable("mechanical_power") if phase_ri is not None else np.array([])
                    ])
                ),
                "total_time_RO": phase_ro.return_variable("t")[-1] if phase_ro is not None else np.nan,
                "total_time_RI": phase_ri.return_variable("t")[-1]-phase_ri.return_variable("t")[0] if phase_ri is not None else np.nan,
            }
            cycle_results.append(results)
        except Exception as e:
            print(f"Cycle {i+1} simulation failed: {e}")
            cycle_results.append({
                "RO_max_azimuth_rad": np.nan,
                "RO_avg_elevation_rad": np.nan,
                "RO_rel_elevation_rad": np.nan,
                "RO_max_tether_length_m": np.nan,
                "avg_mechanical_power_RO": np.nan,
                "avg_mechanical_power_RI": np.nan,
                "avg_mechanical_power": np.nan,
                "total_time_RO": np.nan,
                "total_time_RI": np.nan,
            })

    return pd.DataFrame(cycle_results)

# Store the code in a callable function for later use
def run_flight_log_simulation(df_results):
    aero_input_path = "./data/v9_aero_input.json"
    SIMULATION_CONFIG = {
        "dof": 3,
        "area_wing": 47,
        "mass_wing": 78,
        "mass_kcu": 0,  # Assuming no mass for KCU in this simulation
        "tether_diameter": 0.014,
        "quasi_steady": True,
        "wind_model": "logarithmic",
        "speed_friction": 0.45,
        "z0": 0.01,
        "steering_control": "roll",
    }
    return simulate_cycles_from_stats(aero_input_path, SIMULATION_CONFIG, df_results)



# Run the simulation with the extracted flight log statistics
cycle_results = run_flight_log_simulation(df_results)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.plot(cycle_results["avg_mechanical_power"], label="Avg Mechanical Power")
plt.plot(df_results["avg_mechanical_power"], label="Flight Log Avg Mechanical Power", linestyle='--')
plt.xlabel("Cycle Index")
plt.ylabel("Mechanical Power (W)")
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["avg_mechanical_power_RO"], label="Avg Mechanical Power RO")
plt.plot(df_results["avg_mechanical_power_RO"], label="Flight Log Avg Mechanical Power RO", linestyle='--')
plt.plot(cycle_results["avg_mechanical_power_RI"], label="Avg Mechanical Power RI")
plt.plot(df_results["avg_mechanical_power_RI"], label="Flight Log Avg Mechanical Power RI", linestyle='--')
plt.xlabel("Cycle Index")
plt.ylabel("Mechanical Power (W)")
plt.legend()
# plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(df_results["wind_speed_100m_mps"], cycle_results["avg_mechanical_power"], label="Avg Mechanical Power")
plt.scatter(df_results["wind_speed_100m_mps"], df_results["avg_mechanical_power"], label="Flight Log Avg Mechanical Power", linestyle='--')
plt.xlabel("Wind Speed at 100m (m/s)")
plt.legend()
plt.ylabel("Mechanical Power (W)")
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["RO_max_tether_length_m"], label="RO Max Tether Length (m)")
plt.plot(df_results["RO_max_tether_length_m"], label="Flight Log RO Max Tether Length (m)", linestyle='--')
plt.xlabel("Cycle Index")
plt.ylabel("Max Tether Length (m)")
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["total_time_RO"], label="RO Total Time (s)")
plt.plot(df_results["total_time_RO"], label="Flight Log RO Total Time (s)", linestyle='--')
plt.plot(cycle_results["total_time_RI"], label="RI Total Time (s)")
plt.plot(df_results["total_time_RI"], label="Flight Log RI Total Time (s)", linestyle='--')
plt.xlabel("Cycle Index")
plt.ylabel("Total Time (s)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(cycle_results["RO_rel_elevation_rad"], label="RO Relative Elevation (rad)")
plt.plot(df_results["RO_rel_elevation_rad"], label="Flight Log RO Relative Elevation (rad)", linestyle='--')
plt.xlabel("Cycle Index")
plt.ylabel("Relative Elevation (rad)")
plt.legend()
plt.show()
# import ace_tools as tools; tools.display_dataframe_to_user(name="Cycle Statistics Extracted", dataframe=df_results)

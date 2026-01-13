import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from awetrim.kinematics.parametrized_patterns import CasadiSpline, CST_Lissajous
from scipy.optimize import least_squares
from pathlib import Path
from awetrim.system.system_model import create_system_model_from_yaml
from awetrim.timeseries.reelout_phase import Reelout

# Read the CSV file
csv_path = r"./data/LEI-V3-KITE/flight_logs/20191008_0065.csv"
df = pd.read_csv(csv_path)

# Filter data by flight_phase == 1
df_phase = df[df["flight_phase_index"] == 1]

# Define loop indices
first_loop_idxs = [120, 330]
second_loop_idxs = [330, 540]
third_loop_idxs = [540, 790]

# Extract individual loops
loop1 = df_phase.iloc[first_loop_idxs[0] : first_loop_idxs[1]]
loop2 = df_phase.iloc[second_loop_idxs[0] : second_loop_idxs[1]]
loop3 = df_phase.iloc[third_loop_idxs[0] : third_loop_idxs[1]]

# For individual loop fitting, choose one:
# df_filtered = loop1
# df_filtered = loop2
# df_filtered = loop3

# For simultaneous fitting of all three loops:
loops = [loop1, loop2, loop3]

# Create figure with azimuth vs elevation scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot all three loops
for i, loop in enumerate(loops, 1):
    ax.plot(
        loop["kite_azimuth"],
        loop["kite_elevation"],
        label=f"Loop {i}",
        alpha=0.7,
    )

ax.set_xlabel("Kite Azimuth (degrees)", fontsize=12)
ax.set_ylabel("Kite Elevation (degrees)", fontsize=12)
ax.legend()


# plt.tight_layout()
# plt.show()


"""Run least-squares Lissajous fitting."""
print("Starting Lissajous fitting...")
fixed_params = {
    "omega": 1,
    "r0": df_phase["kite_distance"].iloc[0],  # Use mean of all phase data
    "kappa": 0.0,
    "kbeta": 0.0,
    "width_phi": 0.5,
    "width_beta": 0.5,
    "left_first": False,
    "normalize_bumps": False,
    "repeat_phi": False,
    "repeat_beta": False,
    "k_vr": 2716,
    "az_amp0": 0.2,
    "beta_amp0": 0.1,
}
n_coeffs = 10
params_init = {
    # "az_amp0": 0.5,
    # "beta_amp0": 0.08,
    "beta0": 0.48,
    "beta_coeffs": list(np.random.uniform(-1, 1, n_coeffs)),
    "az_coeffs": list(np.random.uniform(-1, 1, n_coeffs)),
}
x0 = np.concatenate(
    [
        # [params_init["az_amp0"]],
        # [params_init["beta_amp0"]],
        [params_init["beta0"]],
        params_init["beta_coeffs"],
        params_init["az_coeffs"],
    ]
)
lower_bounds = [0] + [-1] * n_coeffs + [-1] * n_coeffs
upper_bounds = [1] + [1] * n_coeffs + [1] * n_coeffs


def unpack_params(x):
    return {
        "az_amp0": x[0],
        "beta_amp0": x[1],
        "beta0": x[2],
        "beta_coeffs": x[3 : 3 + n_coeffs].tolist(),
        "az_coeffs": x[3 + n_coeffs :].tolist(),
        **fixed_params,
    }


def residual_all_loops(x):
    """Residual function that fits all three loops simultaneously."""
    params = unpack_params(x)
    obj = CST_Lissajous(**params)

    residuals = []
    for loop in loops:
        # Create normalized s parameter for this loop
        s_loop = np.linspace(0, 2 * np.pi, len(loop))

        # Compute model predictions
        az_model = obj.azimuth(params["r0"], s_loop)
        el_model = obj.elevation(params["r0"], s_loop)

        # Compute residuals for this loop
        az_residual = loop["kite_azimuth"].values - np.array(az_model).flatten()
        el_residual = loop["kite_elevation"].values - np.array(el_model).flatten()

        residuals.append(az_residual)
        residuals.append(el_residual)

    return np.concatenate(residuals)


res = least_squares(
    residual_all_loops,  # Fit all three loops simultaneously
    x0,
    bounds=(lower_bounds, upper_bounds),
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    verbose=2,
)

best_params = unpack_params(res.x)
print("\n" + "=" * 60)
print("COMBINED FIT - Best Lissajous parameters found:")
print("=" * 60)
for key, val in best_params.items():
    if isinstance(val, list):
        print(f"{key}: {val[:3]}... (length {len(val)})")
    else:
        print(f"{key}: {val}")

# Plot fitted curves for each loop using combined fit
obj = CST_Lissajous(**best_params)
for i, loop in enumerate(loops, 1):
    s_loop = np.linspace(0, 2 * np.pi, len(loop))
    L_shape_az_fit = obj.azimuth(best_params["r0"], s_loop)
    L_shape_el_fit = obj.elevation(best_params["r0"], s_loop)

    plt.plot(
        np.array(L_shape_az_fit).flatten(),
        np.array(L_shape_el_fit).flatten(),
        label=f"Combined Fit Loop {i}",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )

print("✅ Combined Lissajous fitting completed.")

# Now fit each loop individually
print("\n" + "=" * 60)
print("INDIVIDUAL FITS")
print("=" * 60)

individual_params = []
colors = ["red", "green", "purple"]

for i, loop in enumerate(loops, 1):
    print(f"\nFitting Loop {i} individually...")

    # Define residual function for this specific loop
    def residual_single_loop(x, loop_data=loop):
        params = unpack_params(x)
        obj_temp = CST_Lissajous(**params)
        s_temp = np.linspace(0, 2 * np.pi, len(loop_data))

        az_model = obj_temp.azimuth(params["r0"], s_temp)
        el_model = obj_temp.elevation(params["r0"], s_temp)

        az_residual = loop_data["kite_azimuth"].values - np.array(az_model).flatten()
        el_residual = loop_data["kite_elevation"].values - np.array(el_model).flatten()

        return np.concatenate([az_residual, el_residual])

    # Fit this loop
    res_individual = least_squares(
        residual_single_loop,
        x0,
        bounds=(lower_bounds, upper_bounds),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        verbose=0,
    )

    params_individual = unpack_params(res_individual.x)
    individual_params.append(params_individual)

    print(f"Loop {i} - Cost: {res_individual.cost:.4f}")

    # Plot individual fit
    obj_individual = CST_Lissajous(**params_individual)
    s_loop = np.linspace(0, 2 * np.pi, len(loop))
    az_fit_individual = obj_individual.azimuth(params_individual["r0"], s_loop)
    el_fit_individual = obj_individual.elevation(params_individual["r0"], s_loop)

    plt.plot(
        np.array(az_fit_individual).flatten(),
        np.array(el_fit_individual).flatten(),
        label=f"Individual Fit Loop {i}",
        linestyle=":",
        linewidth=2.5,
        color=colors[i - 1],
        alpha=0.9,
    )

print("\n✅ Individual Lissajous fitting completed.")
# print("\nCombined fit cost: {:.4f}".format(res.cost))
# for i, params in enumerate(individual_params, 1):
#     # Calculate individual cost for comparison
#     print(f"Loop {i} individual fit - extracted from optimization")

plt.legend(loc="best", fontsize=9)
plt.title("Three Loops: Data + Combined Fit + Individual Fits")
plt.grid(True, alpha=0.3)
plt.show()


import h5py
import pandas as pd
import numpy as np
from awetrim import SystemModel
from awetrim.system.kite import Kite
from awetrim.system.tether import (
    RigidLumpedTether,
    FlexibleLumpedTether,
    RigidLinkTether,
)
from awetrim.environment.Wind import Wind
import casadi as ca
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "/flight_logs/"
    date = str(year) + "-" + str(month) + "-" + str(day)
    file_name = str(kite_model) + "_" + date
    hdf5_path = path_to_main + path + file_name + addition + ".h5"
    ekf_output_df, flight_data_df, config_data = read_results_from_hdf5(hdf5_path)
    return ekf_output_df, flight_data_df, config_data


def read_results_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as hf:
        # Read the ekf_output_df DataFrame
        ekf_group = hf["ekf_output"]
        ekf_output_df = pd.DataFrame(
            {
                col: (
                    ekf_group[col][:].astype(str)
                    if ekf_group[col].dtype.kind == "S"
                    else ekf_group[col][:]
                )
                for col in ekf_group.keys()
            }
        )

        # Read the flight_data DataFrame
        flight_group = hf["flight_data"]
        flight_data_df = pd.DataFrame(
            {
                col: (
                    flight_group[col][:].astype(str)
                    if flight_group[col].dtype.kind == "S"
                    else flight_group[col][:]
                )
                for col in flight_group
                if isinstance(flight_group[col], h5py.Dataset)
            }
        )

        # Read config_data
        config_group = hf["config_data"]
        config_data = read_dict_from_group(config_group)

    return ekf_output_df, flight_data_df, config_data


def read_dict_from_group(group):
    config_dict = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")  # Decode byte strings back to regular strings
        config_dict[key] = value

    for subgroup_name in group:
        subgroup = group[subgroup_name]
        config_dict[subgroup_name] = read_dict_from_group(subgroup)

    return config_dict


results, flight_data, config_data = read_results(
    "2019", "10", "08", "v3", addition="", path_to_main="./data/LEI-V3-KITE/"
)
print(max(flight_data.cycle))
# mask = (flight_data.cycle>10)&(flight_data.cycle<70)
mask = flight_data.cycle.isin(range(10, 120))
# mask = flight_data.cycle.isin(range(64, 68))
# mask = flight_data.cycle == 65
# mask = mask & (flight_data.kite_elevation < 0.75)
flight_data = flight_data[mask]
results = results[mask]
results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

for col in flight_data.columns:
    if "power" in col:
        print(col)

# Extract time ranges from the original loops
loop_time_ranges = []
for i, loop in enumerate(loops, 1):
    time_start = loop["time"].iloc[0]
    time_end = loop["time"].iloc[-1]
    loop_time_ranges.append((time_start, time_end))
    print(f"Loop {i} time range: {time_start:.2f} - {time_end:.2f}s")

# Extract the same loops from flight_data by matching time ranges
print("\n" + "=" * 60)
print("Extracting loops from flight_data by matching time ranges")
print("=" * 60)

flight_loops = []
results_loops = []
for i, (t_start, t_end) in enumerate(loop_time_ranges, 1):
    # Find rows in flight_data where time is within this loop's time range

    mask = (flight_data["unix_time"] >= t_start) & (flight_data["unix_time"] <= t_end)
    loop_flight = flight_data[mask].copy()
    loop_result = results[mask].copy()
    results_loops.append(loop_result)
    flight_loops.append(loop_flight)

    print(
        f"Flight Loop {i}: {len(loop_flight)} points, time range: {t_start:.2f} - {t_end:.2f}s"
    )
    print(
        f'Mean tether reelout speed: {loop_flight["tether_reelout_speed"].mean():.2f} m/s'
    )
    print(
        "Mean mech power reelout:",
        (
            loop_flight["ground_tether_force"] * loop_flight["tether_reelout_speed"]
        ).mean(),
    )

print(
    f"\n✅ Extracted {len(flight_loops)} loops from flight_data matching original time ranges"
)

# Extract the same loops from results dataframe by matching time ranges
print("\n" + "=" * 60)
print("Extracting loops from results by matching time ranges")
print("=" * 60)

# Summary of wind speeds
print("\n" + "=" * 60)
print("WIND SPEED SUMMARY")
print("=" * 60)
for i, loop_results in enumerate(results_loops, 1):
    mean_ws = np.mean(loop_results["wind_speed_horizontal"])
    std_ws = np.std(loop_results["wind_speed_horizontal"])
    min_ws = np.min(loop_results["wind_speed_horizontal"])
    max_ws = np.max(loop_results["wind_speed_horizontal"])
    print(f"Loop {i}:")
    print(f"  Mean: {mean_ws:.2f} m/s")
    print(f"  Std:  {std_ws:.2f} m/s")
    print(f"  Min:  {min_ws:.2f} m/s")
    print(f"  Max:  {max_ws:.2f} m/s")


# Wind profile from 10 minutes before last loop ends
print("\n" + "=" * 60)
print("WIND PROFILE ANALYSIS")
print("=" * 60)

# Get last timestamp of the last loop
last_loop_results = results_loops[-1]
last_timestamp = last_loop_results["time"].iloc[-1]
ten_minutes_before = last_timestamp - 300  # 600 seconds = 10 minutes

print(f"Last timestamp: {last_timestamp:.2f}s")
print(f"10 minutes before: {ten_minutes_before:.2f}s")

# Extract data from 10 minutes before last timestamp
mask_profile = (results["time"] >= ten_minutes_before) & (
    results["time"] <= last_timestamp
)
wind_profile_data = results[mask_profile].copy()

print(f"Wind profile data points: {len(wind_profile_data)}")

# Create height bins (50m increments)
height_col = "kite_position_z"

min_height = wind_profile_data[height_col].min()
max_height = wind_profile_data[height_col].max()
bin_size = 20  # 50m bins

print(f"Height range: {min_height:.2f}m - {max_height:.2f}m")

# Create bins
bins = np.arange(
    np.floor(min_height / bin_size) * bin_size,
    np.ceil(max_height / bin_size) * bin_size + bin_size,
    bin_size,
)

# Assign each point to a bin and calculate mean wind speed
wind_profile_data["height_bin"] = pd.cut(wind_profile_data[height_col], bins=bins)

# Group by bin and calculate mean wind speed
wind_profile = wind_profile_data.groupby("height_bin", observed=True)[
    "wind_speed_horizontal"
].agg(["mean", "std", "count"])

print("\n" + "-" * 60)
print("WIND PROFILE (50m bins)")
print("-" * 60)
print(wind_profile)

# Plot wind profile
fig, ax = plt.subplots(figsize=(8, 10))

# Get bin centers
bin_centers = [(interval.left + interval.right) / 2 for interval in wind_profile.index]
mean_speeds = wind_profile["mean"].values
std_speeds = wind_profile["std"].values

# Plot with error bars
ax.errorbar(
    mean_speeds,
    bin_centers,
    xerr=std_speeds,
    fmt="o-",
    linewidth=2,
    markersize=8,
    capsize=5,
    capthick=2,
    label="Mean ± Std",
)

ax.set_xlabel("Wind Speed Horizontal (m/s)", fontsize=12)
ax.set_ylabel("Height (m)", fontsize=12)
ax.set_title(f"Wind Profile (10 min before t={last_timestamp:.1f}s)", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

print("\n✅ Wind profile analysis completed.")


KITE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_kite_input.yaml")
WIND_CONFIG = {
    "speed_wind_at_100": 10,
    "z0": 0.5,
    "model_type": "uniform",
}
WIND_CONFIG = {
    "model_type": "tabulated",
    "tabulated_heights": bin_centers,
    "tabulated_speeds": mean_speeds,
}
RADIAL_PARAMETERS = {
    "reeling_strategy": "force",
    "force_model": "linear",
    "reeling_speed": 1.17,
    "max_tether_force": 8400,
    "min_tether_force": 750,
    "softplus": True,
    "softplus_beta": 0.01,
    "softminus": True,
    "softminus_beta": 0.001,
    "slope_winch_ro": 5555.55,
    "offset_winch_ro": 0.58,
}

REELOUT_CONFIG = {
    "pattern_type": "cst_lissajous",
    "path_parameters": individual_params[0],  # Use first loop's fitted params
    "radial_parameters": RADIAL_PARAMETERS,
    "sim_parameters": {
        "start_time": 0,
        "end_time": 35,
        "start_angle": -np.pi / 2,
        "end_angle": 2 * np.pi - np.pi / 2,
        "n_points": 300,
        "input_depower": 0.0,
    },
}


def build_wind_model(WIND_CONFIG):
    """Create a wind model using the supplied parameters."""

    if WIND_CONFIG["model_type"] == "logarithmic":
        wind_model = Wind(
            wind_model=WIND_CONFIG["model_type"],
            z0=WIND_CONFIG.get("z0", 0.01),
        )
        speed_friction = (
            0.41 * WIND_CONFIG["speed_wind_at_100"] / np.log(100 / wind_model.z0)
        )
        wind_model.speed_friction = speed_friction
    elif WIND_CONFIG["model_type"] == "uniform":
        wind_model = Wind(
            wind_model=WIND_CONFIG["model_type"],
            z0=WIND_CONFIG.get("z0", 0.01),
        )
        speed_friction = (
            0.41 * WIND_CONFIG["speed_wind_at_100"] / np.log(100 / wind_model.z0)
        )
        wind_model.speed_friction = speed_friction
        wind_model.speed_wind_ref = WIND_CONFIG["speed_wind_at_100"]
    elif WIND_CONFIG["model_type"] == "tabulated":
        wind_model = Wind(
            wind_model="tabulated",
            z0=WIND_CONFIG.get("z0", 0.01),
            tabulated_heights=WIND_CONFIG["tabulated_heights"],
            tabulated_speeds=WIND_CONFIG["tabulated_speeds"],
        )
    return wind_model


system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)

wind_model = build_wind_model(WIND_CONFIG)
system_model.wind = wind_model
reelout = Reelout(
    system_model=system_model,
    pattern_config=REELOUT_CONFIG,
)

phase, _ = reelout.run_simulation(run_plots=True, s_dot=5)
plt.show()

print(phase.energy_metrics())


path_parameters_start_opti = {
    "r0": df_phase["kite_distance"].iloc[0],
    "az_amp0": 0.25,
    "beta_amp0": 0.1,
    "beta0": 0.5,
    "beta_coeffs": np.zeros(5),
    "az_coeffs": np.zeros(5),
}
REELOUT_CONFIG["path_parameters"] = path_parameters_start_opti
reelout = Reelout(
    system_model=system_model,
    pattern_config=REELOUT_CONFIG,
    depower=0,
)
optimization_params = [
    # "az_amp0",
    "beta0",
    "beta_coeffs",
    "az_coeffs",
]
solution = reelout.run_simulation_opti(optimization_params=optimization_params)
phase, _ = reelout.run_simulation(run_plots=True)
print(phase.energy_metrics())

system_model.angle_pitch_tether = system_model.angle_pitch_tether + np.radians(4)
reelout = Reelout(
    system_model=system_model,
    pattern_config=REELOUT_CONFIG,
    depower=0,
)
optimization_params = [
    "az_coeffs",
    "beta0",
    "beta_coeffs",
]
solution = reelout.run_simulation_opti(optimization_params=optimization_params)
phase, _ = reelout.run_simulation(run_plots=True)
print(phase.energy_metrics())
plt.show()

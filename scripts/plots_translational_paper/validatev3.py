import h5py
import pandas as pd
import numpy as np
from picawe import SystemModel
from picawe.system.kite import Kite
import casadi as ca
import time


def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "data/flight_logs/"
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


results, flight_data, config_data = read_results("2019", "10", "08", "v3", addition="")
mask = flight_data.cycle.isin([64,65])
mask = mask & (flight_data.tether_reelout_speed > 0.2)
flight_data = flight_data[mask]
results = results[mask]
results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

csv_file = "./processed_data/VSM_results_alpha_sweep.csv"
v3_polar_data = pd.read_csv(csv_file)

# -----------------------------------------
# Define the system
# -----------------------------------------

import json

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# Example Usage
kite = Kite(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=20, steering_control="asymmetric")
kite_model = SystemModel(dof=3, quasi_steady=True, kite=kite)




solutions = []

start = time.time()
uf_window = []
vw_averaged = []
wdir_window = []

unknown_vars = [
    "tension_tether_ground",
    "input_steering",
    "speed_tangential",
]

# print(kite_model.tension_tether_ground)
sideslip_func = kite_model.extract_function("angle_sideslip")
# T_func = kite_model.extract_function("tension_tether")
solver_options = {
    "ipopt": {
        "print_level": 0,  # Suppresses IPOPT output
        # 'max_iter': 200,  # Maximum number of iterations
        "sb": "yes",  # Suppresses more detailed solver information
    },
    "print_time": False,  # Disables CasADi's internal timing output
}
# Sliding window size
window_size = 5
position = np.array(
    [results.kite_position_x, results.kite_position_y, results.kite_position_z]
).T
velocity = np.array(
    [results.kite_velocity_x, results.kite_velocity_y, results.kite_velocity_z]
).T
distance_radial = np.linalg.norm(position, axis=1)
speed_tangential = np.linalg.norm(velocity, axis=1)
qs_guess = [1e2, 0, 40]
kite_model.establish_residual()
solve_func, inputs_name = kite_model.solve_quasi_steady_state(
        unknown_vars, solver_options=solver_options
    )
print(solve_func)
for i, row in flight_data.iterrows():

    # Wind speed (vw) sliding window average
    uf_window.append(results.wind_speed_horizontal[i]*kite_model.kappa/np.log(results.kite_position_z[i]/kite_model.z0))
    wdir_window.append(results.wind_direction[i])
    if len(uf_window) > window_size:
        uf_window.pop(0)  # Keep the window size constant
        wdir_window.pop(0)
    print(i)
    uf = np.mean(uf_window)  # Compute the average of the current window


    current_state = {
        "distance_radial": distance_radial[i],
        "angle_course": row.kite_course,
        "speed_radial": row.tether_reelout_speed,
        "angle_azimuth": row.kite_azimuth,
        "angle_elevation": row.kite_elevation,
        "speed_friction": uf,
        "timeder_angle_course": row.kite_yaw_rate_1,
        "input_depower": row.up,
    }

    p = [current_state[name] for name in inputs_name]
    # print(p)
    lbx,ubx,lbg,ubg = kite_model.get_boundaries(unknown_vars)
    # print(lbx,ubx,lbg,ubg)
    sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    qs_guess = sol["x"]
    qs_state = {name: float(qs_guess[i]) for i, name in enumerate(unknown_vars)}

    state_combined = {**qs_state, **current_state}
    sideslip = sideslip_func(
        *[state_combined[name] for name in sideslip_func.name_in()]
    )
    state_combined["sideslip"] = float(sideslip)
    # state_combined["tension_tether"] = float(T)
    solutions.append(state_combined)
    # print(f"Solution: {solution}")



end = time.time()
print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")



# At dt of 0.1, the time taken is:
time_per_iteration = (end - start) / (len(flight_data))
print(f"Time per iteration: {time_per_iteration} seconds")

# Display the solutions

solutions_df = pd.DataFrame(solutions)
# Filter out rows where 'T' is None
solutions_df = solutions_df[solutions_df["tension_tether_ground"].notna()]

dt = 0.1
total_time = len(flight_data) * dt
# print('Estimated power: ', sum(solutions_df['T']*solutions_df['v_r']*dt)/total_time, 'W')
# print('Measured power: ', sum(flight_data['ground_tether_force']*flight_data['tether_reelout_speed']*dt)/total_time, 'W')


import matplotlib.pyplot as plt

plt.figure()
plt.plot(speed_tangential, label="tangential speed")
plt.plot(solutions_df["speed_tangential"], label="speed_tangential")
plt.legend()

plt.figure()
plt.plot(flight_data["ground_tether_force"], label="ground_tether_force")
plt.plot(solutions_df["tension_tether_ground"], label="estimated tension_tether")
plt.legend()

# plt.figure()
# plt.plot(np.degrees(solutions_df["angle_roll"]), label="roll tether-kite")
# plt.plot(np.degrees(solutions_df["angle_pitch"]), label="pitch tether-kite")
# plt.plot(np.degrees(solutions_df["angle_yaw"]), label="yaw tether-kite")
# plt.legend()
# plt.show()

# Plot sideslip
plt.figure()
plt.plot(np.degrees(solutions_df["sideslip"]), label="sideslip")
plt.legend()


# PLot steering input
plt.figure()
plt.plot(flight_data["us"], label="us measured")
plt.plot(solutions_df["input_steering"], label="u_s")
plt.legend()
# plt.show()


# plt.figure()
# plt.plot(results["wing_lift_coefficient"], label='wing_lift_coefficient')
# plt.plot(solutions_df['CL'], label='CL')
# plt.legend()
# # plt.show()

# plt.figure()
# plt.plot(results["wing_drag_coefficient"], label='wing_drag_coefficient')
# plt.plot(solutions_df['CD'], label='CD')
# plt.legend()
# plt.show()

total_power = (
    sum(solutions_df["tension_tether_ground"] * solutions_df["speed_radial"] * dt) / total_time
)
measured_power = (
    sum(flight_data["ground_tether_force"] * flight_data["tether_reelout_speed"] * dt)
    / total_time
)
print("Estimated power: ", total_power, "W")
print("Measured power: ", measured_power, "W")


fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
scatter = axs[0].scatter(
    solutions_df["angle_azimuth"],
    solutions_df["angle_elevation"],
    c=solutions_df["speed_tangential"],
    cmap="viridis",
    s=10,
    vmin = 10,
    vmax = 30
)  # `s` adjusts marker size

# Correct way to add a colorbar
cbar = fig.colorbar(scatter, ax=axs[0])
cbar.set_label("Tangential Speed [m/s]", fontsize=12)


scatter = axs[1].scatter(
    flight_data["kite_azimuth"],
    flight_data["kite_elevation"],
    c=speed_tangential,
    cmap="viridis",
    s=10,
    vmin = 10,
    vmax = 30
)  # `s` adjusts marker size
plt.show()

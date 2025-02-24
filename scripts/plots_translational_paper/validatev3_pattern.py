import h5py
import pandas as pd
import numpy as np
from picawe import SystemModel
import casadi as ca
import time
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.kinematics.parametrized_patterns import FigureEight
from picawe.system.kite import Kite
import matplotlib.pyplot as plt


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
mask = flight_data.cycle.isin([65])

mask = mask & (flight_data.tether_reelout_speed > 0.2)

flight_data = flight_data[mask]
results = results[mask]
results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

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

quasi_steady = True
# -----------------------------------------------
# Define the kite model
# -----------------------------------------------
kite = Kite(mass_wing=15, area_wing=19.75, aero_input=aero_input, steering_control="asymmetric")
kite_model = SystemModel(
    dof=3,
    quasi_steady=quasi_steady,
    kite=kite,
    wind_model="logarithmic",
)

# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
r0 = np.linalg.norm(
    [
        flight_data["kite_position_x"].iloc[0],
        flight_data["kite_position_y"].iloc[0],
        flight_data["kite_position_z"].iloc[0],
    ]
)
vr = np.mean(flight_data["tether_reelout_speed"])
kite_model.speed_radial = vr+1
kite_model.wind.speed_wind_ref = np.mean(results["wind_speed_horizontal"])-1
kite_model.input_depower = 0
pattern_config = {
    "pattern_type": "figure_eight",
    "initial_parameters": {
        "omega": -1.0,
        "r0": r0,
        "ry": 70,
        "rz": 50,
        "ky": 0.5,
        "kz": 1,
        "beta": np.radians(24),
        "kappa": 1,
        "vr": vr,   
    },
}

start_state = {
    "t": 0,
    "s": 0,
    "s_dot": 0.489,
    "s_ddot": 0,
    "tension_tether_ground": 1e3,
    "input_steering": 0,
    "angle_roll": 0,
    "angle_pitch": 0,
    "angle_yaw": 0,
}
time = np.arange(0, 100, 0.1)
dof = 3


# Run simulation
phase = PhaseParameterized(kite_model, quasi_steady=quasi_steady, pattern_config=pattern_config)
phase.run_simulation(start_state=start_state, time_array=time)
# Extract variables
s = phase.return_variable("s")
s_dot = phase.return_variable("s_dot")


fig, slider = phase.interactive_plot(animate=False)

vel_exp = np.array(
    [
        flight_data["kite_velocity_x"],
        flight_data["kite_velocity_y"],
        flight_data["kite_velocity_z"],
    ]
)[:,50:-50]
pos_exp = np.array(
    [
        flight_data["kite_position_x"],
        flight_data["kite_position_y"],
        flight_data["kite_position_z"],
    ]
)[:,50:-50]
print(vel_exp.shape)
plt.figure()
plt.plot(pos_exp[1,:]-np.mean(pos_exp[1,:]),pos_exp[2,:])
plt.plot(phase.return_variable("y"), phase.return_variable("z"))

# -----------------------------------------------
# Display results
# -----------------------------------------------

v_exp = np.linalg.norm(vel_exp, axis=0)
max_v_exp = max(v_exp)
max_v = np.max(phase.return_variable("speed_tangential"))
print(f"Max speed from flight data: {max_v_exp} m/s")
print(f"Max speed from simulation: {max_v} m/s")
print(f"Min speed from flight data: {np.min(v_exp)} m/s")
print(f"Min speed from simulation: {np.min(phase.return_variable('speed_tangential'))} m/s")
print(f"Mean speed from flight data: {np.mean(v_exp)} m/s")
print(f"Mean speed from simulation: {np.mean(phase.return_variable('speed_tangential'))} m/s")

print("----------------------------------------------------")

print(f"Mean exp lift coefficient: {np.mean(results['wing_lift_coefficient'])}")
print(f"Mean epx drag coefficient: {np.mean(results['wing_drag_coefficient'])}")
print(f"Mean lift coefficient: {np.mean(phase.return_variable('lift_coefficient'))}")
print(f"Mean drag coefficient: {np.mean(phase.return_variable('drag_coefficient'))}")
print(f"Mean exp angle of attack: {np.mean(results['wing_angle_of_attack_bridle'])}")
print(f"Mean angle of attack: {np.mean(phase.return_variable('angle_of_attack'))*180/np.pi}")

time_arr = np.array(results["time"][50:-50])
plt.figure()
plt.plot(phase.return_variable("t"), phase.return_variable("speed_tangential"))
plt.plot(time_arr-time_arr[0], v_exp)
plt.plot(time_arr-time_arr[0], flight_data["tether_reelout_speed"][50:-50])

plt.show()
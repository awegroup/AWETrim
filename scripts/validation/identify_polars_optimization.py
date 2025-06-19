import h5py
import pandas as pd
import numpy as np
from picawe import SystemModel
from picawe.system.kite import Kite
from picawe.system.tether import (
    RigidLumpedTether,
    FlexibleLumpedTether,
    RigidLinkTether,
)
from picawe.environment.Wind import Wind
import casadi as ca
import time
import matplotlib.pyplot as plt


def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "data/LEI-V9-KITE/flight_logs/"
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


results, flight_data, config_data = read_results("2023", "11", "27", "v9", addition="")
# mask = (flight_data.cycle>10)&(flight_data.cycle<70)
mask = flight_data.cycle.isin(range(15, 19))
mask = flight_data.cycle.isin(range(9, 16))
mask = flight_data.cycle == 10
# mask = mask & (flight_data.kite_elevation < 0.75)
mask = mask & (flight_data.up < 0.2)  # Filter out cycles with up > 0.5
# mask = mask & (flight_data.up > 0.9)  # Filter out cycles with up > 0.5
flight_data = flight_data[mask]
results = results[mask]
# flight_data = flight_data.iloc[500:700]
# results = results.iloc[500:700]
results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

csv_file = "./processed_data/VSM_results_alpha_sweep.csv"
v3_polar_data = pd.read_csv(csv_file)

if "kite_course" not in flight_data.columns:
    flight_data["kite_course"] = np.unwrap(flight_data["kite_yaw_0"])
# -----------------------------------------
# Define the system
# -----------------------------------------

import json

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/LEI-V9-KITE/v9_aero_input.json"
# file_path = "./data/v3_aero_input_identified.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

tether = RigidLumpedTether(diameter=0.014)
wind_model = Wind(
    wind_model="logarithmic",
    z0=0.01,  # Roughness length
)
# Example Usage
opti = ca.Opti()

CD0_ini = aero_input["params"]["CD0"]
angle_pitch_depower_0_ini = aero_input["params"]["angle_pitch_depower_0"]
delta_pitch_depower_ini = aero_input["params"]["delta_pitch_depower"]

aero_input["params"]["CD0"] = ca.SX.sym("CD0")
aero_input["params"]["angle_pitch_depower_0"] = ca.SX.sym(
    "angle_pitch_depower_0"
)  # Initial pitch angle for depower
aero_input["params"]["delta_pitch_depower"] = ca.SX.sym(
    "delta_pitch_depower"
)  # Change in pitch angle for depower
aero_model_inputs = [
    aero_input["params"]["CD0"],
    aero_input["params"]["angle_pitch_depower_0"],
    aero_input["params"]["delta_pitch_depower"],
]
opti_CD0 = opti.variable()
opti_angle_pitch_depower_0 = opti.variable()
opti_delta_pitch_depower = opti.variable()
kite = Kite(
    mass_wing=50 + 28,
    area_wing=47,
    aero_input=aero_input,
    mass_kcu=0,
    steering_control="asymmetric",
)

kite_model = SystemModel(
    dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind_model
)

N = len(flight_data)
opti_speed_tangential = opti.variable(N)
opti_tension_tether_ground = opti.variable(N)
opti_input_steering = opti.variable(N)

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
# sideslip_func = kite_model.extract_function("angle_sideslip")
# T_func = kite_model.extract_function("tension_tether")
solver_options = {
    "ipopt": {
        "print_level": 0,  # Suppresses IPOPT output
        # 'max_iter': 400,  # Maximum number of iterations
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
qs_guess = [1e5, 0, 60]
flight_data.kite_azimuth = (
    flight_data.kite_azimuth
)  # -0.1            # Calculate misalignment!!! at each cycle

residual = ca.Function(
    "residual",
    [
        kite_model.distance_radial,
        kite_model.angle_elevation,
        kite_model.angle_azimuth,
        kite_model.angle_course,
        kite_model.speed_radial,
        kite_model.speed_tangential,
        kite_model.input_steering,
        kite_model.tension_tether_ground,
        kite_model.input_depower,
        kite_model.wind.speed_friction,
        kite_model.timeder_angle_course,
    ]
    + aero_model_inputs,
    [kite_model.force_residual],
)

if "kite_yaw_rate_1" not in flight_data.columns:
    flight_data["kite_yaw_rate_1"] = np.gradient(
        np.unwrap(flight_data["kite_course"]), flight_data["time"]
    )

for column in flight_data.columns:
    if "100m" in column:
        print(f"Column {column} found in flight_data")

err = 0
for i, row in flight_data.iterrows():

    # Wind speed (vw) sliding window average
    uf_window.append(
        results.wind_speed_horizontal[i]
        * kite_model.wind.kappa
        / np.log(results.kite_position_z[i] / kite_model.wind.z0)
    )
    wdir_window.append(results.wind_direction[i])
    if len(uf_window) > window_size:
        uf_window.pop(0)  # Keep the window size constant
        wdir_window.pop(0)
        # print(uf_window)
    # print(i)

    uf = (
        row["100m_Wind_Speed_m_s"]
        * kite_model.wind.kappa
        / np.log(100 / kite_model.wind.z0)
    )

    residual_inputs = [
        distance_radial[i],
        row["kite_elevation"],
        row["kite_azimuth"],
        row["kite_course"],
        row["tether_reelout_speed"],
        opti_speed_tangential[i],
        opti_input_steering[i],
        opti_tension_tether_ground[i],
        row["up"],
        uf,
        row["kite_yaw_rate_1"],
        opti_CD0,
        opti_angle_pitch_depower_0,
        opti_delta_pitch_depower,
    ]
    res = residual(*residual_inputs)
    opti.subject_to(res[0] / 1e4 == 0)
    opti.subject_to(res[1] / 1e4 == 0)
    opti.subject_to(res[2] / 1e4 == 0)

    err += (
        (speed_tangential[i] - opti_speed_tangential[i]) / (speed_tangential[i] + 1e-6)
    ) ** 2
    err += (
        (flight_data.ground_tether_force[i] - opti_tension_tether_ground[i])
        / (flight_data.ground_tether_force[i] + 1e-6)
    ) ** 2  # Tether tension at ground


# Define the optimization objective: minimize RMSE between measured and optimized tangential speed

opti.minimize(err)
# Display the solutions

opti.solver(
    "ipopt",
    {
        "ipopt": {
            "max_iter": 1000,
            "bound_relax_factor": 0,
            "tol": 1e-3,  # Main tolerance
            "acceptable_iter": 3,  # Accept if solution is good for 3 iter
            "acceptable_tol": 1e-5,  # Acceptable early termination
            # "constr_viol_tol": 1e-6,  # Constraint violation tolerance
            # "dual_inf_tol": 1e-6,  # Dual infeasibility}#,"mu_init": 1e-2},
        }
        # "max_iter": 500,
        # "bound_relax_factor": 0,  # <--- critical
        # "mu_init": 1e-2,
        # "acceptable_tol": 1e-4
    },
)
opti.set_initial(opti_speed_tangential, speed_tangential)
opti.set_initial(opti_tension_tether_ground, flight_data.ground_tether_force + 1000)
opti.subject_to(opti_speed_tangential >= 0)  # Speed tangential lower limit
opti.subject_to(opti_speed_tangential <= 40)  # Speed tangential upper limit
opti.subject_to(opti_tension_tether_ground >= 200)  # Tension lower limit
opti.subject_to(opti_tension_tether_ground <= 100000)  # Tension upper limit
opti.set_initial(opti_CD0, CD0_ini)
opti.set_initial(opti_angle_pitch_depower_0, angle_pitch_depower_0_ini)
opti.set_initial(opti_delta_pitch_depower, delta_pitch_depower_ini)
opti.subject_to(opti_CD0 > 0.05)  # CD0 lower limit
opti.subject_to(opti_CD0 < 0.15)  # CD0 upper limit
opti.subject_to(
    opti_angle_pitch_depower_0 > -0.2
)  # Initial pitch angle for depower lower limit
opti.subject_to(
    opti_angle_pitch_depower_0 < 0.1
)  # Initial pitch angle for depower upper limit
opti.subject_to(opti_delta_pitch_depower > -0.3)
# opti.subject_to(
#     opti_delta_pitch_depower > -0.3
# )  # Change in pitch angle for depower limits
try:

    solver = opti.solve()
    print("Solver status:", solver.stats()["return_status"])

    print("Aerodynamic parameters:")
    print("CD0:", solver.value(opti_CD0))
    print("Angle Pitch Depower 0:", solver.value(opti_angle_pitch_depower_0))
    print("Delta Pitch Depower:", solver.value(opti_delta_pitch_depower))

    vtau = solver.value(opti_speed_tangential)
    tension_tether_ground = solver.value(opti_tension_tether_ground)
    plt.figure()
    plt.plot(flight_data.time, vtau, label="Optimized Tangential Speed")
    plt.plot(flight_data.time, speed_tangential, label="Measured Tangential Speed")

    plt.figure()
    plt.plot(flight_data.time, tension_tether_ground, label="Optimized Tension")
    plt.plot(
        flight_data.time, flight_data.ground_tether_force, label="Measured Tension"
    )
    plt.show()
except Exception as e:
    print("Solver failed with error:", e)
    print(opti.debug.value(opti_CD0))
    print(opti.debug.value(opti_angle_pitch_depower_0))
    print(opti.debug.value(opti_delta_pitch_depower))

    vtau = opti.debug.value(opti_speed_tangential)
    tension_tether_ground = opti.debug.value(opti_tension_tether_ground)
    plt.figure()
    plt.plot(flight_data.time, vtau, label="Optimized Tangential Speed")
    plt.plot(flight_data.time, speed_tangential, label="Measured Tangential Speed")

    plt.figure()
    plt.plot(flight_data.time, tension_tether_ground, label="Optimized Tension")
    plt.plot(
        flight_data.time, flight_data.ground_tether_force, label="Measured Tension"
    )
    plt.show()


# solver.set_initial(opti_input_steering, flight_data.us)

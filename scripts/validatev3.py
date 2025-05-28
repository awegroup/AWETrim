import h5py
import pandas as pd
import numpy as np
from picawe import SystemModel
from picawe.system.kite import Kite
from picawe.system.tether import RigidLumpedTether, FlexibleLumpedTether, RigidLinkTether
from picawe.environment.Wind import Wind
import casadi as ca
import time
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


results, flight_data, config_data = read_results("2019", "10", "08", "v3", addition="_va")
# mask = (flight_data.cycle>10)&(flight_data.cycle<70)
mask = flight_data.cycle.isin(range(63,66))
# mask = (flight_data.cycle==65)
# mask = mask & (flight_data.kite_elevation < 0.75) 
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
# file_path = "./data/v3_aero_input_identified.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

tether = RigidLumpedTether()
wind_model = Wind(
    wind_model="logarithmic",
    z0=0.01)
# Example Usage
kite = Kite(mass_wing=18, area_wing=20, aero_input=aero_input, mass_kcu=28, steering_control="asymmetric")
kite_model = SystemModel(dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind_model)
kite2 = Kite(mass_wing=43, area_wing=20, aero_input=aero_input, mass_kcu=0, steering_control="roll")
kite_model2 = SystemModel(dof=3, quasi_steady=True, kite=kite2, tether=tether, wind_model=wind_model)
# kite_model2 = kite_model


# print(kite_model.angle_pitch)

solutions = []
solutions2 = []

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
qs_guess = [1e5, 0, 20]
qs_guess2 = qs_guess
flight_data.kite_azimuth = flight_data.kite_azimuth#-0.1            # Calculate misalignment!!! at each cycle

# print(kite_model.tension_kite)
solve_func, inputs_name,_ = kite_model.setup_qs_solver(
        unknown_vars, solver_options=solver_options
    )

print(solve_func)
vtau = []
vtau2 = []
cl_func = kite_model.extract_function("lift_coefficient")
cd_func = kite_model.extract_function("drag_coefficient")
aoa_func = kite_model.extract_function("angle_of_attack")
tension_func = kite_model.extract_function("tension_tether_ground")
for i, row in flight_data.iterrows():

    # Wind speed (vw) sliding window average
    uf_window.append(results.wind_speed_horizontal[i]*kite_model.wind.kappa/np.log(results.kite_position_z[i]/kite_model.wind.z0))
    wdir_window.append(results.wind_direction[i])
    if len(uf_window) > window_size:
        uf_window.pop(0)  # Keep the window size constant
        wdir_window.pop(0)
    # print(i)
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
    lbx,ubx,lbg,ubg = kite_model.get_boundaries(current_state,unknown_vars)

    # qs_guess[0] = 1e5
    # qs_guess[2] = 20
    # print(lbx,ubx,lbg,ubg)
    sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    qs_guess = sol["x"]
    if np.linalg.norm(sol["g"]) < 1:
        qs_state = {name: float(qs_guess[i]) for i, name in enumerate(unknown_vars)}
        

        state_combined = {**qs_state, **current_state}


        state_combined["lift_coefficient"] = float(cl_func(
            *[state_combined[name] for name in cl_func.name_in()]
        ))
        state_combined["drag_coefficient"] = float(cd_func(
            *[state_combined[name] for name in cd_func.name_in()]
        ))
        state_combined["angle_of_attack"] = float(aoa_func(
            *[state_combined[name] for name in aoa_func.name_in()]
        ))
        state_combined["tension_tether_ground"] = float(tension_func(
            *[state_combined[name] for name in tension_func.name_in()]
        ))
        state_combined["time"] = row.time
        solutions.append(state_combined)
    else:
        print("Quasi steady solution not found")
        # continue


    
    # print(state_combined["tension_tether_ground"])
    # print(f"Solution: {solution}")


end = time.time()
print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")

solve_func2, inputs_name2,_ = kite_model2.setup_qs_solver(
        unknown_vars, solver_options=solver_options
    )
cl_func2 = kite_model2.extract_function("lift_coefficient")
cd_func2 = kite_model2.extract_function("drag_coefficient")
aoa_func2 = kite_model2.extract_function("angle_of_attack")
tension_func2 = kite_model2.extract_function("tension_kite")
for i, row in flight_data.iterrows():

    # Wind speed (vw) sliding window average
    uf_window.append(results.wind_speed_horizontal[i]*kite_model.wind.kappa/np.log(results.kite_position_z[i]/kite_model.wind.z0))
    wdir_window.append(results.wind_direction[i])
    if len(uf_window) > window_size:
        uf_window.pop(0)  # Keep the window size constant
        wdir_window.pop(0)
    # print(i)
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

    p = [current_state[name] for name in inputs_name2]
    # print(p)
    lbx,ubx,lbg,ubg = kite_model2.get_boundaries(current_state,unknown_vars)

    # print(lbx,ubx,lbg,ubg)
    sol2 = solve_func2(x0=qs_guess2, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    qs_guess2 = sol2["x"]

    qs_state2 = {name: float(sol2["x"][i]) for i, name in enumerate(unknown_vars)}

    state_combined2 = {**qs_state2, **current_state}
    
    state_combined2["lift_coefficient"] = float(cl_func2(
        *[state_combined2[name] for name in cl_func2.name_in()]
    ))
    state_combined2["drag_coefficient"] = float(cd_func2(
        *[state_combined2[name] for name in cd_func2.name_in()]
    ))
    state_combined2["angle_of_attack"] = float(aoa_func2(
        *[state_combined2[name] for name in aoa_func2.name_in()]
    ))
    state_combined2["tension_tether_ground"] = float(tension_func2(
        *[state_combined2[name] for name in tension_func2.name_in()]
    ))
    solutions2.append(state_combined2)






# At dt of 0.1, the time taken is:
time_per_iteration = (end - start) / (len(flight_data))
print(f"Time per iteration: {time_per_iteration} seconds")

# Display the solutions

solutions_df = pd.DataFrame(solutions)
solutions_df2 = pd.DataFrame(solutions2)
# Filter out rows where 'T' is None
solutions_df = solutions_df[solutions_df["tension_tether_ground"].notna()]

dt = 0.1
total_time = len(flight_data) * dt
# print('Estimated power: ', sum(solutions_df['T']*solutions_df['v_r']*dt)/total_time, 'W')
# print('Measured power: ', sum(flight_data['ground_tether_force']*flight_data['tether_reelout_speed']*dt)/total_time, 'W')

CD = results["wing_drag_coefficient"] + results["kcu_drag_coefficient"]+ results["bridles_drag_coefficient"]#+ results["tether_drag_coefficient"]
# Print mean CL and CD
mask_pow = (flight_data.up<0.1)&(flight_data.kite_elevation<0.75)
print("Mean CL powered, exp. data: ", np.mean(results["wing_lift_coefficient"][mask_pow]))
print("Mean CD powered, exp. data: ", np.mean(CD[mask_pow]))
print('Mean CL powered,  kcu: ', np.mean(solutions_df['lift_coefficient'][mask_pow]))
print('Mean CD powered,  kcu: ', np.mean(solutions_df['drag_coefficient'][mask_pow]))
print("Mean CL depowered, exp. data: ", np.mean(results["wing_lift_coefficient"][~mask_pow]))
print("Mean CD depowered, exp. data: ", np.mean(CD[~mask_pow]))
print('Mean CL depowered,  kcu: ', np.mean(solutions_df['lift_coefficient'][~mask_pow]))
print('Mean CD depowered,  kcu: ', np.mean(solutions_df['drag_coefficient'][~mask_pow]))
print('Mean CL,  no kcu: ', np.mean(solutions_df2['lift_coefficient'][mask_pow]))
print('Mean CD,  no kcu: ', np.mean(solutions_df2['drag_coefficient'][mask_pow]))


print("Mean aoa, exp. data: ", np.mean(results["wing_angle_of_attack_bridle"]))
print('Mean aoa powered,  kcu: ', np.mean(solutions_df['angle_of_attack'][mask_pow])*180/np.pi)
print('Mean aoa depowered,  kcu: ', np.mean(solutions_df['angle_of_attack'][~mask_pow])*180/np.pi)

mask = mask_pow

total_power = (
    sum(solutions_df["tension_tether_ground"][mask] * solutions_df["speed_radial"][mask] * dt) / total_time
)
total_power2 = (
    sum(solutions_df2["tension_tether_ground"][mask] * solutions_df2["speed_radial"][mask] * dt) / total_time
)
measured_power = (
    sum(flight_data["ground_tether_force"][mask] * flight_data["tether_reelout_speed"][mask] * dt)
    / total_time
)
print("Estimated power KCU reelout: ", total_power, "W")
print("Estimated power no KCU reelout: ", total_power2, "W")
print("Measured power reelout: ", measured_power, "W")

# -----------------------------------------
mask_dep = ~mask_pow
total_power_dep = (
    sum(solutions_df["tension_tether_ground"][mask_dep] * solutions_df["speed_radial"][mask_dep] * dt) / total_time
)
total_power2_dep = (
    sum(solutions_df2["tension_tether_ground"][mask_dep] * solutions_df2["speed_radial"][mask_dep] * dt) / total_time
)
measured_power_dep = (
    sum(flight_data["ground_tether_force"][mask_dep] * flight_data["tether_reelout_speed"][mask_dep] * dt)
    / total_time
)
print("Estimated power KCU depower: ", total_power_dep, "W")
print("Estimated power no KCU depower: ", total_power2_dep, "W")
print("Measured power depower: ", measured_power_dep, "W")


from picawe.utils.color_palette import set_plot_style
set_plot_style()
# Create figure with a custom grid layout
fig = plt.figure(figsize=(12, 6)) 

# Define grid layout (2 rows, 3 columns)
gs = fig.add_gridspec(6, 3, width_ratios=[1, 0.25,2], height_ratios=[1, 1,1,1,1,1])

# Left side subplots (square-like aspect ratio)
ax1 = fig.add_subplot(gs[:3, 0])  # Top-left
ax2 = fig.add_subplot(gs[3:6, 0])  # Bottom-left

# Right side subplots (time series)
ax3 = fig.add_subplot(gs[:2, 2])  # Top-right (spanning two columns)
ax4 = fig.add_subplot(gs[2:4, 2])  # Middle-right
ax5 = fig.add_subplot(gs[4:6, 2])  # Bottom-right


from picawe.utils.defaults import PLOT_LABELS
# add labels
ax1.set_ylabel(PLOT_LABELS["angle_elevation"])
ax2.set_xlabel(PLOT_LABELS["angle_azimuth"])
ax2.set_ylabel(PLOT_LABELS["angle_elevation"])
ax5.set_xlabel(PLOT_LABELS["phase"])
ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
ax4.set_ylabel(PLOT_LABELS["tension_tether_ground"])
ax5.set_ylabel(PLOT_LABELS["input_steering"])



vmin = min(np.min(solutions_df["speed_tangential"]), np.min(speed_tangential,))
vmax = max(np.max(solutions_df["speed_tangential"]), np.max(speed_tangential,))
scatter = ax1.scatter(
    solutions_df["angle_azimuth"],
    solutions_df["angle_elevation"],
    c=solutions_df["speed_tangential"],
    cmap="viridis",
    s=10,
    vmin = vmin,
    vmax = vmax
)  # `s` adjusts marker size

cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])  # Manually positioned colorbar
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label(PLOT_LABELS["speed_tangential"])
cbar.set_ticks(np.linspace(vmin, vmax, num=5))


scatter = ax2.scatter(
    flight_data["kite_azimuth"],
    flight_data["kite_elevation"],
    c=speed_tangential,
    cmap="viridis",
    s=10,
    vmin = vmin,
    vmax = vmax
)  # `s` adjusts marker size

ax3.plot(flight_data["time"], speed_tangential, label="Meas. $v_{\\tau}$")
ax3.plot(solutions_df["time"], solutions_df["speed_tangential"], label="QS $v_{\\tau}$, KCU")
ax3.plot(flight_data["time"], solutions_df2["speed_tangential"], label="QS $v_{\\tau}$, no KCU")

ax4.plot(flight_data["time"], flight_data["ground_tether_force"], label="Meas. $F_{t,g}$")
ax4.plot(solutions_df["time"], solutions_df["tension_tether_ground"], label="QS $F_{t,g}$, KCU")
ax4.plot(flight_data["time"], solutions_df2["tension_tether_ground"], label="QS $F_{t,g}$, no KCU")

ax5.plot(flight_data["time"], flight_data["kcu_actual_steering"]/100, label="Meas. $u_s$")
ax5.plot(solutions_df["time"], solutions_df["input_steering"], label="QS $u_s$, KCU")
ax5.plot(flight_data["time"], solutions_df2["input_steering"], label="QS $u_s$, no KCU")
ax5.set_ylim(-2, 2)

#Save the figure as pdf
save_folder = "./results/figures/translational_paper/"
plt.savefig(save_folder+"validation_v3_cycle65.pdf", bbox_inches='tight')

plt.show()
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


def compute_spherical_course(position, velocity):
    distance_radial = np.linalg.norm(position, axis=1)
    azimuth = np.arctan2(position[:, 0], position[:, 1])
    elevation = np.arcsin(position[:, 2] / distance_radial)

    e_r = position / distance_radial[:, np.newaxis]
    e_phi = np.stack(
        [-np.sin(azimuth), np.cos(azimuth), np.zeros_like(azimuth)], axis=1
    )
    e_theta = np.stack(
        [
            -np.cos(azimuth) * np.sin(elevation),
            -np.sin(azimuth) * np.sin(elevation),
            np.cos(elevation),
        ],
        axis=1,
    )

    vr_scalar = np.sum(velocity * e_r, axis=1)
    vtau = velocity - vr_scalar[:, np.newaxis]

    v_phi = np.sum(vtau * e_phi, axis=1)
    v_theta = np.sum(vtau * e_theta, axis=1)

    course_angle = np.arctan2(v_phi, v_theta)
    return course_angle


results, flight_data, config_data = read_results("2024", "02", "15", "v9", addition="")
# mask = (flight_data.cycle>10)&(flight_data.cycle<70)
mask = flight_data.cycle.isin(range(24, 26))
# mask = flight_data.cycle == 25
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
kite = Kite(
    mass_wing=60 + 28,
    area_wing=47,
    aero_input=aero_input,
    mass_kcu=0,
    steering_control="asymmetric",
)
kite_model = SystemModel(
    dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind_model
)
file_path = "./data/LEI-V9-KITE/v9_aero_input.json"
# file_path = "./data/v3_aero_input_identified.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)
# kite2 = Kite(
#     mass_wing=78,
#     area_wing=47,
#     aero_input=aero_input,
#     mass_kcu=0,
#     steering_control="roll",
# )
# kite_model2 = SystemModel(
#     dof=3, quasi_steady=True, kite=kite2, tether=tether, wind_model=wind_model
# )
# kite_model2 = kite_model


# print(kite_model.angle_pitch)

solutions = []
# solutions2 = []

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
qs_guess2 = qs_guess
flight_data.kite_azimuth = (
    flight_data.kite_azimuth
)  # -0.1            # Calculate misalignment!!! at each cycle

# print(kite_model.tension_kite)
solve_func, inputs_name, _ = kite_model.setup_qs_solver(
    unknown_vars, solver_options=solver_options
)

course_angle = compute_spherical_course(position, velocity)


flight_data["course_rate"] = np.gradient(
    np.unwrap(flight_data["kite_course"]), flight_data["time"]
)
# Smooth the course rate
flight_data["course_rate"] = (
    flight_data["course_rate"].rolling(window=window_size, min_periods=1).mean()
)
print(solve_func)
vtau = []
vtau2 = []
cl_func = kite_model.extract_function("lift_coefficient")
cd_func = kite_model.extract_function("drag_coefficient")
aoa_func = kite_model.extract_function("angle_of_attack")
tension_func = kite_model.extract_function("tension_tether_ground")
for column in flight_data.columns:
    if "100m" in column:
        print(f"Column {column} found in flight_data")
count = 0
wind_heights = []
for column in flight_data.columns:
    if "m_Wind_Speed_m_s" in column:
        wind_heights.append(int(column.split("m_")[0]))

# Main loop
for i, row in flight_data.iterrows():
    kite_height = row.kite_position_z if "kite_position_z" in row else 100
    closest_idx = np.argmin([abs(h - kite_height) for h in wind_heights])
    closest_height = wind_heights[closest_idx]
    wind_col = f"{closest_height}m_Wind_Speed_m_s"
    wind_speed = row[wind_col]

    uf = (
        wind_speed * kite_model.wind.kappa / np.log(closest_height / kite_model.wind.z0)
    )

    current_state = {
        "distance_radial": distance_radial[i],
        "angle_course": row.kite_course,
        "speed_radial": row.tether_reelout_speed,
        "angle_azimuth": row.kite_azimuth,
        "angle_elevation": row.kite_elevation,
        "speed_friction": uf,
        "timeder_angle_course": row.course_rate,
        "input_depower": row.up,
    }

    p = [current_state[name] for name in inputs_name]
    lbx, ubx, lbg, ubg = kite_model.get_boundaries(current_state, unknown_vars)
    sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    if np.linalg.norm(sol["g"]) < 1:
        qs_guess = sol["x"]
        qs_state = {name: float(qs_guess[i]) for i, name in enumerate(unknown_vars)}
        state_combined = {**qs_state, **current_state}
        state_combined["lift_coefficient"] = float(
            cl_func(*[state_combined[name] for name in cl_func.name_in()])
        )
        state_combined["drag_coefficient"] = float(
            cd_func(*[state_combined[name] for name in cd_func.name_in()])
        )
        state_combined["angle_of_attack"] = float(
            aoa_func(*[state_combined[name] for name in aoa_func.name_in()])
        )
        state_combined["tension_tether_ground"] = float(
            tension_func(*[state_combined[name] for name in tension_func.name_in()])
        )
        state_combined["time"] = row.time
        solutions.append(state_combined)
    else:
        print("Quasi steady solution not found")
        count += 1

print(f"Number of failed solutions: {count}/ {len(flight_data)}")
end = time.time()
print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")

solutions_df = pd.DataFrame(solutions)
solutions_df = solutions_df[solutions_df["tension_tether_ground"].notna()]

dt = 0.1
total_time = len(flight_data) * dt

CD = results["wing_drag_coefficient"]
mask_pow = (flight_data.up < 0.1) & (flight_data.kite_elevation < 0.75)
print(
    "Mean CL powered, exp. data: ", np.mean(results["wing_lift_coefficient"][mask_pow])
)
print("Mean CD powered, exp. data: ", np.mean(CD[mask_pow]))
print("Mean CL powered,  kcu: ", np.mean(solutions_df["lift_coefficient"][mask_pow]))
print("Mean CD powered,  kcu: ", np.mean(solutions_df["drag_coefficient"][mask_pow]))
print(
    "Mean CL depowered, exp. data: ",
    np.mean(results["wing_lift_coefficient"][~mask_pow]),
)
print("Mean CD depowered, exp. data: ", np.mean(CD[~mask_pow]))
print("Mean CL depowered,  kcu: ", np.mean(solutions_df["lift_coefficient"][~mask_pow]))
print("Mean CD depowered,  kcu: ", np.mean(solutions_df["drag_coefficient"][~mask_pow]))

print("Mean aoa, exp. data: ", np.mean(results["wing_angle_of_attack_bridle"]))
print(
    "Mean aoa powered,  kcu: ",
    np.mean(solutions_df["angle_of_attack"][mask_pow]) * 180 / np.pi,
)
print(
    "Mean aoa depowered,  kcu: ",
    np.mean(solutions_df["angle_of_attack"][~mask_pow]) * 180 / np.pi,
)

mask = mask_pow

total_power = (
    sum(
        solutions_df["tension_tether_ground"][mask]
        * solutions_df["speed_radial"][mask]
        * dt
    )
    / total_time
)
measured_power = (
    sum(
        flight_data["ground_tether_force"][mask]
        * flight_data["tether_reelout_speed"][mask]
        * dt
    )
    / total_time
)
print("Estimated power KCU reelout: ", total_power, "W")
print("Measured power reelout: ", measured_power, "W")

mask_dep = ~mask_pow
total_power_dep = (
    sum(
        solutions_df["tension_tether_ground"][mask_dep]
        * solutions_df["speed_radial"][mask_dep]
        * dt
    )
    / total_time
)
measured_power_dep = (
    sum(
        flight_data["ground_tether_force"][mask_dep]
        * flight_data["tether_reelout_speed"][mask_dep]
        * dt
    )
    / total_time
)
print("Estimated power KCU depower: ", total_power_dep, "W")
print("Measured power depower: ", measured_power_dep, "W")


from picawe.utils.color_palette import set_plot_style_no_latex
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from picawe.utils.defaults import PLOT_LABELS

set_plot_style_no_latex()


def plot_main_results(
    solutions_df,
    flight_data,
    speed_tangential,
    PLOT_LABELS,
    save_folder="./results/figures/translational_paper/",
    show=True,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[1:, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 1])

    ax1.set_xlabel(PLOT_LABELS["angle_azimuth"])
    ax1.set_ylabel(PLOT_LABELS["angle_elevation"])
    ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
    ax4.set_ylabel(PLOT_LABELS["tension_tether_ground"])
    ax5.set_ylabel(PLOT_LABELS["input_steering"])
    ax5.set_xlabel(PLOT_LABELS["phase"])

    vmin = min(np.min(solutions_df["speed_tangential"]), np.min(speed_tangential))
    vmax = max(np.max(solutions_df["speed_tangential"]), np.max(speed_tangential))

    scatter = ax1.scatter(
        solutions_df["angle_azimuth"],
        solutions_df["angle_elevation"],
        c=solutions_df["speed_tangential"],
        cmap="viridis",
        s=20,
        vmin=vmin,
        vmax=vmax,
    )

    cbar_ax = fig.add_axes([0.1, 0.82, 0.2, 0.03])
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(PLOT_LABELS["speed_tangential"])
    cbar.set_ticks(np.linspace(vmin, vmax, num=5))

    ax3.plot(flight_data["time"], speed_tangential, label="Meas. $v_{\\tau}$")
    ax3.plot(
        solutions_df["time"], solutions_df["speed_tangential"], label="QS $v_{\\tau}$"
    )

    ax4.plot(
        flight_data["time"], flight_data["ground_tether_force"], label="Meas. $F_{t,g}$"
    )
    ax4.plot(
        solutions_df["time"],
        solutions_df["tension_tether_ground"],
        label="QS $F_{t,g}$",
    )

    ax5.plot(
        flight_data["time"],
        flight_data["kcu_actual_steering"] / max(flight_data["kcu_actual_steering"]),
        label="Meas. $u_s$",
    )
    ax5.plot(solutions_df["time"], -solutions_df["input_steering"], label="QS $u_s$")

    for ax in [ax1, ax3, ax4, ax5]:
        ax.legend(loc="best", fontsize="small")

    plt.tight_layout()
    plt.savefig(save_folder + "validation_v9.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_coefficients_vs_time(solutions_df, show=True):
    import matplotlib.pyplot as plt

    fig2, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(
        solutions_df["time"], solutions_df["lift_coefficient"], label="QS $C_L$"
    )
    axs[0].set_ylabel("$C_L$")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(
        solutions_df["time"],
        solutions_df["drag_coefficient"],
        label="QS $C_D$",
        color="tab:orange",
    )
    axs[1].set_ylabel("$C_D$")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(
        solutions_df["time"],
        np.degrees(solutions_df["angle_of_attack"]),
        label="QS AoA",
        color="tab:green",
    )
    axs[2].set_ylabel("AoA (deg)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()
    axs[2].grid(True)

    fig2.suptitle("Lift, Drag Coefficient and Angle of Attack vs Time")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()
    plt.close(fig2)


# --- Plotting section ---
plot_main_results(
    solutions_df,
    flight_data,
    speed_tangential,
    PLOT_LABELS,
    save_folder="./results/figures/translational_paper/",
    show=True,
)
plot_coefficients_vs_time(solutions_df, show=True)

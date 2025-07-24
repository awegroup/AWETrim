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
# mask = (flight_data.cycle>10)&(flight_data.cycle<70)
mask = flight_data.cycle.isin(range(63, 72))
mask = flight_data.cycle == 65
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

# Preprocessing - calculate once for all simulations
position = np.array(
    [results.kite_position_x, results.kite_position_y, results.kite_position_z]
).T
velocity = np.array(
    [results.kite_velocity_x, results.kite_velocity_y, results.kite_velocity_z]
).T
distance_radial = np.linalg.norm(position, axis=1)
speed_tangential = np.linalg.norm(velocity, axis=1)

azimuth = np.arctan2(results.kite_position_y, results.kite_position_x)
flight_data["kite_azimuth"] = azimuth - results.wind_direction

# Normalize up between 0 and 1 which now is between 0.08 and 0.8
flight_data["up"] = (flight_data["up"] - flight_data["up"].min()) / (
    flight_data["up"].max() - flight_data["up"].min()
)

course_rate = np.gradient(np.unwrap(flight_data.kite_course), flight_data.time)

# Run simulation for both aerodynamic models
aero_files = [
    "./data/LEI-V3-KITE/v3_aero_input.json",
]
aero_labels = ["Variable"]
all_solutions = {}

for aero_file, label in zip(aero_files, aero_labels):
    print(f"\nRunning simulation with {aero_file}...")

    with open(aero_file, "r") as file:
        aero_input = json.load(file)

    tether = RigidLumpedTether(diameter=0.01)
    wind_model = Wind(wind_model="logarithmic", z0=0.1)
    kite = Kite(
        mass_wing=43,
        area_wing=20,
        aero_input=aero_input,
        mass_kcu=0,
        steering_control="asymmetric",
    )
    kite_model = SystemModel(
        dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind_model
    )

    # ...existing code...

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

    solver_options = {
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
        },
        "print_time": False,
    }

    window_size = 10
    qs_guess = [1e5, 0, 60]

    cl_func = kite_model.extract_function("lift_coefficient")
    cd_func = kite_model.extract_function("drag_coefficient")
    aoa_func = kite_model.extract_function("angle_of_attack")
    tension_func = kite_model.extract_function("tension_tether_ground")

    kite_model.setup_qs_solver(unknown_vars, solver_options=solver_options)

    uf = (
        results.wind_speed_horizontal
        * kite_model.wind.kappa
        / np.log(results.kite_position_z / kite_model.wind.z0)
    )
    uf_mean = np.mean(uf)

    # Run the simulation loop for this aerodynamic model
    for i, row in flight_data.iterrows():
        # ...existing simulation loop code...
        uf_window.append(
            results.wind_speed_horizontal[i]
            * kite_model.wind.kappa
            / np.log(results.kite_position_z[i] / kite_model.wind.z0)
        )
        wdir_window.append(results.wind_direction[i])
        if len(uf_window) > window_size:
            uf_window.pop(0)
            wdir_window.pop(0)

        uf = np.mean(uf_window)

        current_state = {
            "distance_radial": distance_radial[i],
            "angle_course": row.kite_course,
            "speed_radial": row.tether_reelout_speed,
            "angle_azimuth": row.kite_azimuth,
            "angle_elevation": row.kite_elevation,
            "speed_friction": uf,
            "timeder_angle_course": course_rate[i],
            "input_depower": row.up,
        }

        p = [current_state[name] for name in kite_model._qs_inputs]
        lbx, ubx, lbg, ubg = kite_model.get_boundaries(current_state, unknown_vars)

        sol = kite_model._qs_solver(
            x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg
        )

        if np.linalg.norm(sol["g"]) < 0.1:
            qs_guess = sol["x"]
            qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}

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
            qs_guess[0] = 1e5
            qs_guess[2] = 30
            print("Quasi steady solution not found")

    end = time.time()
    print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")

    # Store solutions for this aerodynamic model
    solutions_df = pd.DataFrame(solutions)
    solutions_df = solutions_df[solutions_df["tension_tether_ground"].notna()]
    all_solutions[label] = solutions_df

# Print comparison results for both models
for label, solutions_df in all_solutions.items():
    print(f"\n=== Results for {label} aerodynamic model ===")
    dt = 0.1
    total_time = len(flight_data) * dt

    CD = (
        results["wing_drag_coefficient"]
        + results["kcu_drag_coefficient"]
        + results["bridles_drag_coefficient"]
    )

    mask_pow = (flight_data.up < 0.1) & (flight_data.kite_elevation < 0.75)
    print(
        "Mean CL powered, exp. data: ",
        np.mean(results["wing_lift_coefficient"][mask_pow]),
    )
    print("Mean CD powered, exp. data: ", np.mean(CD[mask_pow]))
    print(
        "Mean CL powered,  kcu: ", np.mean(solutions_df["lift_coefficient"][mask_pow])
    )
    print(
        "Mean CD powered,  kcu: ", np.mean(solutions_df["drag_coefficient"][mask_pow])
    )
    print(
        "Mean CL depowered, exp. data: ",
        np.mean(results["wing_lift_coefficient"][~mask_pow]),
    )
    print("Mean CD depowered, exp. data: ", np.mean(CD[~mask_pow]))
    print(
        "Mean CL depowered,  kcu: ",
        np.mean(solutions_df["lift_coefficient"][~mask_pow]),
    )
    print(
        "Mean CD depowered,  kcu: ",
        np.mean(solutions_df["drag_coefficient"][~mask_pow]),
    )

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
from picawe.utils.defaults import PLOT_LABELS

set_plot_style_no_latex()


def plot_main_results_comparison(
    all_solutions,
    flight_data,
    speed_tangential,
    PLOT_LABELS,
    save_folder="./results/figures/translational_paper/",
    show=True,
):
    import matplotlib.pyplot as plt
    from picawe.utils.color_palette import get_color_list

    colors = get_color_list()

    # Figure 1: Dynamics (left panels)
    fig1 = plt.figure(figsize=(5, 6))
    gs1 = fig1.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    ax3 = fig1.add_subplot(gs1[0, 0])
    ax4 = fig1.add_subplot(gs1[1, 0])
    ax5 = fig1.add_subplot(gs1[2, 0])

    ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
    ax4.set_ylabel(PLOT_LABELS["tension_tether_ground"])
    ax5.set_ylabel(PLOT_LABELS["input_steering"])
    ax5.set_xlabel("Time (s)")

    # Plot measured data first
    ax3.plot(
        flight_data["time"],
        speed_tangential,
        label="Measured",
        color=colors[0],
    )
    ax4.plot(
        flight_data["time"],
        flight_data["ground_tether_force"],
        label="Measured",
        color=colors[0],
    )
    ax5.plot(
        flight_data["time"],
        flight_data["kcu_actual_steering"] / max(flight_data["kcu_actual_steering"]),
        label="Measured",
        color=colors[0],
    )

    # Plot results for aerodynamic models
    for i, (label, solutions_df) in enumerate(all_solutions.items()):
        color = colors[i + 1]

        ax3.plot(
            solutions_df["time"],
            solutions_df["speed_tangential"],
            label=f"QS {label}",
            color=color,
        )
        ax4.plot(
            solutions_df["time"],
            solutions_df["tension_tether_ground"],
            label=f"QS {label}",
            color=color,
        )
        ax5.plot(
            solutions_df["time"],
            -solutions_df["input_steering"],
            label=f"QS {label}",
            color=color,
        )

    # Add legends and grid for dynamics figure
    for ax in [ax3, ax4, ax5]:
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_folder + "validation_v3_dynamics.pdf", bbox_inches="tight")
    if show:
        plt.show()

    # Figure 2: Aerodynamics (right panels)
    fig2 = plt.figure(figsize=(8, 6))
    gs2 = fig2.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    ax6 = fig2.add_subplot(gs2[0, 0])
    ax7 = fig2.add_subplot(gs2[1, 0])
    ax8 = fig2.add_subplot(gs2[2, 0])

    ax6.set_ylabel("$C_L$")
    ax7.set_ylabel("$C_D$")
    ax8.set_ylabel("AoA (deg)")
    ax8.set_xlabel("Time (s)")

    # Plot measured aerodynamic data
    ax6.plot(
        flight_data["time"],
        results["wing_lift_coefficient"],
        label="Measured",
        color=colors[0],
    )
    ax7.plot(
        flight_data["time"],
        results["wing_drag_coefficient"]
        + results["kcu_drag_coefficient"]
        + results["bridles_drag_coefficient"],
        label="Measured",
        color=colors[0],
    )
    ax8.plot(
        flight_data["time"],
        results["wing_angle_of_attack_bridle"],
        label="Measured",
        color=colors[0],
    )

    # Plot aerodynamic coefficients for each model
    for i, (label, solutions_df) in enumerate(all_solutions.items()):
        color = colors[i + 1]

        ax6.plot(
            solutions_df["time"],
            solutions_df["lift_coefficient"],
            label=f"QS {label}",
            color=color,
        )
        ax7.plot(
            solutions_df["time"],
            solutions_df["drag_coefficient"],
            label=f"QS {label}",
            color=color,
        )
        ax8.plot(
            solutions_df["time"],
            np.degrees(solutions_df["angle_of_attack"]),
            label=f"QS {label}",
            color=color,
        )

    # Add legends and grid for aerodynamics figure
    for ax in [ax6, ax7, ax8]:
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_folder + "validation_v3_aerodynamics.pdf", bbox_inches="tight")
    if show:
        plt.show()


# --- Plotting section ---
plot_main_results_comparison(
    all_solutions,
    flight_data,
    speed_tangential,
    PLOT_LABELS,
    save_folder="./results/figures/translational_paper/",
    show=True,
)


plt.figure()
plt.plot(
    flight_data["time"],
    flight_data["up"],
    label="Measured Up",
)
plt.show()

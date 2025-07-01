import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.kinematics.Kinematics import ParametrizedKinematics, KiteKinematics
from picawe.system.kite import Kite
from picawe import SystemModel
import casadi as ca
import time as timet
import json
from picawe.utils.color_palette import (
    get_color_list,
    set_plot_style,
    set_plot_style_no_latex,
)

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
save_folder = "./results/figures/translational_paper/"
# set_plot_style()
# Define aerodynamic input
file_path = "./data/LEI-V3-KITE/v3_aero_input.json"
# file_path = "./data/rigid_kite.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

aero_input = {
    "model": "inviscid",
    "params": {
        "CD0": 0.05,
        "aspect_ratio": 10,
        "oswald_efficiency": 1,
        "angle_pitch_depower_0": 0,
        "delta_pitch_depower": 0,
    },
    "dependencies": {
        # "u_s": { "k_cl": 0, "k_cd": 0.0, "k_cs": 0.23, "k_cn": 0.005 },
    },
}
# -----------------------------------------------
# Define the state
# -----------------------------------------------
# kite_model.__bases__ = (KiteKinematics, Tether, Wind, RigidKite)
kite = Kite(
    mass_wing=20,
    area_wing=20,
    aero_input=aero_input,
    mass_kcu=0,
    steering_control="roll",
)
kite_model = SystemModel(dof=3, quasi_steady=True, kite=kite)

speed_wind = 12
kite_model.wind.speed_wind_ref = speed_wind

from picawe.system.system_model import State  # ensure you're importing State

unknown_vars = ["length_tether", "input_steering", "speed_tangential"]
kite_model.default_unknown_vars = unknown_vars  # optional if you want to persist it

angles_elevation = np.radians(np.linspace(0, 60, 10))
angles_course = np.linspace(0, 2 * np.pi, 100)
speed_radial = 0

elevation_states = []

for elevation in angles_elevation:
    states = []
    for course in angles_course:
        state_obj = State(
            distance_radial=200,
            angle_elevation=elevation,
            angle_azimuth=0,
            speed_radial=speed_radial,
            angle_course=course,
            timeder_angle_course=0,
            input_depower=0,
            input_steering=0,  # placeholder, to be solved
            speed_tangential=100,  # placeholder, to be solved
            length_tether=200,  # placeholder, to be solved
        )

        new_state = kite_model.solve_quasi_steady(state_obj, unknown_vars=unknown_vars)
        if new_state:
            states.append(new_state.to_dict())
        else:
            break  # exit inner loop on failure

    elevation_states.append(pd.DataFrame(states))


angles_azimuth = np.linspace(0, 60, 10) / 180 * np.pi
angles_course = np.linspace(0, 2 * np.pi, 100)

azimuth_states = []
for azimuth in angles_azimuth:
    states = []
    for course in angles_course:
        state_obj = State(
            distance_radial=200,
            angle_elevation=0,
            angle_azimuth=azimuth,
            speed_radial=speed_radial,
            angle_course=course,
            timeder_angle_course=0,
            input_depower=0,
            input_steering=0,  # placeholder, to be solved
            speed_tangential=100,  # placeholder, to be solved
            length_tether=200,  # placeholder, to be solved
        )

        new_state = kite_model.solve_quasi_steady(state_obj, unknown_vars=unknown_vars)
        if new_state:
            states.append(new_state.to_dict())
        else:
            break  # Exit inner loop if solver fails

    azimuth_states.append(pd.DataFrame(states))


# === First Figure ===
plt.figure(figsize=(5, 4))

X = []
Y = []
Z = []

for state_i in elevation_states:

    X.append(state_i["angle_course"] * 180 / np.pi)
    Y.append(state_i["speed_tangential"] / speed_wind)
    Z.append(state_i["angle_elevation"] * 180 / np.pi)

X, Y, Z = np.array(X), np.array(Y), np.array(Z)

contour = plt.contour(X, Y, Z, levels=5, colors="black")  # Contour lines
plt.clabel(contour, inline=True, fontsize=12, fmt="%.2f")  # Remove line at label

plt.xlabel(r"Course angle $\chi$ ($^\circ$)")
plt.ylabel("Tangential speed factor $\lambda$ (-)")

# Set x-ticks explicitly
plt.xticks([0, 90, 180, 270, 360])
# Save the figure as pdf
plt.tight_layout()
plt.savefig(save_folder + "tangential_factor_elevation.pdf")
plt.show()

# === Second Figure ===
plt.figure(figsize=(5, 4))

X = []
Y = []
Z = []
for state_i in azimuth_states:
    X.append(state_i["angle_course"] * 180 / np.pi)
    Y.append(state_i["speed_tangential"] / speed_wind)
    Z.append(state_i["angle_azimuth"] * 180 / np.pi)

X, Y, Z = np.array(X), np.array(Y), np.array(Z)

contour = plt.contour(X, Y, Z, levels=5, colors="black")  # Contour lines
plt.clabel(contour, inline=True, fontsize=12, fmt="%.2f")  # Remove line at label

plt.xlabel(r"Course angle $\chi$ ($^\circ$)")
plt.ylabel("Tangential speed factor $\lambda$ (-)")

# Set x-ticks explicitly
plt.xticks([0, 90, 180, 270, 360])
# Save the figure as pdf
plt.tight_layout()
plt.savefig(save_folder + "tangential_factor_azimuth.pdf")

plt.show()


speeds_radial = np.linspace(0, 10, 500)
angles_course = [np.pi / 2, 0, np.pi]

course_states = []

for course in angles_course:
    states = []
    for vr in speeds_radial:
        state_obj = State(
            distance_radial=200,
            angle_elevation=np.radians(0),
            angle_azimuth=0,
            speed_radial=vr,
            angle_course=course,
            timeder_angle_course=0,
            input_depower=0,
            input_steering=0,  # placeholder to be solved
            speed_tangential=100,  # placeholder to be solved
            length_tether=200,  # placeholder to be solved
        )

        new_state = kite_model.solve_quasi_steady(state_obj, unknown_vars=unknown_vars)
        if new_state:
            states.append(new_state.to_dict())
        else:
            break  # Exit if solver fails

    course_states.append(pd.DataFrame(states))


colors = get_color_list()
# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for i, state_i in enumerate(course_states):
    state_i = state_i[state_i["angle_of_attack"] * 180 / np.pi < 20]
    reeling_factor = state_i["speed_radial"] / speed_wind
    power_harvesting_factor = (
        state_i["tension_tether_ground"]
        * state_i["speed_radial"]
        / (0.5 * kite_model.rho * kite_model.area_wing * speed_wind**3)
    )
    max_idx = np.argmax(state_i["tension_tether_ground"] * state_i["speed_radial"])
    axs[0].plot(
        reeling_factor,
        power_harvesting_factor,
        label=f"$\chi$ = {np.degrees(state_i['angle_course'].iloc[0])}$^\circ$",
    )
    axs[0].axvline(reeling_factor[max_idx], linestyle="--", color=colors[i])
    axs[1].plot(reeling_factor, state_i["angle_of_attack"] * 180 / np.pi)
    axs[1].axvline(reeling_factor[max_idx], linestyle="--", color=colors[i])
axs[0].set_xlabel("Reeling factor $f$ (-)")
axs[0].set_ylabel(r"Power harvesting factor $\zeta$ (-)")
axs[1].set_xlabel("Reeling factor $f$ (-)")
axs[1].set_ylabel(r"Angle of attack $\alpha$ ($^\circ$)")
fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1),
    ncol=len(angles_course) // 2,
    frameon=True,
)
plt.xlim([0, 1])

# Save the figure as pdf
plt.tight_layout()
plt.savefig(save_folder + "reeling_factor_loyd.pdf")
plt.legend()


plt.figure()
for i, state_i in enumerate(course_states):
    max_idx = np.argmax(state_i["tension_tether_ground"] * state_i["speed_radial"])
    plt.plot(
        state_i["speed_radial"],
        state_i["lift_coefficient"] / state_i["drag_coefficient"],
        label=f"$\chi$ = {np.degrees(state_i['angle_course'].iloc[0])}",
    )
    plt.axvline(state_i["speed_radial"].iloc[max_idx], linestyle="--", color=colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("L/D")
plt.legend()
plt.show()


angles_course = [0]  # or as defined
speeds_radial = np.linspace(-5, 0, 100)
unknown_vars = ["length_tether", "input_steering", "angle_elevation"]

course_states = []

for course in angles_course:
    states = []
    for vr in speeds_radial:
        state_obj = State(
            distance_radial=200,
            speed_tangential=0,
            angle_azimuth=0,
            speed_radial=vr,
            angle_course=course,
            timeder_angle_course=0,
            input_depower=0,
            input_steering=0,  # to be solved
            length_tether=200,  # to be solved
            angle_elevation=np.radians(40),  # initial guess for solve
        )

        new_state = kite_model.solve_quasi_steady(state_obj, unknown_vars=unknown_vars)
        if new_state:
            print(np.degrees(new_state.angle_elevation))
            states.append(new_state.to_dict())
        else:
            continue  # optionally break instead if desired

    course_states.append(pd.DataFrame(states))

set_plot_style_no_latex()
colors = get_color_list()
tether_reel = 200
# Plot the results
plt.figure()
for i, state in enumerate(course_states):
    max_idx = np.argmin(state["tension_tether_ground"] * tether_reel)
    plt.plot(
        state["speed_radial"],
        state["tension_tether_ground"] * tether_reel / 3.6e3,
        label=f"Consumed power [Wh]",
    )
    plt.plot(
        state["speed_radial"], state["tension_tether_ground"], label=f"Tension [N]"
    )
    plt.axvline(state["speed_radial"].iloc[max_idx], linestyle="--", color=colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("Power [W]")
plt.legend()

plt.figure()
for i, state in enumerate(course_states):
    max_idx = np.argmin(state["tension_tether_ground"] * state["speed_radial"])
    plt.plot(
        state["speed_radial"],
        state["angle_of_attack"] * 180 / np.pi,
        label=f"$\chi$ = {np.degrees(state['angle_course'].iloc[0])}",
    )
    plt.axvline(state["speed_radial"].iloc[max_idx], linestyle="--", color=colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("Angle of attack [rad]")
plt.legend()


plt.figure()
for i, state in enumerate(course_states):
    max_idx = np.argmin(state["tension_tether_ground"] * state["speed_radial"])
    plt.plot(
        state["speed_radial"],
        state["lift_coefficient"] / state["drag_coefficient"],
        label=f"$\chi$ = {np.degrees(state['angle_course'].iloc[0])}",
    )
    plt.axvline(state["speed_radial"].iloc[max_idx], linestyle="--", color=colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("L/D")
plt.legend()


plt.figure()
for i, state in enumerate(course_states):
    max_idx = np.argmin(state["tension_tether_ground"] * state["speed_radial"])
    plt.plot(
        state["speed_radial"],
        state["angle_elevation"] * 180 / np.pi,
        label=f"$\chi$ = {np.degrees(state['angle_course'].iloc[0])}",
    )
    plt.axvline(state["speed_radial"].iloc[max_idx], linestyle="--", color=colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("Elevation angle [deg]")
plt.legend()

plt.show()

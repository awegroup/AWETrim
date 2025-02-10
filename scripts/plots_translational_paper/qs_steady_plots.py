import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.Kinematics import ParametrizedKinematics, KiteKinematics
from picawe import State
import casadi as ca
import time as timet
import json
from picawe.Tether import Tether
from picawe.Wind import Wind
from picawe.RigidKite import RigidKite
from picawe.color_palette import get_color_list, set_plot_style, set_plot_style_no_latex

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
save_folder = "./results/plots_point_mass/"
set_plot_style()
# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
# file_path = "./data/rigid_kite.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

aero_input =    {
        "model": "inviscid",
        "params": {
            "CD0": 0.05,
            "aspect_ratio": 10,
            "oswald_efficiency": 1,
            "angle_pitch_depower_0": 0,
        },
       "dependencies": {
        # "u_s": { "k_cl": 0, "k_cd": 0.0, "k_cs": 0.23, "k_cn": 0.005 },
    } 

    }
# -----------------------------------------------
# Define the state
# -----------------------------------------------
# State.__bases__ = (KiteKinematics, Tether, Wind, RigidKite)
state = State(mass_wing=80, area_wing=20, aero_input=aero_input, mass_kcu=0, dof=3, quasi_steady=True, steering_control="roll")

speed_wind = 10
state.speed_wind = speed_wind
state.input_depower = 0

unknown_vars = ["tension_tether_ground", "angle_roll", "speed_tangential"]


aoa_func = state.extract_function("angle_of_attack")    
CL_func = state.extract_function("lift_coefficient")
CD_func = state.extract_function("drag_coefficient")
solve_func, inputs_name = state.solve_quasi_steady_state(
        unknown_vars
    )

speed_radial = 0
angles_elevation = np.linspace(0,75,10)/180*np.pi
angles_course = np.linspace(0, 2*np.pi, 100)

elevation_states = []
for i, elevation in enumerate(angles_elevation):
    qs_guess = [1e5, 0, 100]
    states = []
    for j, course in enumerate(angles_course):
    
        current_state = {
            "distance_radial": 200,
            "angle_elevation": elevation,
            "angle_azimuth": 0,
            "speed_radial": speed_radial,
            "angle_course": course,
            "timeder_angle_course": 0,
        }
        p = [current_state[name] for name in inputs_name]

        lbx,ubx,lbg,ubg = state.get_boundaries(current_state)
        sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        qs_guess = sol["x"]
        if np.linalg.norm(sol["g"]) > 1e-6:
            print(sol["g"])
            break
        else:
            qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}

            
            full_state = {**current_state, **qs_state}
            full_state["angle_of_attack"] = float(aoa_func(*[full_state[name] for name in aoa_func.name_in()]))
            full_state["lift_coefficient"] = float(CL_func(*[full_state[name] for name in CL_func.name_in()]))
            full_state["drag_coefficient"] = float(CD_func(*[full_state[name] for name in CD_func.name_in()]))
            # print(full_state["angle_of_attack"])
            states.append(full_state)
    elevation_states.append(pd.DataFrame(states))

angles_azimuth = np.linspace(0,65,10)/180*np.pi
angles_course = np.linspace(0, 2*np.pi, 100)

azimuth_states = []
for i, azimuth in enumerate(angles_azimuth):
    qs_guess = [1e5, 0, 100]
    states = []
    for j, course in enumerate(angles_course):
    
        current_state = {
            "distance_radial": 200,
            "angle_elevation": 0,
            "angle_azimuth": azimuth,
            "speed_radial": speed_radial,
            "angle_course": course,
            "timeder_angle_course": 0,
        }
        p = [current_state[name] for name in inputs_name]

        lbx,ubx,lbg,ubg = state.get_boundaries(current_state)
        sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        qs_guess = sol["x"]
        if np.linalg.norm(sol["g"]) > 1e-6:
            print(sol["g"])
            break
        else:
            qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}

            
            full_state = {**current_state, **qs_state}
            full_state["angle_of_attack"] = float(aoa_func(*[full_state[name] for name in aoa_func.name_in()]))
            full_state["lift_coefficient"] = float(CL_func(*[full_state[name] for name in CL_func.name_in()]))
            full_state["drag_coefficient"] = float(CD_func(*[full_state[name] for name in CD_func.name_in()]))
            # print(full_state["angle_of_attack"])
            states.append(full_state)
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

contour = plt.contour(X, Y, Z, levels=5, colors='black')  # Contour lines
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

contour = plt.contour(X, Y, Z, levels=5, colors='black')  # Contour lines
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
angles_course = [np.pi/2,0,np.pi]

course_states = []
for i, course in enumerate(angles_course):
    qs_guess = [1e5, 0, 100]
    states = []
    for j, vr in enumerate(speeds_radial):
    
        current_state = {
            "distance_radial": 200,
            "angle_elevation": np.radians(0),
            "angle_azimuth": 0,
            "speed_radial": vr,
            "angle_course": course,
            "timeder_angle_course": 0,
        }
        p = [current_state[name] for name in inputs_name]

        lbx,ubx,lbg,ubg = state.get_boundaries(current_state)
        sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        qs_guess = sol["x"]
        if np.linalg.norm(sol["g"]) > 1e-6:
            print(sol["g"])
            break
        else:
            qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}

            
            full_state = {**current_state, **qs_state}
            full_state["angle_of_attack"] = float(aoa_func(*[full_state[name] for name in aoa_func.name_in()]))
            full_state["lift_coefficient"] = float(CL_func(*[full_state[name] for name in CL_func.name_in()]))
            full_state["drag_coefficient"] = float(CD_func(*[full_state[name] for name in CD_func.name_in()]))
            states.append(full_state)
    course_states.append(pd.DataFrame(states))


colors = get_color_list()
# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for i,state_i in enumerate(course_states):
    state_i = state_i[state_i["angle_of_attack"]*180/np.pi < 20]
    reeling_factor = state_i["speed_radial"]/speed_wind
    power_harvesting_factor = state_i["tension_tether_ground"]*state_i["speed_radial"]/(0.5*state.rho*state.area_wing*speed_wind**3)
    max_idx = np.argmax(state_i["tension_tether_ground"]*state_i["speed_radial"])
    axs[0].plot(reeling_factor, power_harvesting_factor, label=f"$\chi$ = {np.degrees(state_i['angle_course'].iloc[0])}$^\circ$")
    axs[0].axvline(reeling_factor[max_idx], linestyle='--', color = colors[i])
    axs[1].plot(reeling_factor, state_i["angle_of_attack"]*180/np.pi)
    axs[1].axvline(reeling_factor[max_idx], linestyle='--', color = colors[i])
axs[0].set_xlabel("Reeling factor $f$ (-)")
axs[0].set_ylabel(r"Power harvesting factor $\zeta$ (-)")
axs[1].set_xlabel("Reeling factor $f$ (-)")
axs[1].set_ylabel(r"Angle of attack $\alpha$ ($^\circ$)")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=len(angles_course)//2, frameon=True)
plt.xlim([0, 1])

#Save the figure as pdf
plt.tight_layout()
plt.savefig(save_folder + "reeling_factor_loyd.pdf")
plt.legend()


plt.figure()
for i,state_i in enumerate(course_states):
    max_idx = np.argmax(state_i["tension_tether_ground"]*state_i["speed_radial"])
    plt.plot(state_i["speed_radial"], state_i["lift_coefficient"]/state_i["drag_coefficient"], label=f"$\chi$ = {np.degrees(state_i['angle_course'].iloc[0])}")
    plt.axvline(state_i["speed_radial"].iloc[max_idx], linestyle='--', color = colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("L/D")
plt.legend()
plt.show()


unknown_vars = ["tension_tether_ground", "angle_roll", "angle_elevation"]


solve_func, inputs_name = state.solve_quasi_steady_state(
        unknown_vars
    )

speeds_radial = np.linspace(-10, 0, 500)
angles_course = [0]

course_states = []
for i, course in enumerate(angles_course):
    qs_guess = [1e3, 0, np.radians(80)]
    states = []
    for j, vr in enumerate(speeds_radial):
        current_state = {
            "distance_radial": 200,
            "speed_tangential": 0,
            "angle_azimuth": 0,
            "speed_radial": vr,
            "angle_course": course,
            "timeder_angle_course": 0,
        }
        p = [current_state[name] for name in inputs_name]

        lbx,ubx,lbg,ubg = state.get_boundaries(current_state)
        sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        qs_guess = sol["x"]
        if np.linalg.norm(sol["g"]) > 1:
            print(sol["g"])
            break
        else:
            qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}

            
            full_state = {**current_state, **qs_state}
            full_state["angle_of_attack"] = float(aoa_func(*[full_state[name] for name in aoa_func.name_in()]))
            full_state["lift_coefficient"] = float(CL_func(*[full_state[name] for name in CL_func.name_in()]))
            full_state["drag_coefficient"] = float(CD_func(*[full_state[name] for name in CD_func.name_in()]))
            states.append(full_state)
        print(j)
    course_states.append(pd.DataFrame(states))

set_plot_style_no_latex()
colors = get_color_list()
tether_reel = 200
# Plot the results
plt.figure()
for i,state in enumerate(course_states):
    max_idx = np.argmin(state["tension_tether_ground"]*tether_reel)
    plt.plot(state["speed_radial"], state["tension_tether_ground"]*tether_reel/3.6e3, label=f"Consumed power [Wh]")
    plt.plot(state["speed_radial"], state["tension_tether_ground"], label=f"Tension [N]")
    plt.axvline(state["speed_radial"].iloc[max_idx], linestyle='--', color = colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("Power [W]")
plt.legend()

plt.figure()
for i,state in enumerate(course_states):
    max_idx = np.argmin(state["tension_tether_ground"]*state["speed_radial"])
    plt.plot(state["speed_radial"], state["angle_of_attack"]*180/np.pi, label=f"$\chi$ = {np.degrees(state['angle_course'].iloc[0])}")
    plt.axvline(state["speed_radial"].iloc[max_idx], linestyle='--', color = colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("Angle of attack [rad]")
plt.legend()


plt.figure()
for i,state in enumerate(course_states):
    max_idx = np.argmin(state["tension_tether_ground"]*state["speed_radial"])
    plt.plot(state["speed_radial"], state["lift_coefficient"]/state["drag_coefficient"], label=f"$\chi$ = {np.degrees(state['angle_course'].iloc[0])}")
    plt.axvline(state["speed_radial"].iloc[max_idx], linestyle='--', color = colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("L/D")
plt.legend()


plt.figure()
for i,state in enumerate(course_states):
    max_idx = np.argmin(state["tension_tether_ground"]*state["speed_radial"])
    plt.plot(state["speed_radial"], state["angle_elevation"]*180/np.pi, label=f"$\chi$ = {np.degrees(state['angle_course'].iloc[0])}")
    plt.axvline(state["speed_radial"].iloc[max_idx], linestyle='--', color = colors[i])
plt.xlabel("Speed radial [m/s]")
plt.ylabel("Elevation angle [deg]")
plt.legend()

plt.show()






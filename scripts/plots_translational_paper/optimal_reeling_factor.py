import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.kinematics.Kinematics import ParametrizedKinematics
from picawe import SystemModel
import casadi as ca
import time as timet
import json
import os
from picawe.utils.color_palette import get_color_list, set_plot_style, set_plot_style_no_latex

save_folder = "./results/figures/translational_paper/"
# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
#Load initial state

file_path = "./results/impact_inertial_forces/"
file_name = "helix_quasi_steady.csv"
results = pd.read_csv(os.path.join(file_path, file_name))

set_plot_style()
# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
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
state = SystemModel(mass_wing=80, area_wing=20, aero_input=aero_input, mass_kcu=0, dof=3, quasi_steady=True, steering_control="roll", wind_model="uniform")

speed_wind = 10
state.speed_wind_ref = speed_wind
state.input_depower = 0
state.timeder_angle_course = 0
state.distance_radial = 200

solver_options = {
    "ipopt": {"print_level": 0, "sb": "yes","tol": 1e-8,},
    "print_time": False,
}

state.establish_residual()
variables = ca.symvar(state.residual)
name_vars = [var.name() for var in variables]
residual = ca.Function("residual", variables, [state.residual], name_vars, ["residual"])
print(residual)
angles_elevation = np.linspace(0, 75, 10)/180*np.pi
angles_course = np.linspace(0, 2*np.pi, 100)
f = np.zeros((len(angles_elevation), len(angles_course)))
ft = np.zeros((len(angles_elevation), len(angles_course)))
for i, angle_elevation in enumerate(angles_elevation):
    for j,angle_course in enumerate(angles_course):
        opti = ca.Opti()  
        reeling_factor = opti.variable()   
        speed_tangential = opti.variable()
        angle_roll = opti.variable()
        tension_tether = opti.variable()
        opti.subject_to(residual(angle_course = angle_course,
                                 angle_azimuth = 0,
                                    angle_elevation = angle_elevation,
                                    speed_tangential = speed_tangential,
                                    input_steering = angle_roll,
                                    speed_radial = reeling_factor*speed_wind,
                                    tension_tether_ground = tension_tether)["residual"] == 0)
        
        opti.subject_to(0 <= (reeling_factor <= 0.8))
        opti.set_initial(reeling_factor, 0.3)
        opti.set_initial(speed_tangential, 30)
        opti.set_initial(angle_roll, 0)
        opti.set_initial(tension_tether, 1000)
        opti.minimize(-tension_tether*reeling_factor*speed_wind)
        opti.solver("ipopt", solver_options)
        sol = opti.solve()
        f[i,j] = sol.value(reeling_factor)
        ft[i,j] = sol.value(tension_tether)
        # print(f[i,j])


X, Z = np.meshgrid(np.degrees(angles_course), np.degrees(angles_elevation))

Y = f

fig = plt.figure(figsize=(5, 4))

contour = plt.contour(X, Y, Z, levels=5, colors='black')  # Contour lines
plt.clabel(contour, inline=True, fontsize=12, fmt="%.2f")  # Remove line at label
# Set x-ticks explicitly
plt.xticks([0, 90, 180, 270, 360])
plt.xlabel(r"Course angle $\chi$ ($^\circ$)")
plt.ylabel(r"Optimal reeling factor $f$ (-)")
# Save the figure as pdf
plt.tight_layout()
plt.savefig(save_folder + "optimal_reeling_factor_elevation.pdf")
# plt.show()


angles_azimuth = np.linspace(0, 65, 10)/180*np.pi
angles_course = np.linspace(0, 2*np.pi, 100)
f_az = np.zeros((len(angles_elevation), len(angles_course)))
ft_az = np.zeros((len(angles_elevation), len(angles_course)))
for i, angle_azimuth in enumerate(angles_azimuth):
    for j,angle_course in enumerate(angles_course):
        opti = ca.Opti()  
        reeling_factor = opti.variable()   
        speed_tangential = opti.variable()
        angle_roll = opti.variable()
        tension_tether = opti.variable()
        opti.subject_to(residual(angle_course = angle_course,
                                 angle_azimuth = angle_azimuth,
                                    angle_elevation = 0,
                                    speed_tangential = speed_tangential,
                                    input_steering = angle_roll,
                                    speed_radial = reeling_factor*speed_wind,
                                    tension_tether_ground = tension_tether)["residual"] == 0)
        
        opti.subject_to(0 <= (reeling_factor <= 0.8))
        opti.set_initial(reeling_factor, 0.3)
        opti.set_initial(speed_tangential, 30)
        opti.set_initial(angle_roll, 0)
        opti.set_initial(tension_tether, 1000)
        opti.minimize(-tension_tether*reeling_factor*speed_wind)
        opti.solver("ipopt", solver_options)
        sol = opti.solve()
        f_az[i,j] = sol.value(reeling_factor)
        ft_az[i,j] = sol.value(tension_tether)
        # print(f[i,j])


X, Z = np.meshgrid(np.degrees(angles_course), np.degrees(angles_azimuth))

Y = f
Y1 = ft/(0.5*1.225*speed_wind**2/f*state.area_wing)

print(max(ft[0,:]))

fig = plt.figure(figsize=(5, 4))

contour = plt.contour(X, Y, Z, levels=5, colors='black')  # Contour lines
contour1 = plt.contour(X, Y1, Z, levels=5, colors='red')  # Contour lines
plt.clabel(contour, inline=True, fontsize=12, fmt="%.2f")  # Remove line at label
plt.clabel(contour1, inline=True, fontsize=12, fmt="%.2f")  # Remove line at label
# Set x-ticks explicitly
plt.xticks([0, 90, 180, 270, 360])
plt.xlabel(r"Course angle $\chi$ ($^\circ$)")
plt.ylabel(r"Optimal reeling factor $f$ (-)")
# Save the figure as pdf
plt.tight_layout()
plt.savefig(save_folder + "optimal_reeling_factor_azimuth.pdf")


colors = get_color_list()
plt.figure()
plt.scatter(f[:,0]*speed_wind, ft[:,0], label = f"$\chi = {np.degrees(angles_course[0]):.1f}$")
plt.scatter(f[:,25]*speed_wind, ft[:,25], label = f"$\chi = {np.degrees(angles_course[25]):.1f}$")
plt.scatter(f[:,50]*speed_wind, ft[:,50], label = f"$\chi = {np.degrees(angles_course[50]):.1f}$")
plt.scatter(f_az[:,0]*speed_wind, ft_az[:,0], color = colors[0])
plt.scatter(f_az[:,25]*speed_wind, ft_az[:,25], color = colors[1])
plt.scatter(f_az[:,50]*speed_wind, ft_az[:,50], color = colors[2])

plt.legend()
plt.xlabel(r"Optimal reeling factor $f$ (-)")
plt.ylabel(r"Optimal tension $T$ (-)")
plt.show()